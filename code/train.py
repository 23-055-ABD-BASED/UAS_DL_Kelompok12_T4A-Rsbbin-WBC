"""
Pelatihan + evaluasi satu run. Dipakai SEMUA anggota.

Satu panggilan run_experiment(...) = satu run = satu baris di logs/log_<owner>.csv.
Faktor ablasi diubah lewat argumen fungsi; sisanya tetap pada baseline:
    U-Net ResNet-34 (ImageNet), loss CE+Dice, split 'grouped',
    augmentasi warna aktif, resolusi 256, 20 epoch, seed 42.

Aturan penting (Kisi-Kisi 3.2):
  * Semua pemilihan epoch/hiperparameter memakai set VALIDASI.
  * Set uji HANYA dipakai lewat eval_test=True, sekali, untuk konfigurasi final.
"""
import os, sys, copy, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

DRIVE = "/content/drive/MyDrive/PROYEK UAS DEEP LEARNING"
sys.path.append(f"{DRIVE}/code")
from data import load_split, WBCSeg, _path, MEAN, STD      # noqa: E402
from runlog import RunLog                                  # noqa: E402
from models import build_model, SegLoss                    # noqa: E402
from metrics import (per_image_metrics, summarize, select_score, predict_labels,  # noqa: E402
                     count_params_m, measure_latency_ms)

torch.backends.cudnn.benchmark = True   # lebih cepat; hasil bisa sedikit beda antar-run walau seed sama


# --------------------------------------------------------------------------- data
def load_mask_full(name):
    """Masker pada resolusi ASLI: 0 latar, 1 sitoplasma (255), 2 nukleus (100)."""
    m = np.array(Image.open(_path(name, True)))
    if m.ndim == 3:
        m = m[..., 0]
    y = np.zeros(m.shape, np.uint8)
    y[m == 255] = 1
    y[m == 100] = 2
    return y


class WBCSegEval(Dataset):
    """Untuk evaluasi: citra di-resize ke `size`, masker TETAP resolusi asli.
    Dengan begitu skor 256 dan 384 dihitung pada grid piksel yang sama."""
    def __init__(self, df, size):
        self.names = df["name"].tolist()
        self.tf = A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD), ToTensorV2()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        x = np.array(Image.open(_path(n, False)).convert("RGB"))
        return self.tf(image=x)["image"], torch.from_numpy(load_mask_full(n)).long(), n


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------- evaluasi
def evaluate(model, df, size, device, with_hd95=False, batch_size=8, num_workers=2):
    """Metrik per citra untuk seluruh df -> list of dict (satu dict per citra)."""
    dl = DataLoader(WBCSegEval(df, size), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(device.type == "cuda"))
    cls_of = dict(zip(df["name"], df["cls"]))
    rows = []
    for x, y, names in dl:
        pred = predict_labels(model, x, tuple(y.shape[-2:]), device)
        for p, g, n in zip(pred, y.numpy(), names):
            r = per_image_metrics(p, g, with_hd95)
            r["name"], r["cls"] = n, cls_of[n]
            rows.append(r)
    return rows


# ---------------------------------------------------------------------- 1 run
def run_experiment(owner, ablation, factor_value,
                   arch="unet", encoder="resnet34", loss="ce_dice",
                   img_size=256, seed=42, split="grouped",
                   epochs=20, lr=1e-3, batch_size=16,
                   color_aug=True, pretrained=True, patience=5,
                   eval_test=False, save_path=None, num_workers=2):
    """Latih + evaluasi satu konfigurasi, catat ke log. Mengembalikan dict ringkasan.

    owner         : nama anggota (menentukan file log_<owner>.csv)
    ablation      : 'baseline' | 'split' | 'loss' | 'color_aug' | 'resolution' | ...
    factor_value  : nilai faktor yang diuji (mis. 'CE+Dice', 'on', '384')
    eval_test     : True hanya untuk konfigurasi FINAL, sekali di akhir.
    save_path     : bila diisi, bobot terbaik disimpan ke path itu (jangan semua run,
                    1 checkpoint ~100 MB).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}
    try:
        with RunLog(owner, ablation=ablation, factor_value=str(factor_value), arch=arch,
                    encoder=encoder, loss=loss, img_size=img_size, seed=seed, split=split,
                    epochs=epochs, lr=lr, batch_size=batch_size, color_aug=color_aug,
                    pretrained=pretrained, used_test=eval_test) as run:
            set_seed(seed)
            tr, va = load_split(split, "train"), load_split(split, "val")
            gen = torch.Generator()
            gen.manual_seed(seed)
            dl_tr = DataLoader(WBCSeg(tr, train=True, size=img_size, color_aug=color_aug),
                               batch_size=batch_size, shuffle=True, drop_last=True,
                               num_workers=num_workers, generator=gen,
                               pin_memory=(device.type == "cuda"))

            model = build_model(arch, encoder, pretrained).to(device)
            crit = SegLoss(loss).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            use_amp = device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            best, best_ep, best_state, bad, ep_run = -1.0, 0, None, 0, 0
            for ep in range(1, epochs + 1):
                model.train()
                tot, nb = 0.0, 0
                for x, y, _ in dl_tr:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        logits = model(x)
                    l = crit(logits.float(), y)
                    scaler.scale(l).backward()
                    scaler.step(opt)
                    scaler.update()
                    tot += l.item()
                    nb += 1
                sched.step()
                ep_run = ep
                val = summarize(evaluate(model, va, img_size, device, num_workers=num_workers), "val_")
                score = select_score(val)
                print(f"ep {ep:2d}/{epochs} | loss {tot / nb:.4f} | val dice cyto {val['val_dice_cyto']:.4f}"
                      f" nuc {val['val_dice_nuc']:.4f} | skor {score:.4f}")
                if score > best:
                    best, best_ep, bad = score, ep, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        print(f"early stopping di epoch {ep} (terbaik: epoch {best_ep})")
                        break

            model.load_state_dict(best_state)          # kembali ke epoch terbaik (berdasar VALIDASI)
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)

            # metrik validasi final (dengan HD95) + efisiensi
            val = summarize(evaluate(model, va, img_size, device, with_hd95=True,
                                     num_workers=num_workers), "val_")
            run.metrics(**val, best_epoch=best_ep, epochs_run=ep_run,
                        params_m=count_params_m(model),
                        latency_gpu_ms=measure_latency_ms(model, img_size, device))
            summary = dict(val, run_id=run.row["run_id"], best_epoch=best_ep)

            # set uji: HANYA untuk konfigurasi final
            if eval_test:
                os.makedirs(f"{DRIVE}/results", exist_ok=True)
                parts = ["test_insession", "test"] if split == "cross_session" else ["test"]
                for p in parts:
                    rows = evaluate(model, load_split(split, p), img_size, device,
                                    with_hd95=True, num_workers=num_workers)
                    res = pd.DataFrame(rows)
                    res.to_csv(f"{DRIVE}/results/{run.row['run_id']}_{p}_per_image.csv", index=False)
                    tm = summarize(rows, f"{p}_")
                    run.metrics(**tm)
                    summary.update(tm)
                    print(f"\n[{p}] Dice per jenis sel:")
                    print(res.groupby("cls")[["dice_cyto", "dice_nuc"]].agg(["mean", "std", "count"]).round(3))
                cpu_model = copy.deepcopy(model).cpu().float()
                run.metrics(latency_cpu_ms=measure_latency_ms(cpu_model, img_size, torch.device("cpu"),
                                                              n=50, warmup=5))
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summary


def run_seeds(seeds=(42, 43, 44), **kw):
    """Ulangi satu konfigurasi dengan beberapa seed -> tabel rerata +- SD.
    Untuk perbandingan utama (dua konfigurasi terbaik), sesuai Kisi-Kisi 3.2."""
    res = [run_experiment(seed=s, **kw) for s in seeds]
    df = pd.DataFrame(res)
    cols = [c for c in df.columns if c.startswith(("val_", "test_")) and not c.endswith("_sd")]
    tab = pd.DataFrame({"mean": df[cols].mean(), "sd": df[cols].std(ddof=1)}).round(4)
    print(f"\nRingkasan {len(seeds)} seed:")
    print(tab)
    return df, tab
