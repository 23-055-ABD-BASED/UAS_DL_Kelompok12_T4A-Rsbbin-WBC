"""
Metrik evaluasi segmentasi semantik, dihitung PER CITRA.
Pemegang: arifin (evaluasi & efisiensi).

Kelas masker: 0 = latar, 1 = sitoplasma, 2 = nukleus sesuai data.py.

  Kisi-Kisi 3.3 meminta Dice/IoU dihitung per citra lalu dirata-ratakan
  (rerata +- simpangan baku). Agregat piksel membuat sel besar mendominasi
  skor dan menyembunyikan kegagalan pada sel kecil (mis. limfosit).
"""
import time
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion

CLASS_NAMES = {0: "bg", 1: "cyto", 2: "nuc"}


def dice_iou(pred, gt, cls):
    """Dice dan IoU satu kelas pada satu citra. Mengembalikan (nan, nan) bila
    kelas itu tidak ada di prediksi maupun di gt (tidak ada yang dinilai)."""
    p, g = (pred == cls), (gt == cls)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    s = p.sum() + g.sum()
    if s == 0:
        return np.nan, np.nan
    return 2.0 * inter / s, inter / union


def hd95(pred, gt, cls):
    """Hausdorff distance persentil ke-95 (satuan: piksel) antara permukaan
    prediksi dan gt untuk satu kelas. nan bila salah satunya kosong."""
    a, b = (pred == cls), (gt == cls)
    if not a.any() or not b.any():
        return np.nan
    sa = a ^ binary_erosion(a)          # piksel permukaan prediksi
    sb = b ^ binary_erosion(b)          # piksel permukaan gt
    d_a_to_b = distance_transform_edt(~sb)[sa]
    d_b_to_a = distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([d_a_to_b, d_b_to_a]), 95))


def per_image_metrics(pred, gt, with_hd95=False):
    """Semua metrik satu citra -> dict. pred dan gt: array 2D int (H, W)."""
    out = {}
    for c, name in CLASS_NAMES.items():
        d, i = dice_iou(pred, gt, c)
        out[f"dice_{name}"], out[f"iou_{name}"] = d, i
    out["miou"] = float(np.nanmean([out["iou_bg"], out["iou_cyto"], out["iou_nuc"]]))
    if with_hd95:
        out["hd95_cyto"] = hd95(pred, gt, 1)
        out["hd95_nuc"] = hd95(pred, gt, 2)
    return out


def summarize(rows, prefix="val_"):
    """Rerata dan SD (nanmean/nanstd) seluruh citra -> dict siap untuk RunLog.
    rows: list of dict dari per_image_metrics."""
    keys = [k for k in rows[0].keys() if k not in ("name", "cls")]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[f"{prefix}{k}"] = round(float(np.nanmean(v)), 4)
        if k in ("dice_cyto", "dice_nuc", "miou", "hd95_cyto", "hd95_nuc"):
            out[f"{prefix}{k}_sd"] = round(float(np.nanstd(v)), 4)
    return out


def select_score(summary, prefix="val_"):
    """Skor untuk memilih epoch terbaik: rerata Dice sitoplasma & nukleus."""
    return 0.5 * (summary[f"{prefix}dice_cyto"] + summary[f"{prefix}dice_nuc"])


@torch.no_grad()
def predict_labels(model, x, out_hw, device, amp=True):
    """Logit -> di-resize ke ukuran masker asli -> argmax. Evaluasi SELALU
    pada resolusi masker asli supaya ablasi resolusi (256 vs 384) adil."""
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(amp and device.type == "cuda")):
        logits = model(x.to(device, non_blocking=True))
    logits = torch.nn.functional.interpolate(
        logits.float(), size=out_hw, mode="bilinear", align_corners=False)
    return logits.argmax(1).cpu().numpy()


def count_params_m(model):
    """Jumlah parameter dalam juta."""
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 3)


@torch.no_grad()
def measure_latency_ms(model, size, device, n=50, warmup=10):
    """Latensi per citra (batch 1) dalam ms, setelah warm-up, rerata n citra.
    Sesuai Kisi-Kisi 3.3: warm-up dulu, batch 1, minimal 50 citra."""
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / n * 1000, 2)
