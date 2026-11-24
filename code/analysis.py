"""
analysis.py  --  Analisis kesalahan dan interpretabilitas (Grad-CAM) untuk segmentasi.
Pemegang: Anggota 4.

Dua alat:
  1) categorize()  : mengelompokkan kesalahan satu citra ke kategori berbasis ATURAN
                     (heuristik, bukan label manusia; hasilnya harus dicek visual).
  2) seg_gradcam() : Grad-CAM untuk segmentasi (Seg-Grad-CAM). Skor yang diturunkan
                     adalah jumlah logit kelas target pada area yang DIPREDIKSI model,
                     sehingga peta menjelaskan "mengapa model menandai area ini".
Kisi-Kisi 3.4 menerima "Grad-CAM, EigenCAM, atau sejenisnya".
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import _path, MEAN, STD
from models import build_model
from train import load_mask_full

CAT_NAMES = {
    "sitoplasma_berlebih": "Sitoplasma berlebih (latar dilabeli sitoplasma)",
    "sitoplasma_kurang": "Sitoplasma kurang (sitoplasma dilabeli latar)",
    "inti_sitoplasma_tertukar": "Inti dan sitoplasma tertukar",
    "inti_vs_latar": "Inti vs latar (inti hilang atau berlebih)",
    "tanpa_kesalahan": "Tanpa kesalahan piksel",
}


def load_model(arch, ckpt_path, device):
    """Bangun arsitektur (tanpa mengunduh bobot ImageNet) lalu isi dengan bobot final."""
    m = build_model(arch, pretrained=False)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.to(device).eval()


def get_sample(name, size=256):
    """Citra asli (uint8), masker gt resolusi asli (0/1/2), dan tensor masukan (1,3,size,size)."""
    img = np.array(Image.open(_path(name, False)).convert("RGB"))
    gt = load_mask_full(name)
    tf = A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD), ToTensorV2()])
    x = tf(image=img)["image"].unsqueeze(0)
    return img, gt, x


@torch.no_grad()
def predict(model, x, out_hw, device):
    """Label prediksi (H, W) pada resolusi masker asli (logit di-resize bilinear)."""
    logits = model(x.to(device))
    logits = F.interpolate(logits.float(), size=out_hw, mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def categorize(gt, pred, extra_frac=0.10):
    """Kategori kesalahan utama = jenis kesalahan piksel TERBANYAK.
    Juga menandai 'objek ekstra': komponen terhubung pada prediksi yang tidak
    menyentuh sel gt dan luasnya >= extra_frac x luas sel gt (mis. sel tetangga).
    Mengembalikan (kategori, dict jumlah piksel, flag_objek_ekstra)."""
    types = {
        "sitoplasma_berlebih": int(((pred == 1) & (gt == 0)).sum()),
        "sitoplasma_kurang": int(((gt == 1) & (pred == 0)).sum()),
        "inti_sitoplasma_tertukar": int((((gt == 2) & (pred == 1)) | ((gt == 1) & (pred == 2))).sum()),
        "inti_vs_latar": int((((gt == 2) & (pred == 0)) | ((gt == 0) & (pred == 2))).sum()),
    }
    cat = "tanpa_kesalahan" if sum(types.values()) == 0 else max(types, key=types.get)
    lab, n = ndimage.label(pred > 0)
    gt_cell = gt > 0
    extra = sum(int((lab == i).sum()) for i in range(1, n + 1) if not ((lab == i) & gt_cell).any())
    return cat, types, bool(extra >= extra_frac * max(int(gt_cell.sum()), 1))


def seg_gradcam(model, x, target_class, pred_mask, device):
    """Peta Grad-CAM (H_in, W_in) bernilai 0..1 untuk satu kelas.
    - Lapisan target: keluaran terakhir encoder ResNet (layer4).
    - Skor = jumlah logit kelas target pada piksel yang diprediksi kelas itu.
    - Bobot kanal = rata-rata gradien; peta = ReLU(sum bobot x aktivasi)."""
    model.eval()
    store = {}

    def fwd_hook(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = model.encoder.layer4.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            x = x.to(device).requires_grad_(False)
            logits = model(x)                                        # (1, 3, H, W)
            region = F.interpolate(torch.from_numpy((pred_mask == target_class).astype(np.float32))[None, None],
                                   size=logits.shape[-2:], mode="nearest").to(device)[0, 0]
            if region.sum() == 0:                                    # kelas tidak diprediksi sama sekali
                return np.zeros(logits.shape[-2:], np.float32)
            score = (logits[0, target_class] * region).sum()
            model.zero_grad()
            score.backward()
    finally:
        h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def cam_energy_shares(cam, gt, size):
    """Porsi energi peta (jumlah nilai) yang jatuh pada gt nukleus, sitoplasma, dan latar.
    gt di-resize (nearest) ke resolusi peta. Jawaban kuantitatif untuk 'model melihat apa?'."""
    g = np.array(Image.fromarray(gt.astype(np.uint8)).resize((size, size), Image.NEAREST))
    tot = cam.sum() + 1e-8
    return {"inti": float(cam[g == 2].sum() / tot),
            "sitoplasma": float(cam[g == 1].sum() / tot),
            "latar": float(cam[g == 0].sum() / tot)}
