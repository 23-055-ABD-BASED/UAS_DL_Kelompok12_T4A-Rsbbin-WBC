
import os, numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DRIVE = "/content/drive/MyDrive/PROYEK UAS DEEP LEARNING"
BMP = "/content/raabin_gt/bmp"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # normalisasi ImageNet

def load_split(scheme, part):
    # scheme: grouped | random | cross_session ; part: train | val | test
    return pd.read_csv(f"{DRIVE}/splits/{scheme}_{part}.csv")

def _path(name, mask):
    for s in ("Train", "Test"):
        p = f"{BMP}/{s}_GT_bmp/{name}" if mask else f"{BMP}/{s}_bmp/{name}"
        if os.path.exists(p): return p
    raise FileNotFoundError(name)

def make_tf(train, size=256, color_aug=True):
    t = [A.Resize(size, size)]
    if train:
        t += [A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90()]
        if color_aug: t += [A.ColorJitter(0.2, 0.2, 0.2, 0.02, p=0.8)]
    t += [A.Normalize(MEAN, STD), ToTensorV2()]
    return A.Compose(t)

class WBCSeg(Dataset):
    # mask: 0=latar, 1=sitoplasma, 2=nukleus
    def __init__(self, df, train=False, size=256, color_aug=True):
        self.names = df["name"].tolist(); self.tf = make_tf(train, size, color_aug)
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        n = self.names[i]
        x = np.array(Image.open(_path(n, False)).convert("RGB"))
        m = np.array(Image.open(_path(n, True)))
        if m.ndim == 3: m = m[..., 0]
        y = np.zeros(m.shape, np.uint8); y[m == 255] = 1; y[m == 100] = 2
        o = self.tf(image=x, mask=y)
        return o["image"], o["mask"].long(), n
