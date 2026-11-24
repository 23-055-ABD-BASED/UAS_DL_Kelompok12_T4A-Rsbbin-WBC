"""
Pembuat model dan fungsi loss.
Pemegang: Aditya (U-Net, loss) dan Anggota 3 (DeepLabV3+).

Dua paradigma yang dibandingkan (syarat Kisi-Kisi 3.2):
  - U-Net          : encoder-decoder dengan skip connection
  - DeepLabV3+     : konvolusi atrous (ASPP) + decoder ringan
Keduanya memakai encoder ResNet-34 berbobot pralatih ImageNet.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

NUM_CLASSES = 3  # 0 latar, 1 sitoplasma, 2 nukleus


def build_model(arch="unet", encoder="resnet34", pretrained=True, classes=NUM_CLASSES):
    """arch: 'unet' atau 'deeplabv3plus'. pretrained=False -> inisialisasi acak
    (untuk ablasi opsional 'encoder pralatih vs acak')."""
    weights = "imagenet" if pretrained else None
    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=weights,
                        in_channels=3, classes=classes)
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=weights,
                                 in_channels=3, classes=classes)
    raise ValueError(f"arch tidak dikenal: {arch}")


class SegLoss(nn.Module):
    """Loss gabungan. name: 'ce' | 'dice' | 'ce_dice' | 'focal_dice'.

    - Dice dihitung pada kelas FOREGROUND (sitoplasma, nukleus) saja, karena
      latar menempati ~84% piksel (lihat EDA) dan akan mendominasi Dice.
    - ce_dice   = CE + Dice (bobot 1:1)
    - focal_dice = Focal + Dice (bobot 1:1)
    """
    def __init__(self, name="ce_dice"):
        super().__init__()
        self.name = name
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2])
        self.focal = smp.losses.FocalLoss(mode="multiclass")

    def forward(self, logits, target):
        if self.name == "ce":
            return self.ce(logits, target)
        if self.name == "dice":
            return self.dice(logits, target)
        if self.name == "ce_dice":
            return self.ce(logits, target) + self.dice(logits, target)
        if self.name == "focal_dice":
            return self.focal(logits, target) + self.dice(logits, target)
        raise ValueError(f"loss tidak dikenal: {self.name}")
