# ---------------------------------------------------------------
# TEP (Totally Extraperitoneal) Inguinal Hernia Surgery Dataset
# Semantic segmentation dataset for EoMT
#
# Directory structure expected:
#   TEP_dataset/ro/
#     train/images/*.png  train/masks/*.png   ← 학습
#     tune/images/*.png   tune/masks/*.png    ← validation
#     int_val/images/*.png  int_val/masks/*.png  ← test (internal)
#     ext_val/images/*.png  ext_val/masks/*.png  ← test (external)
#
# Classes (index mask):
#   0: background
#   1: inferior epigastric vessels (IEV)
#   2: pubic arch
#   3: spermatic cord
#   4: vas deferens
#
# Synthetic image 파일명 패턴:
#   환자번호_frame번호_숫자_75.png
#   환자번호_frame번호_숫자_85.png
#   환자번호_frame번호_75.png
#   환자번호_frame번호_85.png
# ---------------------------------------------------------------

from pathlib import Path
from typing import Optional, Union
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

NUM_CLASSES = 5
CLASS_IDS = [0, 1, 2, 3, 4]


def is_synthetic(img_path: Path) -> bool:
    """파일명 기준으로 synthetic image 여부 판별."""
    stem = img_path.stem
    return stem.endswith('_75') or stem.endswith('_85')


class TEPDataset(TorchDataset):
    def __init__(
        self,
        img_dir: Path,
        mask_dir: Path,
        transforms=None,
        transforms_syn=None,  # synthetic 전용 transform (None이면 transforms와 동일)
        img_suffix: str = ".png",
        mask_suffix: str = ".png",
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        self.transforms_syn = transforms_syn if transforms_syn is not None else transforms
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        self.img_paths = sorted(self.img_dir.glob(f"*{img_suffix}"))

        valid = []
        for img_path in self.img_paths:
            mask_path = self.mask_dir / (img_path.stem + mask_suffix)
            if mask_path.exists():
                valid.append((img_path, mask_path))

        self.samples = valid

        n_syn  = sum(1 for p, _ in self.samples if is_synthetic(p))
        n_real = len(self.samples) - n_syn
        print(f"[TEPDataset] {img_dir.parent.name}/{img_dir.name}: "
              f"{len(self.samples)} samples (real={n_real}, synthetic={n_syn})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        mask = tv_tensors.Mask(Image.open(mask_path), dtype=torch.long)

        if img.shape[-2:] != mask.shape[-2:]:
            mask = F.resize(
                mask,
                list(img.shape[-2:]),
                interpolation=F.InterpolationMode.NEAREST,
            )

        masks, labels = [], []
        for cls_id in CLASS_IDS:
            binary = (mask[0] == cls_id)
            if binary.any():
                masks.append(binary)
                labels.append(cls_id)

        if len(masks) == 0:
            masks.append(torch.zeros(mask.shape[-2:], dtype=torch.bool))
            labels.append(0)

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor([False] * len(labels)),
        }

        # real/synthetic에 따라 다른 transform 적용
        transform = self.transforms_syn if is_synthetic(img_path) else self.transforms
        if transform is not None:
            img, target = transform(img, target)

        return img, target


class TEPSemantic(LightningDataModule):
    """
    LightningDataModule for TEP surgery semantic segmentation.

    splits:
        split_train: 학습 데이터 (default: 'train')
        split_val:   validation 데이터 (default: 'tune')
        → int_val, ext_val은 학습 후 inference_tep.py로 평가

    Args:
        path: TEP_dataset/ro/ 경로
        img_size: (504, 504) for DINOv2, (512, 512) for DINOv3
        color_jitter_enabled: real image augmentation 여부
        scale_range: real image scale jitter 범위
        syn_color_jitter_enabled: synthetic image color jitter 여부 (default: False)
        syn_scale_range: synthetic image scale jitter 범위 (default: (1.0, 1.0))
    """

    def __init__(
        self,
        path: str,
        split_train: str = "train",
        split_val: str = "tune",
        num_workers: int = 4,
        batch_size: int = 8,
        img_size: tuple = (504, 504),
        num_classes: int = NUM_CLASSES,
        color_jitter_enabled: bool = True,
        scale_range: tuple = (0.5, 2.0),
        syn_color_jitter_enabled: bool = False,
        syn_scale_range: tuple = (1.0, 1.0),
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=False,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.split_train = split_train
        self.split_val = split_val

        # real image transform (strong aug)
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

        # synthetic image transform (weak aug)
        self.transforms_syn = Transforms(
            img_size=img_size,
            color_jitter_enabled=syn_color_jitter_enabled,
            scale_range=syn_scale_range,
        )

    def setup(self, stage: Optional[str] = None) -> "TEPSemantic":
        base = Path(self.path)

        train_img  = base / self.split_train / "images"
        train_mask = base / self.split_train / "masks"
        self.train_dataset = TEPDataset(
            train_img, train_mask,
            transforms=self.transforms,
            transforms_syn=self.transforms_syn,
        )
        print(f"[TEPSemantic] Train: {len(self.train_dataset)} samples")

        val_img  = base / self.split_val / "images"
        val_mask = base / self.split_val / "masks"
        self.val_dataset = TEPDataset(val_img, val_mask)
        print(f"[TEPSemantic] Val ({self.split_val}): {len(self.val_dataset)} samples")

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )