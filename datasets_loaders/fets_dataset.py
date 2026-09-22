"""FeTS 2022 3D MRI Brain Tumor Dataset Loader with MONAI preprocessing."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    Spacingd,
)

from .base import BaseDatasetLoader

PATCH_SIZE = (96, 96, 96)


class FeTSDatasetLoader(BaseDatasetLoader):
    """Loads 3D multi-parametric MRI (T1, T1ce, T2, FLAIR) for FeTS 2022."""

    def __init__(
        self,
        root_dir: str | Path,
        partition_csv: str | Path | None = None,
        global_test_fraction: float = 0.15,
        seed: int = 42,
    ):
        super().__init__(root_dir, partition_csv)
        self.global_test_fraction = global_test_fraction
        self.seed = seed
        self._groups = None

    # --- Built-in MONAI 3D Preprocessing Transforms ---
    @staticmethod
    def get_train_transforms():
        return Compose([
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            Spacingd(keys=("image", "label"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
            RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3),
            EnsureTyped(keys=("image", "label")),
        ])

    @staticmethod
    def get_val_transforms():
        return Compose([
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            Spacingd(keys=("image", "label"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            EnsureTyped(keys=("image", "label")),
        ])

    # --- Internal File Matching ---
    def _case_files(self, subject_dir: Path) -> dict:
        files = list(subject_dir.rglob("*.nii")) + list(subject_dir.rglob("*.nii.gz"))
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {subject_dir}")

        def find(modality: str) -> Path:
            matches = [p for p in files if modality in p.name.lower() and "seg" not in p.name.lower()]
            if not matches:
                raise FileNotFoundError(f"Missing {modality} in {subject_dir}")
            return sorted(matches, key=lambda p: len(p.name))[0]

        images = [str(find(m)) for m in ("t1ce", "t1", "t2", "flair")]
        labels = [p for p in files if "seg" in p.name.lower()]
        if len(labels) != 1:
            raise FileNotFoundError(f"Expected 1 segmentation in {subject_dir}, found {len(labels)}")

        return {"image": images, "label": str(labels[0]), "subject_id": subject_dir.name}

    def _get_partitioned_groups(self):
        if self._groups is not None:
            return self._groups

        table = pd.read_csv(self.partition_csv)
        groups = []
        for partition_id, frame in table.groupby("Partition_ID", sort=True):
            records = [self._case_files(self.root_dir / str(s_id)) for s_id in frame["Subject_ID"]]
            groups.append((str(partition_id), records))
        self._groups = groups
        return self._groups

    def _split_global_test(self, records: list[dict], partition_index: int):
        if not 0.0 < self.global_test_fraction < 1.0 or len(records) < 2:
            return records, []
        rng = np.random.default_rng(self.seed + partition_index)
        n_test = max(1, int(round(len(records) * self.global_test_fraction)))
        test_indices = set(rng.permutation(len(records))[:n_test].tolist())
        trainval = [r for i, r in enumerate(records) if i not in test_indices]
        test = [r for i, r in enumerate(records) if i in test_indices]
        return trainval, test

    # --- Public Loader APIs ---
    def load_partition(self, partition_id: int, batch_size: int = 1):
        """Return (train_loader, val_loader) for client partition_id."""
        groups = self._get_partitioned_groups()
        if not 0 <= partition_id < len(groups):
            raise IndexError(f"Partition {partition_id} invalid (found {len(groups)} partitions)")

        # 1. Reserve unseen global test cases first
        trainval_records, _ = self._split_global_test(groups[partition_id][1], partition_id)

        # 2. Local 85% train / 15% val split for this institution
        rng = np.random.default_rng(self.seed + partition_id)
        n_val = max(1, int(round(len(trainval_records) * 0.15)))
        val_indices = set(rng.permutation(len(trainval_records))[:n_val].tolist())
        train_records = [r for i, r in enumerate(trainval_records) if i not in val_indices]
        val_records = [r for i, r in enumerate(trainval_records) if i in val_indices]

        train_ds = Dataset(train_records, transform=self.get_train_transforms())
        val_ds = Dataset(val_records, transform=self.get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available())
        return train_loader, val_loader

    def load_global_test(self, batch_size: int = 1):
        """Return unseen test DataLoader for ServerApp global evaluation."""
        groups = self._get_partitioned_groups()
        all_test_records = []
        for p_idx, (_, records) in enumerate(groups):
            _, test_records = self._split_global_test(records, p_idx)
            all_test_records.extend(test_records)

        test_ds = Dataset(all_test_records, transform=self.get_val_transforms())
        return DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available())