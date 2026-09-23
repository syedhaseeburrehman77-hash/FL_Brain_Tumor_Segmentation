"""Integration tests for FeTS 2022 3D MRI Dataset Loader."""

from pathlib import Path
import tomllib
import pytest
import torch

from datasets_loaders.fets_dataset import FeTSDatasetLoader


@pytest.fixture(scope="module")
def dataset_config():
    """Load dataset configuration paths from pyproject.toml."""
    config_path = Path("pyproject.toml")
    assert config_path.exists(), "pyproject.toml must exist in the project root"
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    return {
        "root_dir": cfg["data-root"],
        "partition_csv": cfg["partition-csv"],
    }


@pytest.fixture(scope="module")
def loader(dataset_config):
    """Instantiate FeTSDatasetLoader using configured paths."""
    return FeTSDatasetLoader(
        root_dir=dataset_config["root_dir"],
        partition_csv=dataset_config["partition_csv"],
        global_test_fraction=0.15,
        seed=42,
    )


def test_partitions_and_total_cases(loader):
    """Test that partitioning loads all 23 institutions and 1,251 total cases."""
    groups = loader._get_partitioned_groups()
    
    # Check number of clinical institutions (23 in partitioning_1.csv)
    assert len(groups) == 23, f"Expected 23 partitions, found {len(groups)}"

    # Check total labelled subject records
    total_cases = sum(len(records) for _, records in groups)
    assert total_cases == 1251, f"Expected 1251 FeTS cases, found {total_cases}"


def test_client_partition_loaders(loader):
    """Test train and validation loaders for institution 0."""
    train_loader, val_loader = loader.load_partition(partition_id=0, batch_size=1)

    assert len(train_loader.dataset) > 0, "Train dataset should not be empty"
    assert len(val_loader.dataset) > 0, "Validation dataset should not be empty"


def test_3d_batch_tensor_shapes(loader):
    """Test that batches have 4 MRI channels (T1, T1ce, T2, FLAIR) and (96, 96, 96) patch size."""
    train_loader, _ = loader.load_partition(partition_id=0, batch_size=1)
    
    # Get a single 3D training batch
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]

    # Image shape: [batch_size, 4_channels, D, H, W] -> [1, 4, 96, 96, 96]
    assert images.ndim == 5, f"Expected 5D tensor (B, C, D, H, W), got {images.shape}"
    assert images.shape[1] == 4, f"Expected 4 MRI modalities, got {images.shape[1]}"
    assert images.shape[2:] == (96, 96, 96), f"Expected 96x96x96 patch, got {images.shape[2:]}"

    # Label shape: [batch_size, 1_channel, D, H, W] -> [1, 1, 96, 96, 96]
    assert labels.shape[1] == 1, f"Expected 1 label channel, got {labels.shape[1]}"


def test_global_unseen_test_loader(loader):
    """Test that the 15% unseen global test set is properly reserved."""
    test_loader = loader.load_global_test(batch_size=1)
    
    # 15% of 1251 is ~188 unseen test cases
    assert len(test_loader.dataset) == 187, f"Expected 187 unseen test cases, got {len(test_loader.dataset)}"