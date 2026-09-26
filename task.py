"""FeTS 2022 3D MRI Brain Tumor Segmentation: Task and Evaluation Module."""

import csv
import tomllib
from pathlib import Path
import torch
import torch.nn as nn
from flwr.app import Context
from datasets_loaders import create_dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from utils.metrics import fets_region_metrics

_LOADER = None

def get_loader(context=None):
    global _LOADER
    if _LOADER is not None:
        return _LOADER
    
    if context is not None:
        cfg = context.run_config
    else:
        cfg = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]

    # Resolve relative paths against the repository root (use workspace CWD first)
    project_root = Path.cwd().resolve() if (Path.cwd() / "data").exists() else Path(__file__).parent.resolve()
    
    data_root = Path(cfg["data-root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root
    partition_csv = Path(cfg["partition-csv"])
    if not partition_csv.is_absolute():
        partition_csv = project_root / partition_csv

    _LOADER = create_dataset(
        "fets2022",
        root_dir=data_root,
        partition_csv=partition_csv,
        global_test_fraction=float(cfg.get("global-test-fraction", 0.15)),
        seed=int(cfg.get("seed", 42)),
    )
    return _LOADER

def load_data(partition_id: int, context):
    loader = get_loader(context)
    batch_size = int(context.run_config.get("batch-size", 1))
    return loader.load_partition(partition_id=partition_id, batch_size=batch_size)

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    loader = get_loader()
    return loader.load_global_test(batch_size=1)

def test(net: torch.nn.Module, testloader, device: torch.device) -> dict[str, float]:
    """Evaluate 3D U-Net using sliding-window inference and return Dice/HD95 metrics."""
    net.eval()
    net.to(device)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    totals = {
        "eval_loss": 0.0,
        "dice_et": 0.0,
        "dice_tc": 0.0,
        "dice_wt": 0.0,
        "hd95_et": 0.0,
        "hd95_tc": 0.0,
        "hd95_wt": 0.0,
        "pred_et_voxels": 0.0,
        "pred_tc_voxels": 0.0,
        "pred_wt_voxels": 0.0,
        "target_et_voxels": 0.0,
        "target_tc_voxels": 0.0,
        "target_wt_voxels": 0.0,
    }
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            # 3D sliding window inference with 96x96x96 patches
            logits = sliding_window_inference(
                inputs=images,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=net,
            )
            totals["eval_loss"] += criterion(logits, labels).item()
            # Calculate Dice and Hausdorff HD95 for ET, TC, and WT
            region_scores = fets_region_metrics(logits, labels)
            for key in (
                "dice_et", "dice_tc", "dice_wt", "hd95_et", "hd95_tc", "hd95_wt",
                "pred_et_voxels", "pred_tc_voxels", "pred_wt_voxels",
                "target_et_voxels", "target_tc_voxels", "target_wt_voxels",
            ):
                if key in region_scores:
                    totals[key] += region_scores[key]
    count = max(len(testloader), 1)
    return {k: round(v / count, 4) for k, v in totals.items()}

def run_global_benchmark(model: torch.nn.Module, device: torch.device, algorithm: str = "fedavg") -> dict[str, float]:
    """Run one-off final benchmark on the unseen global test set and save to artifacts."""
    print(f"\n[Server] ---> Running Final Benchmark on Unseen Test Set ({algorithm.upper()})...", flush=True)

    test_dataloader = load_centralized_dataset()
    eval_metrics = test(model, test_dataloader, device)

    print(
        f"[Server] ---> Final Test Results | "
        f"Loss: {eval_metrics.get('eval_loss', 0.0):.4f} | "
        f"Dice (ET/TC/WT): {eval_metrics.get('dice_et', 0.0):.4f} / {eval_metrics.get('dice_tc', 0.0):.4f} / {eval_metrics.get('dice_wt', 0.0):.4f} | "
        f"HD95 (ET/TC/WT): {eval_metrics.get('hd95_et', 0.0):.2f} / {eval_metrics.get('hd95_tc', 0.0):.2f} / {eval_metrics.get('hd95_wt', 0.0):.2f}",
        flush=True,
    )

    # Save final benchmark to artifacts/detail_metrics_{algorithm}.csv
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_file = artifacts_dir / f"detail_metrics_{algorithm}.csv"
    file_exists = csv_file.exists()

    row = {
        "strategy": algorithm,
        "test_loss": eval_metrics.get("eval_loss", 0.0),
        "dice_et": eval_metrics.get("dice_et", 0.0),
        "dice_tc": eval_metrics.get("dice_tc", 0.0),
        "dice_wt": eval_metrics.get("dice_wt", 0.0),
        "hd95_et": eval_metrics.get("hd95_et", 0.0),
        "hd95_tc": eval_metrics.get("hd95_tc", 0.0),
        "hd95_wt": eval_metrics.get("hd95_wt", 0.0),
    }
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return eval_metrics
