"""Evaluation metrics for FeTS 2022 3D MRI Brain Tumor Segmentation."""

import gc
import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric

MAX_BRATS_DISTANCE_MM = 373.13  # Standard diagonal penalty for missed lesions


def _compute_hd95_single(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    """Computes HD95 on a single 3D binary volume using bounding-box cropping for minimal RAM usage."""
    p_sum = float(pred_mask.sum().item())
    t_sum = float(target_mask.sum().item())

    # Standard BraTS rule:
    # If both have no lesion, distance is 0.0 (perfect negative)
    if t_sum == 0.0 and p_sum == 0.0:
        return 0.0
    # If one has lesion and other does not, apply max BraTS penalty distance
    if t_sum == 0.0 or p_sum == 0.0:
        return MAX_BRATS_DISTANCE_MM

    # Both have lesion: crop around the union bounding box + 10 voxel margin.
    # Because HD95 is only calculated between foreground surfaces, cropping empty
    # background yields mathematically identical distance values while reducing RAM by >95%.
    combined = pred_mask | target_mask
    coords = torch.nonzero(combined)
    if coords.numel() == 0:
        return 0.0

    min_c = coords.min(dim=0).values
    max_c = coords.max(dim=0).values
    vol_shape = pred_mask.shape

    pad = 10
    d_min = max(0, int(min_c[0]) - pad)
    d_max = min(vol_shape[0], int(max_c[0]) + pad + 1)
    h_min = max(0, int(min_c[1]) - pad)
    h_max = min(vol_shape[1], int(max_c[1]) + pad + 1)
    w_min = max(0, int(min_c[2]) - pad)
    w_max = min(vol_shape[2], int(max_c[2]) + pad + 1)

    c_p = pred_mask[d_min:d_max, h_min:h_max, w_min:w_max].unsqueeze(0).unsqueeze(0).float()
    c_t = target_mask[d_min:d_max, h_min:h_max, w_min:w_max].unsqueeze(0).unsqueeze(0).float()

    try:
        metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        val = metric(c_p, c_t).item()
        if np.isnan(val) or np.isinf(val):
            return MAX_BRATS_DISTANCE_MM
        return float(val)
    except (MemoryError, Exception):
        # Fallback to max distance penalty if system RAM is constrained
        return MAX_BRATS_DISTANCE_MM


def _compute_dice_single(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    """Computes binary Dice coefficient without allocating extra metric tensors."""
    p_sum = float(pred_mask.sum().item())
    t_sum = float(target_mask.sum().item())

    if t_sum == 0.0 and p_sum == 0.0:
        return 1.0
    if t_sum == 0.0 or p_sum == 0.0:
        return 0.0

    intersection = float((pred_mask & target_mask).sum().item())
    return (2.0 * intersection) / (p_sum + t_sum)


def fets_region_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute Dice and 95th Percentile Hausdorff Distance (HD95) for ET, TC, WT.
    
    Sub-regions defined by BraTS / FeTS standards:
      - ET (Enhancing Tumor): Label 3
      - TC (Tumor Core): Labels 1 + 3
      - WT (Whole Tumor): Labels 1 + 2 + 3 (all foreground voxels)
    """
    # Move to CPU in boolean form immediately to free GPU memory and avoid large copies
    pred_classes = torch.argmax(logits, dim=1).detach().cpu()
    target_classes = (labels[:, 0] if labels.ndim == 5 else labels).detach().cpu()

    batch_size = pred_classes.shape[0]

    regions = {
        "et": ((pred_classes == 3), (target_classes == 3)),
        "tc": (((pred_classes == 1) | (pred_classes == 3)), ((target_classes == 1) | (target_classes == 3))),
        "wt": ((pred_classes > 0), (target_classes > 0)),
    }

    results = {}
    for name, (p_reg, t_reg) in regions.items():
        batch_dice = []
        batch_hd95 = []
        total_p_vox = 0.0
        total_t_vox = 0.0

        for b in range(batch_size):
            p_b = p_reg[b]
            t_b = t_reg[b]

            p_vox = float(p_b.sum().item())
            t_vox = float(t_b.sum().item())
            total_p_vox += p_vox
            total_t_vox += t_vox

            batch_dice.append(_compute_dice_single(p_b, t_b))
            batch_hd95.append(_compute_hd95_single(p_b, t_b))

        results[f"pred_{name}_voxels"] = total_p_vox / batch_size
        results[f"target_{name}_voxels"] = total_t_vox / batch_size
        results[f"dice_{name}"] = float(np.mean(batch_dice))
        results[f"hd95_{name}"] = float(np.mean(batch_hd95))

    del pred_classes, target_classes, regions
    gc.collect()

    return results