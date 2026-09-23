"""Evaluation metrics for FeTS 2022 3D MRI Brain Tumor Segmentation."""

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

MAX_BRATS_DISTANCE_MM = 373.13  # Standard diagonal penalty for missed lesions

def fets_region_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute Dice and 95th Percentile Hausdorff Distance (HD95) for ET, TC, WT.
    
    Sub-regions defined by BraTS / FeTS standards:
      - ET (Enhancing Tumor): Label 3
      - TC (Tumor Core): Labels 1 + 3
      - WT (Whole Tumor): Labels 1 + 2 + 3 (all foreground voxels)
    """
    prediction = torch.argmax(logits, dim=1)
    target = labels[:, 0] if labels.ndim == 5 else labels

    pred_et = (prediction == 3).float()
    pred_tc = ((prediction == 1) | (prediction == 3)).float()
    pred_wt = (prediction > 0).float()

    target_et = (target == 3).float()
    target_tc = ((target == 1) | (target == 3)).float()
    target_wt = (target > 0).float()

    pred_regions = torch.stack((pred_et, pred_tc, pred_wt), dim=1)
    target_regions = torch.stack((target_et, target_tc, target_wt), dim=1)

    dice_metric = DiceMetric(include_background=True, reduction="none")
    dice_scores = dice_metric(pred_regions, target_regions).detach().cpu().numpy()

    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
    hd95_scores = hd95_metric(pred_regions, target_regions).detach().cpu().numpy()

    results = {}
    region_names = ["et", "tc", "wt"]
    for i, name in enumerate(region_names):
        p_vox = float(pred_regions[:, i].sum().item())
        t_vox = float(target_regions[:, i].sum().item())
        results[f"pred_{name}_voxels"] = p_vox
        results[f"target_{name}_voxels"] = t_vox

        # Dice calculation
        if t_vox == 0.0 and p_vox == 0.0:
            d_val = 1.0
        elif t_vox == 0.0 or p_vox == 0.0:
            d_val = 0.0
        else:
            d_val = float(np.nan_to_num(dice_scores[:, i].mean(), nan=0.0))
        results[f"dice_{name}"] = d_val

        # HD95 calculation (penalize missed or false positive lesions)
        if t_vox == 0.0 and p_vox == 0.0:
            h_val = 0.0
        elif t_vox == 0.0 or p_vox == 0.0:
            h_val = MAX_BRATS_DISTANCE_MM
        else:
            raw_h = float(hd95_scores[:, i].mean())
            if np.isnan(raw_h) or np.isinf(raw_h):
                h_val = MAX_BRATS_DISTANCE_MM
            else:
                h_val = raw_h
        results[f"hd95_{name}"] = h_val

    return results