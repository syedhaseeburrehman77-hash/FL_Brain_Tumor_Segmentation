"""Federated Learning Round Summary and CSV Export."""

from pathlib import Path
import numpy as np
import pandas as pd


def save_round_summary(result, algorithm: str, num_rounds: int) -> pd.DataFrame:
    """Format and save round-by-round federated learning metrics summary."""
    rows = []
    for r in range(1, num_rounds + 1):
        tr = result.train_metrics_clientapp.get(r, {})
        ev = result.evaluate_metrics_clientapp.get(r, {})

        row = {
            "round": r,
            "strategy": algorithm,
            "profile_phase": tr.get("profile_phase", np.nan),
            "num-examples": tr.get("num-examples", np.nan),
            "train_loss": tr.get("train_loss", np.nan),
            "eval_loss": ev.get("eval_loss", np.nan),
            "eval_dice_et": ev.get("dice_et", np.nan),
            "eval_dice_tc": ev.get("dice_tc", np.nan),
            "eval_dice_wt": ev.get("dice_wt", np.nan),
            "eval_hd95_et": ev.get("hd95_et", np.nan),
            "eval_hd95_tc": ev.get("hd95_tc", np.nan),
            "eval_hd95_wt": ev.get("hd95_wt", np.nan),
            "eval_pred_wt_voxels": ev.get("pred_wt_voxels", np.nan),
            "eval_target_wt_voxels": ev.get("target_wt_voxels", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 1. Print formatted table in terminal (Exact like previous project)
    header = f"FEDERATED LEARNING RESULTS SUMMARY ({algorithm.upper()})"
    print(f"\n{'=' * 78}\n{header:^78}\n{'=' * 78}")
    print(df.to_string(index=False))
    print(f"{'=' * 78}")

    # 2. Save directly to artifacts/{algorithm}_fets2022_metrics.csv
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algorithm}_fets2022_summary_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"[Server] Round results saved to CSV: {out_path}\n", flush=True)

    return df