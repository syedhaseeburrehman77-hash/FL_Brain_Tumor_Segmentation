"""Federated Learning Round Summary and CSV Export."""

from pathlib import Path
import csv
import numpy as np
import pandas as pd


CLIENT_HISTORY_FIELDS = (
    "strategy", "round", "institution_id", "phase", "num_examples", "loss",
    "dice_et", "dice_tc", "dice_wt", "hd95_et", "hd95_tc", "hd95_wt",
    "pred_et_voxels", "pred_tc_voxels", "pred_wt_voxels",
    "target_et_voxels", "target_tc_voxels", "target_wt_voxels",
    "time_sec", "aggregation_weight",
    "institution", "total_cases", "train_cases", "val_cases",
)


def append_client_history(
    run_id: int, strategy: str, round_id: int, institution_id: int,
    phase: str, num_examples: int, loss: float, time_sec: float,
    metrics: dict | None = None,
) -> None:
    """Write one client/round/phase row to a run-specific part CSV."""
    metrics = metrics or {}
    row = {key: "" for key in CLIENT_HISTORY_FIELDS}
    row.update({
        "strategy": strategy,
        "round": int(round_id),
        "institution_id": int(institution_id),
        "phase": phase,
        "num_examples": int(num_examples),
        "loss": float(loss),
        "time_sec": float(time_sec),
    })
    for key in CLIENT_HISTORY_FIELDS:
        if key in metrics:
            row[key] = float(metrics[key])

    parts_dir = Path("artifacts/client_history_parts") / str(run_id)
    parts_dir.mkdir(parents=True, exist_ok=True)
    csv_file = parts_dir / f"client_{institution_id}.csv"
    write_header = not csv_file.exists()
    with csv_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CLIENT_HISTORY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _dataset_distribution(loader) -> tuple[dict[int, dict], dict]:
    """Summarize the loader's exact train/validation/global-test case counts."""
    per_client = {}
    global_test_records = []
    groups = loader._get_partitioned_groups()
    for client_id, (institution, records) in enumerate(groups):
        trainval_records, test_records = loader._split_global_test(records, client_id)
        global_test_records.extend(test_records)

        rng = np.random.default_rng(loader.seed + client_id)
        n_val = max(1, int(round(len(trainval_records) * 0.15)))
        val_indices = set(rng.permutation(len(trainval_records))[:n_val].tolist())
        train_records = [r for i, r in enumerate(trainval_records) if i not in val_indices]
        val_records = [r for i, r in enumerate(trainval_records) if i in val_indices]

        per_client[client_id] = {
            "institution": institution,
            "total_cases": len(trainval_records),
            "train_cases": len(train_records),
            "val_cases": len(val_records),
        }

    global_test = {"global_test_cases": len(global_test_records)}
    return per_client, global_test


def merge_client_history(run_id: int, strategy, loader=None) -> Path:
    """Merge per-client rows and attach strategy aggregation weights."""
    parts_dir = Path("artifacts/client_history_parts") / str(run_id)
    rows = []
    if parts_dir.exists():
        for csv_file in sorted(parts_dir.glob("client_*.csv")):
            with csv_file.open("r", newline="", encoding="utf-8") as file:
                rows.extend(csv.DictReader(file))

    weights_by_round_client = {}
    for audit in getattr(strategy, "aggregation_audit", []):
        for institution_id, weight in zip(
            audit.get("institution_ids", []), audit.get("aggregation_weights", [])
        ):
            weights_by_round_client[(int(audit["round"]), int(institution_id))] = float(weight)
    for row in rows:
        key = (int(row["round"]), int(row["institution_id"]))
        if key in weights_by_round_client:
            row["aggregation_weight"] = weights_by_round_client[key]

    if loader is not None:
        distribution_by_client, global_test_distribution = _dataset_distribution(loader)
        for row in rows:
            row.update(distribution_by_client.get(int(row["institution_id"]), {}))
            row.update(global_test_distribution)

    out_path = Path("artifacts/client_history.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CLIENT_HISTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Server] Client history saved to CSV: {out_path}", flush=True)
    return out_path


def save_round_summary(result, algorithm: str, num_rounds: int) -> pd.DataFrame:
    """Format and save round-by-round federated learning metrics summary."""
    df_hist = None
    hist_path = Path("artifacts/client_history.csv")
    if hist_path.exists() and hist_path.stat().st_size > 0:
        try:
            df_hist = pd.read_csv(hist_path)
        except Exception:
            df_hist = None

    if df_hist is None or df_hist.empty:
        parts_dir = Path("artifacts/client_history_parts")
        if parts_dir.exists():
            csv_files = list(parts_dir.glob("*/*.csv"))
            if csv_files:
                try:
                    df_hist = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
                except Exception:
                    df_hist = None

    rows = []
    for r in range(1, num_rounds + 1):
        tr = result.train_metrics_clientapp.get(r, {})
        ev = result.evaluate_metrics_clientapp.get(r, {})

        num_ex = tr.get("num-examples")
        if (num_ex is None or pd.isna(num_ex)) and df_hist is not None:
            r_rows = df_hist[
                (df_hist["strategy"].astype(str).str.lower() == str(algorithm).lower())
                & (pd.to_numeric(df_hist["round"], errors="coerce") == r)
            ]
            if not r_rows.empty and "num_examples" in r_rows.columns:
                if "phase" in r_rows.columns and (r_rows["phase"] == "train").any():
                    num_ex = float(r_rows[r_rows["phase"] == "train"]["num_examples"].sum())
                else:
                    num_ex = float(r_rows["num_examples"].sum())

        row = {
            "round": r,
            "strategy": algorithm,
            "profile_phase": tr.get("profile_phase", np.nan),
            "num-examples": int(round(num_ex)) if (num_ex is not None and not pd.isna(num_ex)) else np.nan,
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
    if "num-examples" in df.columns and not df["num-examples"].isna().all():
        df["num-examples"] = df["num-examples"].astype("Int64")

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
