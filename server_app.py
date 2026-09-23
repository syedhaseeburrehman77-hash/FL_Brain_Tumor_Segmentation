import csv
from pathlib import Path
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

_ALGORITHM = "fedavg"

from task import get_loader, load_centralized_dataset, test
from algorithms.server_strategies import get_strategy
from models import create_model


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Initialize data loader with run context
    get_loader(context)

    global _ALGORITHM
    _ALGORITHM = context.run_config["algorithm"]
    algorithm = _ALGORITHM

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    model_name = context.run_config.get("model_name", "unet").lower()
    global_model = create_model(model_name)
    arrays = ArrayRecord(global_model.state_dict())

    # Build kwargs relevant to whichever strategy is picked
    strategy_kwargs = {
        "fraction_train": float(context.run_config.get("fraction-train", 1.0)),
        "fraction_evaluate": fraction_evaluate,
        "min_available_nodes": context.run_config["num-clients"],
    }

    if algorithm == "fedprox":
        strategy_kwargs["proximal_mu"] = float(context.run_config["proximal_mu"])

    strategy = get_strategy(algorithm, **strategy_kwargs)
    print(f"\n[Server] ---> Starting Federated Training ({algorithm.upper()}) for {num_rounds} rounds...", flush=True)

    # Start strategy
    train_cfg = {
        "lr": lr,
        "proximal_mu": float(context.run_config.get("proximal_mu", 0.01)),
    }
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_cfg),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config.get("save-model", True):
        # Save final model to artifacts/
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        print("\nSaving final model to artifacts/final_model.pt...", flush=True)
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, artifacts_dir / "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    if server_round == 0:
        return MetricRecord({"info": 0.0})
    print(f"\n[Server] ---> Evaluating global 3D U-Net on unseen test set (Round {server_round})...", flush=True)

    # Load the model using the model registry
    model_name = "unet"
    model = create_model(model_name)

    # Initialize it with the received global weights
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model
    eval_metrics = test(
        model,
        test_dataloader,
        device,
    )
    print(f"[Server] ---> Round {server_round} Test Results | Loss: {eval_metrics.get('eval_loss', 0.0):.4f} | Dice (ET/TC/WT): {eval_metrics.get('dice_et', 0.0):.4f} / {eval_metrics.get('dice_tc', 0.0):.4f} / {eval_metrics.get('dice_wt', 0.0):.4f}", flush=True)

    # Save metrics to artifacts/detail_metrics_{strategy}.csv
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_file = artifacts_dir / f"detail_metrics_{_ALGORITHM}.csv"
    file_exists = csv_file.exists()
    row = {
        "strategy": _ALGORITHM,
        "round": server_round,
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

    # Return evaluation metrics
    return MetricRecord(eval_metrics)