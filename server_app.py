import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from task import load_centralized_dataset, test
from algorithms.server_strategies import get_strategy
from models import create_model


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    algorithm = context.run_config["algorithm"]

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = create_model("unet")
    arrays = ArrayRecord(global_model.state_dict())

    # Build kwargs relevant to whichever strategy is picked
    strategy_kwargs = {
        "fraction_evaluate": fraction_evaluate,
        "min_available_nodes": context.run_config["num-clients"],
    }

    if algorithm == "fedprox":
        strategy_kwargs["proximal_mu"] = context.run_config["proximal_mu"]

    strategy = get_strategy(algorithm, **strategy_kwargs)
    print(f"\n[Server] ---> Starting Federated Training ({algorithm.upper()}) for {num_rounds} rounds...")

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config.get("save-model", True):
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    if server_round == 0:
        return MetricRecord({"info": 0.0})
    print(f"\n[Server] ---> Evaluating global 3D U-Net on unseen test set (Round {server_round})...")

    # Load the model using the model registry
    model = create_model("unet")

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
    print(f"[Server] ---> Round {server_round} Test Results | Loss: {eval_metrics.get('eval_loss', 0.0):.4f} | Dice (ET/TC/WT): {eval_metrics.get('dice_et', 0.0):.4f} / {eval_metrics.get('dice_tc', 0.0):.4f} / {eval_metrics.get('dice_wt', 0.0):.4f}")

    # Return evaluation metrics
    return MetricRecord(eval_metrics)