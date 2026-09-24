from pathlib import Path
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from task import get_loader, run_global_benchmark
from algorithms.server_strategies import get_strategy
from models import create_model


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Initialize data loader with run context
    get_loader(context)

    algorithm = context.run_config["algorithm"]
    model_name = context.run_config.get("model_name", "unet").lower()

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = create_model(model_name)
    arrays = ArrayRecord(global_model.state_dict())

    # Build kwargs relevant to whichever strategy is picked
    strategy_kwargs = {
        "fraction_train": float(context.run_config.get("fraction-train", 1.0)),
        "fraction_evaluate": fraction_evaluate,
        "min_available_nodes": context.run_config["num-clients"],
    }
    if algorithm == "fedprox":
        strategy_kwargs["proximal_mu"] = float(
            context.run_config["proximal_mu"]
        )

    elif algorithm == "fedindar":
        strategy_kwargs["optimization_rounds"] = max(num_rounds - 1, 1)
        strategy_kwargs["base_mu"] = float(
            context.run_config.get("proximal_mu", 0.01)
        )
        strategy_kwargs["alpha"] = float(
            context.run_config.get("fedindar-alpha", 2.0)
        )
        strategy_kwargs["temporal_beta"] = float(
            context.run_config.get("fedindar-temporal-beta", 0.5)
        )
        strategy_kwargs["min_mu"] = float(
            context.run_config.get("fedindar-min-mu", 0.001)
        )
        strategy_kwargs["max_mu"] = float(
            context.run_config.get("fedindar-max-mu", 0.1)
        )

    strategy = get_strategy(algorithm, context=context, **strategy_kwargs)
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
    )

    if context.run_config.get("save-model", True):
        # Save final model to artifacts/
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        print("\nSaving final model to artifacts/final_model.pt...", flush=True)
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, artifacts_dir / "final_model.pt")

    # Run global benchmark on unseen test set after training completes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model.load_state_dict(result.arrays.to_torch_state_dict())
    run_global_benchmark(global_model, device, algorithm=algorithm)
