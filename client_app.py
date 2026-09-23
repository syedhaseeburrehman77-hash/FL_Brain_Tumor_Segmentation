"""FeTS 2022 3D MRI Brain Tumor Segmentation: Flower ClientApp."""

import csv
from pathlib import Path
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from task import load_data, test as test_fn
from algorithms import get_trainer
from models import create_model

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the 3D U-Net on local institutional MRI data."""

    # Create the model using the model registry
    model_name = context.run_config.get("model_name", "unet").lower()
    model = create_model(model_name)

    # Load the weights received from the server
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    # Select device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Load local training data
    partition_id = int(context.node_config["partition-id"])
    trainloader, _ = load_data(partition_id, context)
    print(f"\n[Client {partition_id}] ---> Loaded {len(trainloader.dataset)} patients from FeTS institution partition.", flush=True)

    # Get training algorithm
    algorithm = context.run_config["algorithm"]

    trainer = get_trainer(
        algorithm,
        proximal_mu=float(msg.content["config"].get("proximal_mu", context.run_config.get("proximal_mu", 0.01))),
    )

    print(f"[Client {partition_id}] ---> Starting 3D U-Net training on {device}...", flush=True)
    # Train the model
    train_loss = trainer.train(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )
    print(f"[Client {partition_id}] ---> Training completed! Loss: {train_loss:.4f}", flush=True)

    state = context.state.get("client_state", ConfigRecord({}))
    current_round = int(state.get("current_round", 1))
    context.state["client_state"] = ConfigRecord({
        "last_train_loss": float(train_loss),
        "current_round": current_round,
    })
    # Return updated model and training metrics
    model_record = ArrayRecord(model.state_dict())

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }

    metric_record = MetricRecord(metrics)

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metric_record,
        }
    )

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the 3D U-Net on local institutional MRI data."""

    # Create the model using the model registry
    model_name = context.run_config.get("model_name", "unet").lower()
    model = create_model(model_name)

    # Load the weights received from the server
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    # Select device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Load local validation data
    partition_id = int(context.node_config["partition-id"])
    _, valloader = load_data(partition_id, context)
    print(f"\n[Client {partition_id}] ---> Evaluating on {len(valloader.dataset)} local validation patients...", flush=True)

    # Evaluate the model
    eval_metrics = test_fn(
        model,
        valloader,
        device,
    )
    print(f"[Client {partition_id}] ---> Evaluation completed! Loss: {eval_metrics.get('eval_loss', 0.0):.4f} | Dice WT: {eval_metrics.get('dice_wt', 0.0):.4f}", flush=True)

    # Save client metrics to artifacts/client_history.csv
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_file = artifacts_dir / "client_history.csv"
    file_exists = csv_file.exists()

    state = context.state.get("client_state", ConfigRecord({}))
    train_loss = float(state.get("last_train_loss", 0.0))
    current_round = int(state.get("current_round", 1))

    context.state["client_state"] = ConfigRecord({
    "last_train_loss": train_loss,
    "current_round": current_round + 1,
})
    algorithm = str(context.run_config.get("algorithm", "fedavg"))

    row = {
        "strategy": algorithm,
        "round": current_round,
        "institution_id": partition_id,
        "train_loss": train_loss,
        "eval_loss": eval_metrics.get("eval_loss", 0.0),
        "dice_et": eval_metrics.get("dice_et", 0.0),
        "dice_tc": eval_metrics.get("dice_tc", 0.0),
        "dice_wt": eval_metrics.get("dice_wt", 0.0),
        "hd95_et": eval_metrics.get("hd95_et", 0.0),
        "hd95_tc": eval_metrics.get("hd95_tc", 0.0),
        "hd95_wt": eval_metrics.get("hd95_wt", 0.0),
        "num_examples": len(valloader.dataset),
    }
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Return evaluation metrics
    metrics = {
        **eval_metrics,
        "num-examples": len(valloader.dataset),
    }

    metric_record = MetricRecord(metrics)

    content = RecordDict(
        {
            "metrics": metric_record,
        }
    )
    return Message(content=content, reply_to=msg)