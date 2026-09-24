"""FeTS 2022 3D MRI Brain Tumor Segmentation: Flower ClientApp."""

import csv
from logging import config
from pathlib import Path
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from task import load_data, test as test_fn
from algorithms import get_trainer
from models import create_model

from models.unet import instance_norm_state_keys
from algorithms.fedindar import (
    PROFILE_BINS,
    REGIONS,
    TUMOR_BURDEN_BIN_EDGES,
    profile_metric_key,
)
# Flower ClientApp
app = ClientApp()
def get_local_instance_norm_state(model):
    """Get this client's local InstanceNorm parameters."""
    local_keys = instance_norm_state_keys()
    state_dict = model.state_dict()

    return {
        key: state_dict[key].detach().cpu().clone()
        for key in local_keys
        if key in state_dict
    }

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
    local_in_state = get_local_instance_norm_state(model)
    # Select device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    for key, value in local_in_state.items():
        if key in model.state_dict():
            model.state_dict()[key].copy_(value.to(device))

       # Load local training data
    partition_id = int(context.node_config["partition-id"])
    config = msg.content["config"]
    profile_only = bool(
        config.get("fedindar-profile-only", False)
    )

    if profile_only:
        _, valloader = load_data(
            partition_id,
            context,
        )

        profile_metrics = {
            profile_metric_key(region, i): 0.0
            for region in REGIONS
            for i in range(PROFILE_BINS)
        }

        for region in REGIONS:
            counts = np.zeros(PROFILE_BINS, dtype=np.float64)

            for batch in valloader:
                labels = batch["label"]
                if labels.ndim == 5:
                    labels = labels[:, 0]
                labels = labels.detach().cpu().numpy()
                for label in labels:
                    if region == "wt":
                        mask = label > 0
                    elif region == "tc":
                        mask = np.logical_or(label == 1, label == 3)
                    else:
                        mask = label == 3

                    burden = float(mask.sum()) / max(label.size, 1)
                    counts += np.histogram(
                        [burden], bins=TUMOR_BURDEN_BIN_EDGES
                    )[0]

            if counts.sum() > 0:
                counts /= counts.sum()
            for i, value in enumerate(counts):
                profile_metrics[profile_metric_key(region, i)] = float(value)

        print(
            f"[Client {partition_id}] ---> "
            "FedIN-EDAR profile collected. "
            "Round 1 training skipped.",
            flush=True,
        )
        local_in_dir = Path("artifacts/local_instance_norm")
        local_in_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            get_local_instance_norm_state(model),
            local_in_dir / f"client_{partition_id}.pt",
        )
        return Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord(
                        model.state_dict()
                    ),
                    "metrics": MetricRecord(
                        {
                            **profile_metrics,
                            "num-examples": len(
                                valloader.dataset
                            ),
                            "train_loss": 0.0,
                        }
                    ),
                }
            ),
            reply_to=msg,
        )

    trainloader, _ = load_data(
        partition_id,
        context,
    )
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
    local_in_state = {
        key: model.state_dict()[key].detach().cpu().clone()
        for key in instance_norm_state_keys()
        if key in model.state_dict()
    }
    local_in_dir = Path("artifacts/local_instance_norm")
    local_in_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        local_in_state,
        local_in_dir / f"client_{partition_id}.pt",
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
    partition_id = int(context.node_config["partition-id"])
    # Select device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    # Load the weights received from the server
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )
    local_in_dir = Path("artifacts/local_instance_norm")
    local_in_file = local_in_dir / f"client_{partition_id}.pt"

    if local_in_file.exists():
        local_in_state = torch.load(local_in_file, map_location=device)

        for key, value in local_in_state.items():
            if key in model.state_dict():
                model.state_dict()[key].copy_(value.to(device))

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