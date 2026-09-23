"""FeTS 2022 3D MRI Brain Tumor Segmentation: Flower ClientApp."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
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
    model = create_model("unet")

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

    # Get training algorithm
    algorithm = context.run_config["algorithm"]

    trainer = get_trainer(
        algorithm,
        proximal_mu=msg.content["config"].get("proximal_mu", 0.0),
    )

    # Train the model
    train_loss = trainer.train(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

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
    model = create_model("unet")

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

    # Evaluate the model
    eval_metrics = test_fn(
        model,
        valloader,
        device,
    )

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