"""FeTS 2022 3D MRI Brain Tumor Segmentation: Flower ClientApp."""

from multiprocessing import context

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import load_data, test as test_fn

from algorithms import get_trainer

from ML_model import build_model

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the 3D UNet on local institutional MRI data."""

    # Load the model and initialize it with the received weights
    model = build_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    trainloader, _ = load_data(partition_id, context)

    # Pull algorithm-specific kwargs (e.g. proximal_mu) straight from config —
    # trainers that don't need them just ignore extras via **kwargs
    algorithm = context.run_config["algorithm"]
    trainer = get_trainer(
        algorithm,
        proximal_mu=msg.content["config"].get("proximal_mu", 0.0),
    )

    # Call the training function
    train_loss = trainer.train(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the 3D UNet on local institutional MRI data."""

    # Load the model and initialize it with the received weights
    model = build_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    _, valloader = load_data(partition_id, context)

    # Call 3D sliding-window evaluation (Dice ET/TC/WT, HD95)
    eval_metrics = test_fn(model, valloader, device)

    # Return all segmentation metrics to Flower
    metrics = {
        **eval_metrics,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
