import torch
from .base import BaseTrainer

class FedProxTrainer(BaseTrainer):
    """FedAvg + proximal term penalizing drift from the global model."""

    def __init__(self, proximal_mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        self.global_params = {}

    def on_train_start(self, model):
        # Snapshot global params BEFORE local training mutates them
        self.global_params = {name: p.detach().clone() for name, p in model.named_parameters() if "norm" not in name.lower()}

    def compute_loss(self, model, outputs, labels, criterion):
        loss = criterion(outputs, labels)

        proximal_term = sum(
            (p - self.global_params[name]).pow(2).sum()
            for name, p in model.named_parameters()
            if name in self.global_params
        )
        loss += (self.proximal_mu / 2) * proximal_term
        return loss