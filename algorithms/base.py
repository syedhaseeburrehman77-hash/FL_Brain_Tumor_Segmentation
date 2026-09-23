from abc import ABC, abstractmethod
import torch
from monai.losses import DiceCELoss

class BaseTrainer(ABC):
    """Common interface every algorithm trainer must implement."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def compute_loss(self, model, outputs, labels, criterion) -> torch.Tensor:
        """Return the (possibly augmented) loss for a single batch."""
        raise NotImplementedError

    def on_train_start(self, model):
        """Hook called once before local training begins.
        Override if the algorithm needs to snapshot state (e.g. global params)."""
        pass

    def train(self, model, trainloader, epochs, lr, device):
        """Shared training loop — same for every algorithm.
        Only compute_loss / on_train_start differ per strategy."""
        self.on_train_start(model)

        model.to(device)
        model.train()
        criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        total_loss = 0.0
        num_batches = 0
        for _ in range(epochs):
            for batch in trainloader:
                images, labels = batch["image"].to(device), batch["label"].to(device).long()

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = self.compute_loss(model, outputs, labels, criterion)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)