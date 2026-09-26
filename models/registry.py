from .base import BaseModel
from .unet import UNetModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "3DUnet": UNetModel,
}