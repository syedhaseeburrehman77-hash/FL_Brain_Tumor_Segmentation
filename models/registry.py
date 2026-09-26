from .base import BaseModel
from .unet import UNetModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "3dunet": UNetModel,
}