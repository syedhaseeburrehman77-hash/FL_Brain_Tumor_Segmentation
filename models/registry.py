from .base import BaseModel
from .unet import UNetModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "unet": UNetModel,
    "3dunet": UNetModel,
    "3DUnet": UNetModel,
}