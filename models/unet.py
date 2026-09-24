import torch
from monai.networks.nets import UNet

from .base import BaseModel


class UNetModel(BaseModel):
    """3D U-Net model for FeTS 2022 brain tumor segmentation."""

    def build(self):
        return UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=("INSTANCE", {"affine": True}),
        )


def instance_norm_state_keys() -> set[str]:
    """
    Return state-dict keys belonging to InstanceNorm layers.

    These parameters/statistics are kept local in FedIN-EDAR
    and are not aggregated by the server.
    """
    model = UNetModel().build()

    keys: set[str] = set()

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
            prefix = f"{module_name}." if module_name else ""

            for state_name in module.state_dict().keys():
                keys.add(f"{prefix}{state_name}")

    return keys