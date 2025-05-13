from Skeleton_model.model import CustomUNet, transform
from MiccaiModel.backbone.unet3d import UNet_ODT

Miccai_seg = {
    "model_path": "MiccaiModel/weights/model_weights_best.pth",
    "model_class": UNet_ODT,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 10,
    "predict_iterations": 1,
    "skeleton": False,
}
