from Skeleton_model.model import CustomUNet, transform

Segmentation_g10_p1 = {
    "model_path": "model_checkpoints/model_g10_sFalse_p1.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 10,
    "predict_iterations": 1,
    "skeleton": False,
}
