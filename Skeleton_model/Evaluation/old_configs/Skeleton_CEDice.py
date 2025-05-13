from Skeleton_model.model import CustomUNet, transform

Skeleton_CEDice = {
    "model_path": "model_checkpoints/model_CEDice.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 1,
    "predict_iterations": 1,
    "skeleton": True,
}
