from Skeleton_model.model import CustomUNet, transform

Skeleton_g20_p5 = {
    "model_path": "model_checkpoints/model_g20_sTrue_p1.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 10,
    "predict_iterations": 1,
    "skeleton": False,
}
