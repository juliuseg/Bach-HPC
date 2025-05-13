from Skeleton_model.model import CustomUNet, transform

Segmentation_CELoss = {
    "model_path": "model_checkpoints/model_seg_CEDice.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 10,
    "predict_iterations": 1,
    "skeleton": False,
}
