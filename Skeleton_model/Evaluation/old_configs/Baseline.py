from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Baseline_model import SkeletonBaselineModel

config = {
    "model_class": SkeletonBaselineModel,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 1,
    "predict_iterations": 1,
    "skeleton": True,
}
