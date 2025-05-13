# configs/all_configs.py
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Baseline_model import SkeletonBaselineModel
from MiccaiModel.backbone.unet3d import UNet_ODT



##### NEW ######
# model_g5_sTrue_gc03_l15_w15_dice_20250511-215632pt.pt

Skeleton_g10_p1_gc03_l15_w15_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l15_w15_dice_20250513-152747.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

Skeleton_g10_p1_gc03_l15_w15_conn = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l15_w15_conn_20250513-160111.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}





##### BASELINE #####
Baseline = {
    "model_class": SkeletonBaselineModel,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

##### MICCAI #####
Miccai_seg = {
    "model_path": "MiccaiModel/weights/model_weights_best.pth",
    "model_class": UNet_ODT,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": False,
}


Miccai_skel = {
    "model_path": "MiccaiModel/weights/model_weights_best.pth",
    "model_class": UNet_ODT,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}




# ##### SEGMENTATION #####


Segmentation_g10_p1 = {
    "model_path": "model_checkpoints/model_g10_sFalse_p1.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": False,
}

Segmentation_g10_p5 = {
    "model_path": "model_checkpoints/model_g10_sFalse_p5.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": False,
}

Segmentation_g20_p1 = {
    "model_path": "model_checkpoints/model_g20_sFalse_p1.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": False,
}

##### SKELETON GAP CHANCE #####

Skeleton_g10_p1_gc01 = {
    "model_path": "model_checkpoints/model_g10_sTrue_p1_gc01.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

Skeleton_g10_p1_gc05 = {
    "model_path": "model_checkpoints/model_g10_sTrue_p1_gc05.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}


##### SKELETON #####


Skeleton_g10_p1_gc03 = {
    "model_path": "model_checkpoints/model_g10_sTrue_p1_gc03.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

Skeleton_g10_p1_gc03_conloss = {
    "model_path": "model_checkpoints/model_g10_sTrue_p1_gc03_conloss.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}



Skeleton_g10_p10_gc03 = {
    "model_path": "model_checkpoints/model_g10_sTrue_p10_gc03.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

Skeleton_g20_p1_gc03 = {
    "model_path": "model_checkpoints/model_g20_sTrue_p1_gc03.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}






##### OUTDATED #####

# Segmentation_CELoss = {
#     "model_path": "model_checkpoints/model_seg_CEDice.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": False,
# }

# Segmentation = {
#     "model_path": "model_checkpoints/model_thick.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": False,
# }

# Skeleton_CEDice = {
#     "model_path": "model_checkpoints/model_CEDice.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 1,
#     "predict_iterations": 1,
#     "skeleton": True,
# }

# Skeleton = {
#     "model_path": "model_checkpoints/model_dilation0.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 1,
#     "predict_iterations": 1,
#     "skeleton": True,
# }

# Skeleton_DiceFocal = {
#     "model_path": "model_checkpoints/model_dice_focal.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 1,
#     "predict_iterations": 1,
#     "skeleton": True,
# }