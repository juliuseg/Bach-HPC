# configs/all_configs.py
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Baseline_model import SkeletonBaselineModel
from MiccaiModel.backbone.unet3d import UNet_ODT



##### NEW ######

# Default parameters
Skeleton_g10_p1_gc03_l15_w15_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l15_w15_dice_20250513-203401.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Lower gap size
Skeleton_g5_p1_gc03_l15_w15_dice = {
    "model_path": "model_checkpoints/model_g5_sTrue_gc03_l15_w15_dice_20250513-205911.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Higher gap size
Skeleton_g20_p1_gc03_l15_w15_dice = {
    "model_path": "model_checkpoints/model_g20_sTrue_gc03_l15_w15_dice_20250513-212532.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Lower gap chance
Skeleton_g10_p1_gc01_l15_w15_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc01_l15_w15_dice_20250513-215823.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Higher gap chance
Skeleton_g10_p1_gc05_l15_w15_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc05_l15_w15_dice_20250513-222336.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Lower number of lines
Skeleton_g10_p1_gc03_l7_w15_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l7_w15_dice_20250513-224457.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Higher number of lines
Skeleton_g10_p1_gc03_l30_w15_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l30_w15_dice_20250513-232755.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Wobble of 1.0
Skeleton_g10_p1_gc03_l15_w10_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l15_w10_dice_20250513-235517.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Wobble of 2.0
Skeleton_g10_p1_gc03_l15_w20_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l15_w20_dice_20250514-002541.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Wobble of 3.0
Skeleton_g10_p1_gc03_l15_w30_dice = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l15_w30_dice_20250514-011407.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Loss function: ConLoss
Skeleton_g10_p1_gc03_l15_w15_conloss = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l15_w15_conn_20250514-014957.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# Loss function: FocalLoss
Skeleton_g10_p1_gc03_l15_w15_focal = {
    "model_path": "model_checkpoints/model_g10_sTrue_gc03_l15_w15_focal_20250514-021538.pt",
    "model_class": CustomUNet,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

# One layer deeper:
Skeleton_g10_p1_gc03_l15_w15_dice_layer3 = {
    "model_path": "model_checkpoints/model_deeper_model_20250514-204941.pt",
    "model_class": lambda: CustomUNet(channels=(32, 64, 128, 256), strides=(2, 2, 2)),  # extra layer
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

Skeleton_g10_p1_gc03_l15_w15_dice_kernel5 = {
    "model_path": "model_checkpoints/model_kernel_5_model_20250514-211600.pt",
    "model_class": lambda: CustomUNet(kernel_size=5),  # non-default kernel
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}



#### BASELINE #####
Baseline_5 = {
    "model_class": SkeletonBaselineModel,
    "search_radius": 5,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

Baseline_10 = {
    "model_class": SkeletonBaselineModel,
    "search_radius": 10,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}

Baseline_20 = {
    "model_class": SkeletonBaselineModel,
    "search_radius": 20,
    "transform": transform,
    "patch_size": (256, 256, 256),
    "num_iterations": 32,
    "predict_iterations": 1,
    "skeleton": True,
}


##### MICCAI #####
# Miccai_seg = {
#     "model_path": "MiccaiModel/weights/model_weights_best.pth",
#     "model_class": UNet_ODT,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": False,
# }


# Miccai_skel = {
#     "model_path": "MiccaiModel/weights/model_weights_best.pth",
#     "model_class": UNet_ODT,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": True,
# }








##### OUTDATED #####



# # ##### SEGMENTATION #####


# Segmentation_g10_p1 = {
#     "model_path": "model_checkpoints/model_g10_sFalse_p1.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": False,
# }

# Segmentation_g10_p5 = {
#     "model_path": "model_checkpoints/model_g10_sFalse_p5.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": False,
# }

# Segmentation_g20_p1 = {
#     "model_path": "model_checkpoints/model_g20_sFalse_p1.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": False,
# }

# ##### SKELETON GAP CHANCE #####

# Skeleton_g10_p1_gc01 = {
#     "model_path": "model_checkpoints/model_g10_sTrue_p1_gc01.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": True,
# }

# Skeleton_g10_p1_gc05 = {
#     "model_path": "model_checkpoints/model_g10_sTrue_p1_gc05.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": True,
# }


# ##### SKELETON #####


# Skeleton_g10_p1_gc03 = {
#     "model_path": "model_checkpoints/model_g10_sTrue_p1_gc03.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": True,
# }

# Skeleton_g10_p1_gc03_conloss = {
#     "model_path": "model_checkpoints/model_g10_sTrue_p1_gc03_conloss.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": True,
# }



# Skeleton_g10_p10_gc03 = {
#     "model_path": "model_checkpoints/model_g10_sTrue_p10_gc03.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": True,
# }

# Skeleton_g20_p1_gc03 = {
#     "model_path": "model_checkpoints/model_g20_sTrue_p1_gc03.pt",
#     "model_class": CustomUNet,
#     "transform": transform,
#     "patch_size": (256, 256, 256),
#     "num_iterations": 32,
#     "predict_iterations": 1,
#     "skeleton": True,
# }







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