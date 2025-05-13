import Skeleton_model.No_Warn
from Skeleton_model.model import CustomUNet, transform
# from Skeleton_model.SkeletonDataset import SkeletonDataset
# from Skeleton_model.ThickDataset_2 import ThickDataset_2
from Data_generation.art_dataset import Art_Dataset

# Import Python modules
import os
import sys
import random
import numpy as np
import time

# Import PyTorch modules
import torch
import torch.optim as optim
from torch.utils.data import random_split


# Import MONAI modules
from monai.data import CacheDataset
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from monai.losses import DiceFocalLoss

def conn_loss(pred, target, mask, lambda_param=1.0, omega=1.0, eps=1e-6):
    """
    Connectivity-aware loss from MICCAI 2024 paper.
    
    Args:
        pred: predicted tensor after sigmoid [B, 1, D, H, W]
        target: ground truth tensor [B, 1, D, H, W]
        mask: binary mask M where breaks are expected [B, 1, D, H, W]
        lambda_param: weight for the penalty on high predictions in breaks
        omega: weight for the preservation loss
        eps: small value to avoid divide-by-zero

    Returns:
        scalar loss
    """

    abs_diff = torch.abs(pred - target)
    delta = (pred > 0.5).float()  # indicator function δ(Y > 0.5)

    # L_R: restoring loss in masked region
    L_R_numerator = (mask * abs_diff + lambda_param * delta).sum()
    L_R_denominator = mask.sum() + eps
    L_R = L_R_numerator / L_R_denominator

    # L_P: preservation loss outside the mask
    one_minus_mask = 1.0 - mask
    L_P_numerator = (one_minus_mask * abs_diff).sum()
    L_P_denominator = one_minus_mask.sum() + eps
    L_P = L_P_numerator / L_P_denominator

    return L_R + omega * L_P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print ("Dataset loading")

# Shape:
shape_single_dim = 64
shape = (shape_single_dim,) * 3

# get gapsize and skeleton from arguments. gapsize first which is just an int. Then skeleton which is a boolean. make it read an int and convert that int to a boolean
if len(sys.argv) < 7:
    raise ValueError("Usage: python Train.py <gapsize> <skeleton> <gap_chance> <num_lines> <wobble> <loss_type>")
else:
    gapsize = int(sys.argv[1])
    skeleton = bool(int(sys.argv[2]))
    gap_chance = float(sys.argv[3])
    num_lines = int(sys.argv[4])
    wobble = float(sys.argv[5])
    loss_type = sys.argv[6].lower()  # "conn" or "dice" or "focal"
    print(f"gapsize: {gapsize}, skeleton: {skeleton}, gap_chance: {gap_chance}, num_lines: {num_lines}, wobble: {wobble}", f"loss_type: {loss_type}")

# Full dataset
full_dataset = Art_Dataset(
    num_samples=16_384,
    patch_size=shape,
    transform=transform,
    gapsize=gapsize,
    gap_chance=gap_chance,
    skeleton=skeleton,
    num_lines=num_lines,
    wobble=wobble
)


# dataset_info = full_dataset.get_info()
# print (f"Dataset info: {dataset_info}")




# Split into train/validation sets (80% train, 20% val)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset_raw, val_dataset_raw = random_split(full_dataset, [train_size, val_size])

progress = False
# Wrap with CacheDataset
train_dataset = CacheDataset(
    data=train_dataset_raw,
    cache_rate=1.0,
    progress=progress
)
val_dataset = CacheDataset(
    data=val_dataset_raw,
    cache_rate=1.0,
    progress=progress
)

print ("Dataset loaded")


######################

# Model
model = CustomUNet().to(device)

# Ensure the save directory exists
save_dir = "model_checkpoints"
os.makedirs(save_dir, exist_ok=True)


# Training parameters
num_epochs_max = 50 ########### CHANGE BACK TO 50 ###########
batch_size = 64
patience = 4  # Early stopping patience

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Loss and optimizer
# loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, to_onehot_y=False)
# loss_fn = DiceCELoss(sigmoid=True)


if loss_type == "dice":
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, to_onehot_y=False)
elif loss_type == "focal":
    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, to_onehot_y=False)
elif loss_type == "conn":
    loss_fn = None  # conn_loss is used separately
else:
    raise ValueError("loss_type must be one of: 'conn', 'dice', 'focal'")

# Optimizer
learning_rate = 0.0003
weight_decay = 1e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



# Training loop
print(f"Training started with: lr={learning_rate}, weight_decay={weight_decay}, batch_size={batch_size}, loss_type={loss_type}")

best_val_loss = float("inf")
epochs_no_improve = 0
best_model_state = None

for epoch in range(num_epochs_max):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        image = batch["image"].to(device)     # input with gap
        mask = batch["label"].to(device)      # label = only the gap (used as mask)
        target = torch.clamp(image + mask, max=1.0)  # full skeleton

        # print number sum of image and mask
        # print(f"image sum: {image.sum()}, mask sum: {mask.sum()}, target sum: {target.sum()}")

        optimizer.zero_grad()
        outputs = model(image)
        outputs_sigmoid = torch.sigmoid(outputs)

        if loss_type == "conn":
            loss = conn_loss(outputs_sigmoid, mask, mask, lambda_param=1.0)
        else:
            loss = loss_fn(outputs, mask)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_image = val_batch["image"].to(device)
            val_mask = val_batch["label"].to(device)
            val_target = torch.clamp(val_image + val_mask, max=1.0)

            val_outputs = model(val_image)
            val_outputs_sigmoid = torch.sigmoid(val_outputs)

            if loss_type == "conn":
                loss = conn_loss(val_outputs_sigmoid, val_mask, val_mask, lambda_param=1.0)
            else:
                loss = loss_fn(val_outputs, val_mask)

            val_loss += loss.item()
            val_batches += 1

    avg_val_loss = val_loss / val_batches

    print(f"Epoch {epoch + 1}/{num_epochs_max} - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Track best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Early stopping
    if epochs_no_improve >= patience:
        print("⛔ Early stopping triggered")
        break

print("Training done")

# Save the best model

model_id = f"g{gapsize}_s{skeleton}_gc{gap_chance}_l{num_lines}_w{wobble}_{loss_type}"
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_name = f"model_{model_id}_{timestamp}"
# remove . from model_name
model_name = model_name.replace(".", "")
model_name += ".pt"
model_path = os.path.join(save_dir, model_name)

save_dict = {
    "model_state_dict": best_model_state,
    "val_loss": best_val_loss,
    "epoch": epoch + 1,
    "model_architecture": str(model),  # or model.__class__.__name__
    "optimizer_state_dict": optimizer.state_dict(),
    "loss_fn": loss_type,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "patience": patience,
    "timestamp": timestamp
}

torch.save(save_dict, model_path)
print(f"Best model + metadata saved to {model_path}")