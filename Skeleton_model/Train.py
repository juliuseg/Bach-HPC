import Skeleton_model.No_Warn
import random
import torch
import os
import numpy as np
from monai.data import Dataset, CacheDataset
from monai.transforms import Compose, ToTensor, EnsureChannelFirst, ScaleIntensity
from monai.networks.nets import UNet
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Data_Generation import generate_skeleton_based_data
# from Skeleton_model.ThickDataset_2 import ThickDataset_2
from Skeleton_model.SkeletonDataset import SkeletonDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class SkeletonDataset(Dataset):
#     def __init__(self, num_samples=100, patch_size=(32, 32, 32), transform=None):
#         self.num_samples = num_samples
#         self.shape = shape
#         self.transform = transform

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         skeleton, broken_skeleton = generate_skeleton_based_data()

#         # Apply transforms generate_skeleton_based_dataseparately
#         if self.transform:
#             skeleton = self.transform(skeleton)
#             broken_skeleton = self.transform(broken_skeleton)

#         return {"image": skeleton, "label": broken_skeleton}


from torch.utils.data import random_split
print ("Dataset is loading")

# Shape:
shape_single_dim = 64
shape = (shape_single_dim,) * 3

# Full dataset
full_dataset = SkeletonDataset(
    num_samples=4096,
    patch_size=shape,
    transform=transform
)



# Split into train/validation sets (80% train, 20% val)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset_raw, val_dataset_raw = random_split(full_dataset, [train_size, val_size])

# Wrap with CacheDataset
train_dataset = CacheDataset(
    data=train_dataset_raw,
    cache_rate=1.0,
    progress=True
)

val_dataset = CacheDataset(
    data=val_dataset_raw,
    cache_rate=1.0,
    progress=True
)



print ("Dataset loaded")

# Move to device (MPS or CUDA)
model = CustomUNet().to(device)


from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from monai.losses import DiceFocalLoss
from monai.optimizers import Novograd
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler, CheckpointSaver
import torch.optim as optim
import os

# Ensure the save directory exists
save_dir = "model_checkpoints"
os.makedirs(save_dir, exist_ok=True)


# Set up training
num_epochs = 15
batch_size = 64

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

from monai.losses import DiceFocalLoss


loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, to_onehot_y=False)



# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# Training loop
print("Training started")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
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
            val_images, val_labels = val_batch["image"].to(device), val_batch["label"].to(device)
            val_outputs = model(val_images)
            loss = loss_fn(val_outputs, val_labels)
            val_loss += loss.item()
            val_batches += 1

    avg_val_loss = val_loss / val_batches

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


print("Training done")

# Explicitly save the model
model_id = random.randint(1000, 10000)
model_path = os.path.join(save_dir, "model_"+str(model_id)+".pt")
torch.save({"model": model.state_dict()}, model_path)

print(f"Model saved successfully to {model_path}")
