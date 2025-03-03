import random
import torch
import os
import numpy as np
from monai.data import Dataset, CacheDataset
from monai.transforms import Compose, ToTensor, EnsureChannelFirst, ScaleIntensity
from monai.networks.nets import UNet
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Data_Generation import generate_skeleton_based_data
from Skeleton_model.SkeletonDataset import SkeletonDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class SkeletonDataset(Dataset):
#     def __init__(self, num_samples=100, shape=(32, 32, 32), transform=None):
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


print ("Dataset is loading")
# Use CacheDataset
shape_single_dim = 64
shape = (shape_single_dim,) * 3
dataset = CacheDataset(
    data=SkeletonDataset(
    num_samples=1024, 
    patch_size=shape, 
    transform=transform), 
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


# DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

from monai.losses import DiceFocalLoss


loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, to_onehot_y=False)

# Set up training
num_epochs = 50
batch_size = 32


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training started")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}")

print("Training done")

# Explicitly save the model
model_id = random.randint(1000, 10000)
model_path = os.path.join(save_dir, "model_"+str(model_id)+".pt")
torch.save({"model": model.state_dict()}, model_path)

print(f"Model saved successfully to {model_path}")





