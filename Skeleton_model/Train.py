import torch
import os
import numpy as np
from monai.data import Dataset, CacheDataset
from monai.transforms import Compose, ToTensor, EnsureChannelFirst, ScaleIntensity
from monai.networks.nets import UNet
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.test_art_data import generate_skeleton_based_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkeletonDataset(Dataset):
    def __init__(self, num_samples=100, shape=(32, 32, 32), transform=None):
        self.num_samples = num_samples
        self.shape = shape
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        skeleton, broken_skeleton = generate_skeleton_based_data(self.shape)

        # Apply transforms separately
        if self.transform:
            skeleton = self.transform(skeleton)
            broken_skeleton = self.transform(broken_skeleton)

        return {"image": broken_skeleton, "label": skeleton}


# Use CacheDataset
shape = (32, 32, 32)
dataset = CacheDataset(data=SkeletonDataset(num_samples=8192, shape=shape, transform=transform), cache_rate=1.0)

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
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Use the modified loss function that ignores background
#loss_fn = MaskedDiceFocalLoss(to_onehot_y=False, sigmoid=True, squared_pred=True, reduction="mean")
#loss_fn = DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=False)
from monai.losses import DiceFocalLoss

loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, to_onehot_y=False)


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Define MONAI Trainer (⚡ Updated paths)
trainer = SupervisedTrainer(
    device=device,
    max_epochs=10,
    train_data_loader=dataloader,
    network=model,
    optimizer=optimizer,
    loss_function=loss_fn,
    train_handlers=[
        StatsHandler(name="train_loss", output_transform=lambda x: x[0]),
        CheckpointSaver(
            save_dir="model_checkpoints",  # Ensure correct save path
            save_dict={"model": model},
            save_final=True,  # Ensures a final model is saved
            save_key_metric=False  # Disables key metric saving (optional)
        ),
    ],
)


# Start training
trainer.run()


# Explicitly save the model
model_path = os.path.join(save_dir, "model.pt")
torch.save({"model": model.state_dict()}, model_path)






