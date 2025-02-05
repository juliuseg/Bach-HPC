from monai.networks.nets import UNet
import torch.nn as nn
from monai.transforms import Compose, ScaleIntensity, ToTensor

# Define a MONAI-based 3D U-Net with increased capacity and smooth output
class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=1, 
            channels=(32, 64, 128),  
            strides=(2, 2),  
        )

    def forward(self, x):
        return self.unet(x)  # Remove Sigmoid; DiceLoss applies it

# Define transforms **without applying them to a dict**
transform = Compose([
    ScaleIntensity(),  # Normalize values between 0 and 1
    ToTensor()
])