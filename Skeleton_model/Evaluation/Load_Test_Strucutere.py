import numpy as np
import torch
from torch.utils.data import Dataset
import os
from monai.data.meta_tensor import MetaTensor
import nibabel as nib
from skimage.morphology import skeletonize


class NarwhalDataset(Dataset):
    def __init__(self, num_samples=100, patch_size=(32, 32, 32),skeleton=False, transform=None, seed=42):
        """
        Dataset that extracts random 32x32x32 patches from the full 3D image.
        
        Args:
            num_samples (int): Number of random patches per epoch.
            patch_size (tuple): Size of the 3D patch to extract.
            transform (callable, optional): Transformation function to apply to patches.
        """
        directory = "/work3/s204427/NarwhalData"

        # Step 1: Load the NIfTI file
        nii_file = "broken_segmentation.nii"
        file_name = os.path.join(directory, nii_file)
        data = nib.load(file_name)

        # Convert the entire data to a numpy array
        #data_array = np.array(data.dataobj)
        # Extract a chunk of the data of size 500x500x500 directly from the NIfTI file
        # Define the start point and size
        start_point = (700, 700, 700)
        size = [1024]*3
        
        # Extract a chunk of the data based on the start point and size
        data_array = np.array(data.dataobj[
            start_point[0]:start_point[0] + size[0],
            start_point[1]:start_point[1] + size[1],
            start_point[2]:start_point[2] + size[2]
        ])
        self.image = data_array
        print ("image shape of dataset:",self.image.shape)
        # divide by max
        self.image = self.image / np.max(self.image)

        # Skeletonize the data
        if skeleton:
            self.image = skeletonize(self.image).astype(np.float32)

        # Print info like uniques of the data
        print(f"Unique values in dataset: {np.unique(self.image)}")
        
        
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.transform = transform

        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        self.shape = self.image.shape  # Full 3D shape (e.g., 1024x1024x1024)
        print(f"Loading image with shape: {self.shape}")

    def __len__(self):
        return self.num_samples

    

    def __getitem__(self, idx):
        """Extracts a random patch from the full image, ensuring proper format."""
        # Get max possible start indices
        max_x = self.shape[0] - self.patch_size[0]
        max_y = self.shape[1] - self.patch_size[1]
        max_z = self.shape[2] - self.patch_size[2]

        # Choose a random start point
        x = self.rng.randint(0, max_x + 1)
        y = self.rng.randint(0, max_y + 1)
        z = self.rng.randint(0, max_z + 1)


        # Extract the 3D patch
        skeleton_patch = self.image[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]

        # Apply transformations (if provided)
        if self.transform:
            skeleton_patch = self.transform(skeleton_patch)

        return {
            "image": to_tensor(skeleton_patch),  # Add channel dim
        }

def to_numpy(data):
        """Convert MetaTensor or Torch Tensor to NumPy before passing to torch.from_numpy()."""
        if isinstance(data, MetaTensor) or isinstance(data, torch.Tensor):
            return data.cpu().numpy()  # Convert to NumPy
        return data  # Already a NumPy array

def to_tensor(data):
    """Convert NumPy array to PyTorch Tensor, ensuring float32 type."""
    data = to_numpy(data)  # Ensure NumPy format
    return torch.from_numpy(data).float()

#[np.newaxis,...]