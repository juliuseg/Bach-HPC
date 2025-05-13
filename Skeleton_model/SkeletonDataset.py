import numpy as np
import torch
from torch.utils.data import Dataset
import os
from monai.data.meta_tensor import MetaTensor


class SkeletonDataset(Dataset):
    def __init__(self, num_samples=100, patch_size=(32, 32, 32), transform=None):
        """
        Dataset that extracts random 32x32x32 patches from the full 3D image and label.
        
        Args:
            num_samples (int): Number of random patches per epoch.
            patch_size (tuple): Size of the 3D patch to extract.
            transform (callable, optional): Transformation function to apply to patches.
        """
        directory = "/work3/s204427"



        self.image_path = os.path.join(directory, "1024_skeleton_only.npy") # Remove # on [0] !!! ON LOAD
        self.label_path = os.path.join(directory, "1024_broken_skeleton_only.npy") #  Remove # on [0] !!! ON LOAD

        self.image = np.load(self.image_path).astype(np.uint8)[0]  # Load full 3D image
        self.label = np.load(self.label_path).astype(np.uint8)[0]  # Load full 3D label

        # image_path = os.path.join(directory, "narwhal_data_patch.npy") # Remove # on [0] !!! ON LOAD
        # label_path = os.path.join(directory, "narwhal_data_patch.npy") #  Remove # on [0] !!! ON LOAD

        # self.image = np.load(image_path).astype(np.uint8)  # Load full 3D image
        # self.label = np.load(label_path).astype(np.uint8)  # Load full 3D label
        
        print ("image shape:",self.image.shape)

        
        # divide by max
        self.image = self.image / np.max(self.image)
        self.label = self.label / np.max(self.label)

        # Print info like uniques of the data
        print(f"Unique values in image: {np.unique(self.image)}")
        
        assert self.image.shape == self.label.shape, "Image and label must have the same shape!"
        
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.transform = transform

        self.shape = self.image.shape  # Full 3D shape (e.g., 1024x1024x1024)
        print(f"Loading image with shape: {self.shape}")

    def __len__(self):
        return self.num_samples

    

    def __getitem__(self, idx):
        """Extracts a random patch from the full image and label, ensuring proper format."""
        # Get max possible start indices
        max_x = self.shape[0] - self.patch_size[0]
        max_y = self.shape[1] - self.patch_size[1]
        max_z = self.shape[2] - self.patch_size[2]

        # Choose a random start point
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)
        z = np.random.randint(0, max_z + 1)

        # Extract the 3D patch
        skeleton_patch = self.image[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]
        broken_skeleton_patch = self.label[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]

        # Apply transformations (if provided)
        if self.transform:
            skeleton_patch = self.transform(skeleton_patch)
            broken_skeleton_patch = self.transform(broken_skeleton_patch)

        return {
            "image": to_tensor(skeleton_patch[np.newaxis, ...]),  # Add channel dim
            "label": to_tensor(broken_skeleton_patch[np.newaxis, ...])
        }
    
    def get_info(self):
        """Returns the shape of the full image and label."""
        info = {
            "shape": self.image.shape,
            "patch_size": self.patch_size,
            "image_path": self.image_path,
            "label_path": self.label_path
        }
        return self.shape

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