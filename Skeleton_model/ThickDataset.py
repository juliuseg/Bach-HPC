import numpy as np
import torch
from torch.utils.data import Dataset
import os
from monai.data.meta_tensor import MetaTensor
from scipy.ndimage import label


class ThickDataset(Dataset):
    def __init__(self, num_samples=100, patch_size=(32, 32, 32), transform=None):
        """
        Dataset that extracts random 32x32x32 patches from the full 3D image and label.
        
        Args:
            num_samples (int): Number of random patches per epoch.
            patch_size (tuple): Size of the 3D patch to extract.
            transform (callable, optional): Transformation function to apply to patches.
        """
        directory = "/work3/s204427"

        image_path = os.path.join(directory, "skeleton_data_thick_1024.npy") # Remove # on [0] !!! ON LOAD

        label_path = os.path.join(directory, "broken_skeleton_thick_1024.npy") #  Remove # on [0] !!! ON LOAD


        self.image = np.load(image_path)[0].astype(np.uint8)  # Load full 3D image
        self.label = np.load(label_path)[0].astype(np.uint8)  # Load full 3D label

        # # divide by 255
        # self.image = self.image / 255
        # self.label = self.label / 255

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

        broken_skeleton_patch = self.remove_border_holes(broken_skeleton_patch)

        # Apply transformations (if provided)
        if self.transform:
            skeleton_patch = self.transform(skeleton_patch)
            broken_skeleton_patch = self.transform(broken_skeleton_patch)

        return {
            "image": to_tensor(skeleton_patch[np.newaxis, ...]),  # Add channel dim
            "label": to_tensor(broken_skeleton_patch[np.newaxis, ...])
        }
    
    def remove_border_holes(self, hole_skeleton):
        connectivity = np.ones((3, 3, 3))
        labeled_holes, num_holes = label(hole_skeleton, structure=connectivity)

        # Create a single 3D border mask
        border = np.zeros_like(hole_skeleton, dtype=bool)
        border[0, :, :] = True
        border[-1, :, :] = True
        border[:, 0, :] = True
        border[:, -1, :] = True
        border[:, :, 0] = True
        border[:, :, -1] = True

        # Find labels that touch the border
        border_labels = np.unique(labeled_holes[border])  # Get all hole labels that touch the border
        border_labels = border_labels[border_labels > 0]  # Remove background (0)

        # Create a mask for all holes to remove
        remove_mask = np.isin(labeled_holes, border_labels)

        # Remove holes touching the border
        hole_skeleton[remove_mask] = 0

        return hole_skeleton

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