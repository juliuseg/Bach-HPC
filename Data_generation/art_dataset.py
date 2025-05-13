import Skeleton_model.No_Warn
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from monai.data.meta_tensor import MetaTensor
from scipy.ndimage import label


class Art_Dataset(Dataset):
    def __init__(self, gapsize, skeleton, num_samples=100, patch_size=(32, 32, 32), transform=None,
                 gap_chance=None, num_lines=15, wobble=1.5):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.transform = transform
        self.gapsize = gapsize
        self.skeleton = skeleton
        self.gap_chance = gap_chance
        self.num_lines = num_lines
        self.wobble = wobble

        if skeleton:
            directory = "/work3/s204427/skeleton_data"
        else:
            directory = "/work3/s204427/segmentation_data"

        # Load all datasets into memory
        self.images = []
        self.labels = []
        self.long_holes = []

        
        self.num_datasets = 24

        self.dataset_size = 512

        # Compose new directory name
        gap_chance_str = str(self.gap_chance).replace(".", "")
        directory += f"/s{self.dataset_size}_n{self.num_datasets}_g{self.gapsize}_gs{gap_chance_str}_l{self.num_lines}_w{self.wobble}"

        # directory = "/work3/s204427/skeleton_data/s512_n24_g10"

        if (num_samples == 1):
            # If only one sample, set num_datasets to 1
            self.num_datasets = 1

        print(f"Loading datasets from: {directory}")
        for i in range(self.num_datasets):
            image_path = os.path.join(directory, f"skeleton_data_{i}.npy")
            label_path = os.path.join(directory, f"broken_skeleton_{i}.npy")
            lh_path = os.path.join(directory, f"long_holes_{i}.npy")

            image = np.load(image_path).astype(np.uint8)
            label = np.load(label_path).astype(np.uint8)
            long_hole = np.load(lh_path).astype(np.uint8)

            assert image.shape == label.shape, f"Image and label shapes do not match for dataset {i}!"
            
            self.images.append(image)
            self.labels.append(label)
            self.long_holes.append(long_hole)

            # print(f"Loaded dataset {i} with shape: {image.shape}, unique values: {np.unique(image)}")

        self.shape = self.images[0].shape  # Assume all datasets have the same shape
        print(f"Dataset initialized with {self.num_datasets} datasets, shape: {self.shape}")



    def __len__(self):
        return self.num_samples

    

    def __getitem__(self, idx):
        """Extracts a random patch from a randomly selected dataset."""
        # Randomly select which dataset to use
        if self.num_datasets == 1:
            dataset_idx = 0
        else:
            dataset_idx = np.random.randint(0, self.num_datasets)

        image = self.images[dataset_idx]
        label = self.labels[dataset_idx]
        long_hole = self.long_holes[dataset_idx]

        # Get max possible start indices
        max_x = self.shape[0] - self.patch_size[0]
        max_y = self.shape[1] - self.patch_size[1]
        max_z = self.shape[2] - self.patch_size[2]

        # Choose a random start point
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)
        z = np.random.randint(0, max_z + 1)

        # Extract the 3D patch
        skeleton_patch = image[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]
        broken_skeleton_patch = label[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]
        long_hole_patch = long_hole[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]

        # Apply transformations (if provided)
        if self.transform:
            skeleton_patch = self.transform(skeleton_patch)
            broken_skeleton_patch = self.transform(broken_skeleton_patch)
            long_hole_patch = self.transform(long_hole_patch)

        def ensure_tensor(data):
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).unsqueeze(0).float()
            elif isinstance(data, torch.Tensor):
                return data.unsqueeze(0).float()
            else:
                raise TypeError(f"Unexpected type: {type(data)}")

        return {
            "image": ensure_tensor(skeleton_patch),
            "label": ensure_tensor(broken_skeleton_patch),
            "long_hole": ensure_tensor(long_hole_patch),
        }
    
    # def remove_border_holes(self, hole_skeleton):
    #     connectivity = np.ones((3, 3, 3))
    #     labeled_holes, num_holes = label(hole_skeleton, structure=connectivity)

    #     # Create a single 3D border mask
    #     border = np.zeros_like(hole_skeleton, dtype=bool)
    #     border[0, :, :] = True
    #     border[-1, :, :] = True
    #     border[:, 0, :] = True
    #     border[:, -1, :] = True
    #     border[:, :, 0] = True
    #     border[:, :, -1] = True

    #     # Find labels that touch the border
    #     border_labels = np.unique(labeled_holes[border])  # Get all hole labels that touch the border
    #     border_labels = border_labels[border_labels > 0]  # Remove background (0)

    #     # Create a mask for all holes to remove
    #     remove_mask = np.isin(labeled_holes, border_labels)

    #     # Remove holes touching the border
    #     hole_skeleton[remove_mask] = 0

    #     return hole_skeleton
    


def to_numpy(data):
        """Convert MetaTensor or Torch Tensor to NumPy before passing to torch.from_numpy()."""
        if isinstance(data, MetaTensor) or isinstance(data, torch.Tensor):
            return data.cpu().numpy()  # Convert to NumPy
        return data  # Already a NumPy array

def to_tensor(data):
    """Convert NumPy array to PyTorch Tensor, ensuring float32 type."""
    data = to_numpy(data)  # Ensure NumPy format
    return torch.from_numpy(data).float()

