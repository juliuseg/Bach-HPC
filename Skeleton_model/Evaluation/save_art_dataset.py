import Skeleton_model.No_Warn
import os
import pickle

from Skeleton_model.model import transform
from Data_generation.art_dataset import Art_Dataset

# Parameters
output_dir = "inference_results"
os.makedirs(output_dir, exist_ok=True)

shape = (256, 256, 256)
gapsize = 10
skeleton = False
gap_chance = 0.3
num_lines = 15
wobble = 1.5

# Create dataset with 1 sample
dataset = Art_Dataset(
    num_samples=1,
    patch_size=shape,
    transform=transform,
    gapsize=gapsize,
    gap_chance=gap_chance,
    skeleton=skeleton,
    num_lines=num_lines,
    wobble=wobble
)

# Extract the first sample
sample = dataset[0]

# Convert tensors to numpy arrays
sample_dict = {
    "actual_skeletons": sample["image"][0].numpy(),
    "predicted_skeletons": sample["label"][0].numpy()
}
if "long_hole" in sample:
    sample_dict["long_hole"] = sample["long_hole"][0].numpy()

# Save as pickle
save_path = os.path.join(output_dir, "sample_data.pkl")
with open(save_path, "wb") as f:
    pickle.dump(sample_dict, f)

print(f"Saved sample to {save_path}")
