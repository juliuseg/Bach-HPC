import Skeleton_model.No_Warn
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Data_Generation import generate_skeleton_based_data
from Skeleton_model.Evaluate_utils import remove_non_touching_components
#from Skeleton_model.SkeletonDataset import SkeletonDataset
from Skeleton_model.ThickDataset_2 import ThickDataset_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the plots directory exists
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Define model checkpoint path
model_id = "thick"
model_path = f"model_checkpoints/model_{model_id}.pt"

# Ensure the file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model checkpoint not found at: {model_path}")

# Load the trained model
checkpoint = torch.load(model_path)
model = CustomUNet()
model.load_state_dict(checkpoint["model"])
model.to(device)  # Move to GPU/CPU
model.eval()

print(f"✅ Model loaded successfully from {model_path}")

# Define the shape of the 3D data
shape_single_dim = 64
shape = (shape_single_dim,) * 3
amount_of_data = 1

# Initialize arrays to store 10 sets of data
actual_skeletons = np.zeros((amount_of_data, *shape))
broken_skeletons = np.zeros((amount_of_data, *shape))
predicted_gradients = np.zeros((amount_of_data, *shape))

dataset = ThickDataset_2(num_samples=1, patch_size=shape)

for i in range(amount_of_data):
    # Generate a new black input and actual gradient
    #actual_skeleton, broken_skeleton = generate_skeleton_based_data()#, num_lines=6, hole_length=20)
    sample = dataset[i]
    actual_skeleton = sample["image"].squeeze().numpy()
    broken_skeleton = sample["label"].squeeze().numpy()
    
    # Store the generated data
    actual_skeletons[i] = actual_skeleton
    broken_skeletons[i] = broken_skeleton

    # Apply the correct transform
    actual_skeleton_tensor = transform(actual_skeleton[np.newaxis,...]).unsqueeze(0).to(device)  # Move to device

    # Predict the gaps
    with torch.no_grad():
        predicted_gradient = model(actual_skeleton_tensor)

    # Store the predicted gradient
    predicted_gradients[i] = predicted_gradient.cpu().squeeze().numpy()

    #predicted_gradients[i] = remove_non_touching_components(predicted_gradients[i], actual_skeletons[i])

# Define save paths
save_path_full = os.path.join(plots_dir, "skeleton_data.npy")
save_path_broken = os.path.join(plots_dir, "broken_skeleton.npy")
save_path_predicted = os.path.join(plots_dir, "predicted_skeleton.npy")

# Save the 3D arrays to .npy files
np.save(save_path_full, actual_skeletons)
np.save(save_path_broken, broken_skeletons)
np.save(save_path_predicted, predicted_gradients)