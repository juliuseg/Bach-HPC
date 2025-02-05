import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.test_art_data import generate_skeleton_based_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the plots directory exists
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Define model checkpoint path
model_path = "model_checkpoints/model.pt"

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
shape_single_dim = 128
shape = (shape_single_dim,) * 3
amount_of_data = 10

# Initialize arrays to store 10 sets of data
actual_skeletons = np.zeros((amount_of_data, *shape))
broken_skeletons = np.zeros((amount_of_data, *shape))
predicted_gradients = np.zeros((amount_of_data, *shape))

for i in range(amount_of_data):
    # Generate a new black input and actual gradient
    actual_skeleton, broken_skeleton = generate_skeleton_based_data(shape=shape, num_lines=6, hole_length=20)
    
    # Store the generated data
    actual_skeletons[i] = actual_skeleton
    broken_skeletons[i] = broken_skeleton

    # Apply the correct transform
    broken_skeleton_tensor = transform(broken_skeleton).unsqueeze(0).to(device)  # Move to device

    # Predict the gradient
    with torch.no_grad():
        predicted_gradient = model(broken_skeleton_tensor)

    # Store the predicted gradient
    predicted_gradients[i] = predicted_gradient.cpu().squeeze().numpy()

# Define save paths
save_path_full = os.path.join(plots_dir, "skeleton_data.npy")
save_path_broken = os.path.join(plots_dir, "broken_skeleton.npy")
save_path_predicted = os.path.join(plots_dir, "predicted_skeleton.npy")

# Save the 3D arrays to .npy files
np.save(save_path_full, actual_skeletons)
np.save(save_path_broken, broken_skeletons)
np.save(save_path_predicted, predicted_gradients)