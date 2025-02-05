
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

# Generate a new black input and actual gradient
actual_skeleton, broken_skeleton = generate_skeleton_based_data()

# save broken_skeleton and actual_skeleton
# Define save path
save_path_full = os.path.join(plots_dir, "skeleton_data.npy")
save_path_broken = os.path.join(plots_dir, "broken_skeleton.npy")
save_path_predicted = os.path.join(plots_dir, "predicted_skeleton.npy")

# Save the 3D array to a .npy file
np.save(save_path_full, actual_skeleton)
np.save(save_path_broken, broken_skeleton)

# Apply the correct transform
broken_skeleton_tensor = transform(broken_skeleton).unsqueeze(0).to(device)  # Move to device

# Predict the gradient
with torch.no_grad():
    predicted_gradient = model(broken_skeleton_tensor)

np.save(save_path_predicted, predicted_gradient.cpu().squeeze().numpy())

# # Move data back to CPU for visualization
# predicted_gradient = predicted_gradient.cpu().squeeze().numpy()
# broken_skeleton = broken_skeleton.squeeze()
# actual_skeleton = actual_skeleton.squeeze()

# # Middle slice for visualization
# mid_slice = broken_skeleton.shape[1] // 2  # Assume shape (D, H, W), take middle depth slice

# # Plot input vs. actual vs. predicted
# plt.figure(figsize=(15, 5))

# # Input image
# plt.subplot(1, 3, 1)
# plt.imshow(broken_skeleton[mid_slice], cmap="gray", vmin=0, vmax=1)  # Normalize to [0, 1]
# plt.title("Input (Black Image)")
# plt.colorbar()

# # Actual gradient
# plt.subplot(1, 3, 2)
# plt.imshow(actual_skeleton[mid_slice], cmap="gray", vmin=0, vmax=1)  # Normalize to [0, 1]
# plt.title("Actual Gradient")
# plt.colorbar()

# # Predicted gradient
# plt.subplot(1, 3, 3)
# plt.imshow(predicted_gradient[mid_slice], cmap="gray", vmin=0, vmax=1)  # Normalize to [0, 1]
# plt.title("Predicted Gradient")
# plt.colorbar()

# # Save the plot
# plot_path = os.path.join(plots_dir, "debug_plot.png")
# plt.savefig(plot_path)
# plt.close()

# print(f"✅ Debug plot saved at: {plot_path}")