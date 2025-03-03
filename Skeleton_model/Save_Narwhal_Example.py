import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Data_Generation import generate_skeleton_based_data

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

narwhal_data = np.load(os.path.join(plots_dir, "narwhal_test_data.npy"))
narwhal_data = np.expand_dims(narwhal_data, axis=0)  # Adds a new first dimension
print("narwhal_data shape:", narwhal_data.shape)
# print min and max values
print("min:", np.min(narwhal_data))
print("max:", np.max(narwhal_data))


# Predict the gaps
with torch.no_grad():
    narwhal_data_tensor = transform(narwhal_data).unsqueeze(0).to(device)  # Move to device
    predicted_gap = model(narwhal_data_tensor)
    predicted_gap = predicted_gap.cpu().squeeze().numpy()

save_path_predicted = os.path.join(plots_dir, "narwhal_predicted_skeleton.npy")

np.save(save_path_predicted, predicted_gap)

print(f"✅ Predicted gap saved successfully to {save_path_predicted}")


