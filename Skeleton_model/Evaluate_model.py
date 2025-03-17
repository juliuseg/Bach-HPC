import Skeleton_model.No_Warn
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from Skeleton_model.model import CustomUNet, transform
#from Skeleton_model.Data_Generation import generate_skeleton_based_data
# from Skeleton_model.SkeletonDataset import SkeletonDataset
from Skeleton_model.SkeletonDataset import SkeletonDataset
from Skeleton_model.Evaluate_utils import convert_prediction, get_skeleton_vectors, remove_non_touching_components, perm_test, model_for_iterations
from scipy.ndimage import label
from Skeleton_model.Baseline_model import SkeletonBaselineModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the model from the saved .pt file
model_id = "dilation0_new"

model_path = "model_checkpoints/model_"+str(model_id)+".pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model checkpoint not found at: {model_path}")

# Load the trained model
checkpoint = torch.load(model_path)
model = CustomUNet()
model.load_state_dict(checkpoint["model"])
model.to(device)  # Move to GPU/CPU
model.eval()
print(f"✅ Model loaded successfully from {model_path}")

# Baseline model
bs_model = SkeletonBaselineModel(search_radius=15)

print ("Generating skeleton data")

# Define number of iterations
num_iterations = 1

# Dimensions of the 3D data
shape_single_dim = 256
shape = (shape_single_dim,) * 3

# Store p-values and label ratios
p_values_broken = []
p_values_predicted = []
label_ratios = []
label_ratios_bs = []

skeleton_vectors = []
predicted_vectors = []
skeleton_vectors_bs = []
predicted_vectors_bs = []

dataset = SkeletonDataset(num_samples=num_iterations, patch_size=shape)


# Run multiple iterations
for i in range(num_iterations):
    print(f"🔄 Iteration {i+1}/{num_iterations}")

    #print ("Generating skeleton data")
    # Get skeleton data
    sample = dataset[i]
    actual_skeleton = sample["image"].numpy()
    broken_skeleton = sample["label"].numpy()

    # Apply the correct transform
    # actual_skeleton_tensor = transform(actual_skeleton).unsqueeze(0).to(device)  # Move to device

    # #print ("Applying model")
    # # Predict the gaps
    # with torch.no_grad():
    #     predicted_hole = model(actual_skeleton_tensor)

    # # Convert prediction
    # predicted_hole = predicted_hole.cpu().squeeze().numpy()
    # predicted_hole = convert_prediction(predicted_hole)

    predicted_hole = model_for_iterations(actual_skeleton.copy(), model, transform, device, iterations=5)
    predicted_hole_bs = model_for_iterations(actual_skeleton.copy(), bs_model, transform, device, iterations=1)
    print("prediction done")

    actual_skeleton = actual_skeleton[0]
    broken_skeleton = broken_skeleton[0]

    # Remove non-touching components
    predicted_hole = remove_non_touching_components(predicted_hole, actual_skeleton)

    # Save to /plots: actual_skeleton, predicted_hole as .npy files
    if i == num_iterations - 1:
        save_dir = "plots"
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Save the actual skeleton and predicted hole
        np.save(os.path.join(save_dir, f"actual_skeleton_{i}.npy"), actual_skeleton)
        np.save(os.path.join(save_dir, f"predicted_hole_{i}.npy"), predicted_hole)
        np.save(os.path.join(save_dir, f"predicted_hole_bs_{i}.npy"), predicted_hole_bs)
        np.save(os.path.join(save_dir, f"broken_skeleton_{i}.npy"), broken_skeleton)


    # Label connected components
    connectivity = np.ones((3, 3, 3))
    # Actual skeleton
    actual_skeleton_labeled, actual_skeleton_num_labels = label(actual_skeleton, structure=connectivity)
    # Predicted hole
    predicted_hole_labeled, predicted_hole_num_labels = label(actual_skeleton+predicted_hole, structure=connectivity)
    # Predicted hole baseline
    predicted_hole_bs_labeled, predicted_hole_bs_num_labels = label(actual_skeleton+predicted_hole_bs, structure=connectivity)


    # Get data for p-values for model
    _skeleton_vectors, _predicted_vectors  = get_skeleton_vectors(actual_skeleton, predicted_hole)

    skeleton_vectors = np.vstack((skeleton_vectors, _skeleton_vectors)) if len(skeleton_vectors) > 0 else _skeleton_vectors
    predicted_vectors = np.vstack((predicted_vectors, _predicted_vectors)) if len(predicted_vectors) > 0 else _predicted_vectors

    # Do the same for baseline
    _skeleton_vectors_bs, _predicted_vectors_bs  = get_skeleton_vectors(actual_skeleton, predicted_hole_bs)

    skeleton_vectors_bs = np.vstack((skeleton_vectors_bs, _skeleton_vectors_bs)) if len(skeleton_vectors_bs) > 0 else _skeleton_vectors_bs
    predicted_vectors_bs = np.vstack((predicted_vectors_bs, _predicted_vectors_bs)) if len(predicted_vectors_bs) > 0 else _predicted_vectors_bs

    # Print labels
    print(f"📌 Labels - Actual: {actual_skeleton_num_labels}, Predicted: {predicted_hole_num_labels}, Baseline: {predicted_hole_bs_num_labels}")

    label_ratios.append(actual_skeleton_num_labels / (predicted_hole_num_labels + 1e-6))  # Avoid division by zero
    label_ratios_bs.append(actual_skeleton_num_labels / (predicted_hole_bs_num_labels + 1e-6))  # Avoid division by zero

# Compute permutation tests
p_value = perm_test(skeleton_vectors, predicted_vectors, 1000, 1000, 10000)
print ("\nPerm test between original and predicted: ", (p_value))

p_value_bs = perm_test(skeleton_vectors_bs, predicted_vectors_bs, 1000, 1000, 10000)
print ("\nPerm test between original and predicted BS: ", (p_value_bs))

# Print average label ratios
#print(f"📊 Average P-values - Broken: {np.mean(p_values_broken)}, Predicted: {np.mean(p_values_predicted)}"
print(f" Average Label Ratios: {np.mean(label_ratios)}")
print(f" Average Label Ratios BS: {np.mean(label_ratios_bs)}")




# # Define save directory
# save_dir = "evaluation_plots"
# os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# # Save label ratios plot
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_iterations + 1), label_ratios, label="Actual/Predicted Label Ratio", marker='o', linestyle='-')
# plt.xlabel("Iteration")
# plt.ylabel("Label Ratio (Actual / Predicted)")
# plt.legend()
# plt.title("Label Ratio over Iterations")
# plt.grid(True)
# plt.savefig(os.path.join(save_dir, "label_ratios_over_iterations.png"), dpi=300)
# plt.close()