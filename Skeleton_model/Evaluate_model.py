import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from Skeleton_model.model import CustomUNet, transform
#from Skeleton_model.Data_Generation import generate_skeleton_based_data
from Skeleton_model.SkeletonDataset import SkeletonDataset
from Skeleton_model.Evaluate_utils import convert_prediction, get_skeleton_vectors, remove_non_touching_components, perm_test
from scipy.ndimage import label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the model from the saved .pt file
model_id = "dilation1_100epochs"

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


print ("Generating skeleton data")

# Define number of iterations
num_iterations = 100

# Dimensions of the 3D data
shape_single_dim = 64
shape = (shape_single_dim,) * 3

# Store p-values and label ratios
p_values_broken = []
p_values_predicted = []
label_ratios = []

skeleton_vectors = []
predicted_vectors = []

dataset = SkeletonDataset(num_samples=1, patch_size=shape)


# Run multiple iterations
for i in range(num_iterations):
    print(f"🔄 Iteration {i+1}/{num_iterations}")

    #print ("Generating skeleton data")
    # Get the skeleton data from dataset
    sample = dataset[i]
    actual_skeleton = sample["image"].squeeze().numpy()
    broken_skeleton = sample["label"].squeeze().numpy()

    # Apply the correct transform
    actual_skeleton_tensor = transform(actual_skeleton).unsqueeze(0).to(device)  # Move to device

    #print ("Applying model")
    # Predict the gaps
    with torch.no_grad():
        predicted_hole = model(actual_skeleton_tensor)

    # Convert prediction
    predicted_hole = predicted_hole.cpu().squeeze().numpy()
    predicted_hole = convert_prediction(predicted_hole)

    actual_skeleton = actual_skeleton[0]
    broken_skeleton = broken_skeleton[0]

    # Remove non-touching components
    predicted_hole = remove_non_touching_components(predicted_hole, actual_skeleton)

    # Label connected components
    #print ("Labeling connected components")
    connectivity = np.ones((3, 3, 3))
    actual_skeleton_labeled, actual_skeleton_num_labels = label(actual_skeleton, structure=connectivity)
    predicted_hole_labeled, predicted_hole_num_labels = label(actual_skeleton+predicted_hole, structure=connectivity)

    # Compute p-values
    #print("Computing p-values")
    _skeleton_vectors, _predicted_vectors  = get_skeleton_vectors(actual_skeleton, predicted_hole)

    skeleton_vectors = np.vstack((skeleton_vectors, _skeleton_vectors)) if len(skeleton_vectors) > 0 else _skeleton_vectors
    predicted_vectors = np.vstack((predicted_vectors, _predicted_vectors)) if len(predicted_vectors) > 0 else _predicted_vectors

    print(f"📌 Labels - Actual: {actual_skeleton_num_labels}, Predicted: {predicted_hole_num_labels}")

    label_ratios.append(actual_skeleton_num_labels / (predicted_hole_num_labels + 1e-6))  # Avoid division by zero

# Compute permutation tests
p_value = perm_test(skeleton_vectors, predicted_vectors, 1000, 1000, 10000)
print ("\nPerm test between original and predicted: ", (p_value))

# Print average label ratios
#print(f"📊 Average P-values - Broken: {np.mean(p_values_broken)}, Predicted: {np.mean(p_values_predicted)}"
print(f" Average Label Ratios: {np.mean(label_ratios)}")


# Define save directory
save_dir = "evaluation_plots"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save label ratios plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_iterations + 1), label_ratios, label="Actual/Predicted Label Ratio", marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Label Ratio (Actual / Predicted)")
plt.legend()
plt.title("Label Ratio over Iterations")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "label_ratios_over_iterations.png"), dpi=300)
plt.close()



# actual_skeleton, broken_skeleton = generate_skeleton_based_data(shape=shape, total_sum=1000, num_lines=50)#, num_lines=6, hole_length=20)

# print ("Applying model")
# # Apply the correct transform
# broken_skeleton_tensor = transform(broken_skeleton).unsqueeze(0).to(device)  # Move to device

# # Predict the gaps
# with torch.no_grad():
#     predicted_hole = model(broken_skeleton_tensor)

# # Store the predicted gradient
# predicted_hole = predicted_hole.cpu().squeeze().numpy()
# predicted_hole = convert_prediction(predicted_hole)

# actual_skeleton = actual_skeleton[0]
# broken_skeleton = broken_skeleton[0]

# # Get labels with ndimage.label
# actual_skeleton_labeled, actual_skeleton_num_labels = label(actual_skeleton)
# broken_skeleton_labeled, broken_skeleton_num_labels = label(broken_skeleton)
# predicted_hole_labeled, predicted_hole_num_labels = label(predicted_hole)

# # number of labels
# print ("Number of labels in actual skeleton: ", actual_skeleton_num_labels)
# print ("Number of labels in broken skeleton: ", broken_skeleton_num_labels)
# print ("Number of labels in predicted hole: ", predicted_hole_num_labels)

# # P values
# p_val_broken = get_p_value(actual_skeleton, broken_skeleton)
# p_val_pred = get_p_value(actual_skeleton, predicted_hole)
# print ("P value for broken: ", p_val_broken)
# print ("P value for predicted: ", p_val_pred)


# # Compute the directions
# print ("Getting the structure tensors:")
# print ("Actual Skeleton")
# original_directions = convert_skeleton_vectors( compute_skeleton_vectors(actual_skeleton_labeled, actual_skeleton_num_labels))
# print ("Broken Skeleton")
# hole_directions = convert_skeleton_vectors( compute_skeleton_vectors(broken_skeleton_labeled, broken_skeleton_num_labels))
# print ("Predicted Hole")
# predicted_directions = convert_skeleton_vectors( compute_skeleton_vectors(predicted_hole_labeled,predicted_hole_num_labels))

# # Get the p-values from the permutation tests
# print ("Computing the p-values:")
# p_value = perm_test(original_directions, hole_directions, 1000, 1000, 1000)
# print ("Perm test between original and hole: ", (p_value))
# p_value = perm_test(original_directions, predicted_directions, 1000, 1000, 1000)
# print ("Perm test between original and predicted: ", (p_value))
# p_value = perm_test(hole_directions, predicted_directions, 1000, 1000, 1000)
# print ("Perm test between hole and predicted: ", (p_value))
