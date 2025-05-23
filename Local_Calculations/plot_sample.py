import numpy as np
from scipy.ndimage import convolve


import numpy as np
from scipy.ndimage import binary_dilation, label

def clean_prediction(skeleton, prediction):
    """
    Remove prediction components that don't connect two distinct skeleton segments.

    Parameters:
    - skeleton: binary 3D numpy array of original skeletons
    - prediction: binary 3D numpy array of predicted connections

    Returns:
    - cleaned_prediction: binary 3D numpy array containing only valid predictions
    """
    # Step 1: Label the original skeleton
    skeleton_labeled, _ = label(skeleton, structure=np.ones((3, 3, 3)))

    # Step 2: Dilate the prediction volume
    dilated_prediction = binary_dilation(prediction, structure=np.ones((3, 3, 3)))

    # Step 3: Label the connected components in the dilated prediction
    prediction_labeled, num_pred_labels = label(dilated_prediction, structure=np.ones((3, 3, 3)))

    # Step 4: Initialize empty volume for cleaned predictions
    cleaned_prediction = np.zeros_like(prediction, dtype=np.uint8)

    # Step 5: For each prediction label, check how many unique skeleton labels it touches
    for pred_label in range(1, num_pred_labels + 1):
        # Get the mask of this prediction component
        component_mask = (prediction_labeled == pred_label)

        # Get overlapping skeleton labels at the same voxel positions
        overlapping_skeleton_labels = skeleton_labeled[component_mask]

        # Count how many unique non-zero skeleton labels this prediction touches
        unique_labels = np.unique(overlapping_skeleton_labels)
        unique_labels = unique_labels[unique_labels != 0]

        if len(unique_labels) >= 2:
            # Valid connection, add it to cleaned prediction
            cleaned_prediction[component_mask] = 1

    return cleaned_prediction


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def get_voxel_counts(volume):
    # Label connected components
    structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity
    labeled, num_features = label(volume, structure=structure)

    # Count voxels per component, skip background (label 0)
    voxel_counts = np.bincount(labeled.ravel())[1:]
    return voxel_counts

def plot_voxel_distribution_by_size(volume1, volume2, cutoff=500, title="Voxel Distribution by Component Size"):
    counts1 = get_voxel_counts(volume1)
    counts2 = get_voxel_counts(volume2)

    # Filter by cutoff
    counts1 = counts1[counts1 <= cutoff]
    counts2 = counts2[counts2 <= cutoff]

    # Histogram: size -> count
    hist1 = np.bincount(counts1, minlength=cutoff + 1)
    hist2 = np.bincount(counts2, minlength=cutoff + 1)

    # Multiply size × frequency to get voxel totals per size
    voxel_sum1 = np.arange(cutoff + 1) * hist1
    voxel_sum2 = np.arange(cutoff + 1) * hist2

    # Normalize to percentage
    total1 = voxel_sum1.sum()
    total2 = voxel_sum2.sum()

    percent1 = 100 * voxel_sum1[1:] / total1  # skip 0
    percent2 = 100 * voxel_sum2[1:] / total2

    # Print average component size
    print(f"Average component size (Volume 1): {np.mean(counts1):.2f}")
    print(f"Average component size (Volume 2): {np.mean(counts2):.2f}")

    x = np.arange(1, cutoff + 1)

    # Plot
    plt.plot(x, percent1, label="Volume 1", linestyle='-', marker='')
    plt.plot(x, percent2, label="Volume 2", linestyle='--', marker='')

    plt.xlabel("Component Size (Voxel Count)")
    plt.ylabel("Voxel Contribution (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_voxel_distribution_by_size_hist(volume1, volume2, cutoff=500, bin_size=20, title="Voxel Contribution Histogram"):
    counts1 = get_voxel_counts(volume1)
    counts2 = get_voxel_counts(volume2)

    # Filter by cutoff
    counts1 = counts1[counts1 <= cutoff]
    counts2 = counts2[counts2 <= cutoff]

    # Define bin edges
    bins = np.arange(0, cutoff + bin_size, bin_size)

    # Compute voxel contribution per bin: sum(size * frequency) within each bin
    voxel_contrib1, _ = np.histogram(counts1, bins=bins, weights=counts1)
    voxel_contrib2, _ = np.histogram(counts2, bins=bins, weights=counts2)

    # Normalize to percentage
    percent1 = 100 * voxel_contrib1 / voxel_contrib1.sum()
    percent2 = 100 * voxel_contrib2 / voxel_contrib2.sum()

    # Print average component size
    print(f"Average component size (Volume 1): {np.mean(counts1):.2f}")
    print(f"Average component size (Volume 2): {np.mean(counts2):.2f}")

    # Compute bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot as bar charts
    width = bin_size * 0.8
    plt.bar(bin_centers - width / 4, percent1, width=width / 2, label="Volume 1", alpha=0.7)
    plt.bar(bin_centers + width / 4, percent2, width=width / 2, label="Volume 2", alpha=0.7)

    plt.xlabel("Component Size (Voxel Count)")
    plt.ylabel("Voxel Contribution (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def binary_dilation_no_endpoints(volume):
    """
    Binary dilation that avoids dilating endpoints (voxels with only one neighbor).

    Args:
        volume (np.ndarray): 3D binary input array.

    Returns:
        np.ndarray: Dilated volume where only non-endpoints are allowed to dilate.
    """
    # Define 3D connectivity (26 neighbors)
    struct = np.ones((3, 3, 3), dtype=int)
    struct[1, 1, 1] = 0  # exclude center

    # Count neighbors for each voxel
    neighbor_count = convolve(volume.astype(int), struct, mode='constant', cval=0)

    # Identify non-endpoints: voxel is 1 and has more than 1 neighbor
    non_endpoint_mask = (volume == 1) & (neighbor_count > 1)

    # Create a temporary volume that only includes non-endpoints
    non_endpoint_volume = np.zeros_like(volume)
    non_endpoint_volume[non_endpoint_mask] = 1

    # Perform standard binary dilation on non-endpoints
    dilated = convolve(non_endpoint_volume, struct, mode='constant', cval=0) > 0

    # Keep background unchanged (dilate only into previously 0 voxels)
    result = volume.astype(np.uint8) | dilated.astype(np.uint8)

    return result.astype(np.float32)

import napari
import numpy as np
import pickle
import os
import nibabel as nib
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize


config_name = "sample"

inference_results_folder = "/Volumes/s204427/project/path/Bach/inference_results"
path_to_results = os.path.join(inference_results_folder, f"{config_name}_data.pkl")
print ("Loading inference results from: ", path_to_results)
if not os.path.exists(path_to_results):
    raise FileNotFoundError(f"Inference results not found at: {path_to_results}")

# Load the pickle file
with open(path_to_results, "rb") as f:
    results = pickle.load(f)

# print the data types
print (f"Data types in results: {results['actual_skeletons'].dtype}, {results['predicted_skeletons'].dtype}")
# Extract the data from the results
actual_skeletons = np.array(results["actual_skeletons"]).astype(np.float32)
predicted_holes = np.array(results["predicted_skeletons"]).astype(np.float32)

# # make combined data by taking maximum of the two
# combined_data = np.maximum(actual_skeletons, predicted_holes)
# # Skeletonize the combined data
# combined_data = skeletonize(combined_data > 0.5).astype(np.float32)



# Example usage
clean_predictions = clean_prediction(actual_skeletons, predicted_holes)

combined_data = np.maximum(actual_skeletons, clean_predictions)
# Skeletonize the combined data
combined_data = skeletonize(combined_data > 0.5).astype(np.float32)


plot_voxel_distribution_by_size_hist(actual_skeletons, combined_data, cutoff=200)






# Nomrmalize the data
# actual_skeletons = actual_skeletons / np.max(actual_skeletons)
# predicted_holes = predicted_holes / np.max(predicted_holes)

# predicted_holes = skeletonize(predicted_holes > 0.5).astype(np.float32)
# predicted_holes = predicted_holes / np.max(predicted_holes)
# predicted_holes = np.clip(predicted_holes - binary_dilation_no_endpoints(actual_skeletons), 0, 1)

# print shapes
print(f"Actual skeletons shape: {actual_skeletons.shape}, uniques: {np.unique(actual_skeletons)}")
print(f"Predicted skeletons shape: {predicted_holes.shape}, uniques: {np.unique(predicted_holes)}")

# Crop the data for visualization
crop = (slice(120, 140), slice(0, 200), slice(0, 200))
# entire volume
crop = (slice(160,200), slice(None), slice(None))




# Separate binary masks
actual_mask = actual_skeletons > 0
predicted_mask = predicted_holes > 0

# Optionally dilate each mask
actual_mask = binary_dilation(actual_mask, structure=np.ones((3, 3, 3)))
predicted_mask = binary_dilation(predicted_mask, structure=np.ones((3, 3, 3)))

# Combine into label image, preserving label values
label_image = np.zeros_like(actual_skeletons, dtype=np.uint8)
label_image[predicted_mask] = 2  # Label 2 for predicted skeleton
label_image[actual_mask] = 1  # Label 1 for actual skeleton

# Create output folder
cwd = os.getcwd()
output_folder = os.path.join(cwd, f"{config_name}_nifti_output")
os.makedirs(output_folder, exist_ok=True)

# Save as a single label NIfTI file
label_img = nib.Nifti1Image(label_image[crop], affine=np.eye(4))
nib.save(label_img, os.path.join(output_folder, "skeletons_combined_labels.nii.gz"))

print(f"Saved combined label NIfTI to: {output_folder}")

output_folder = os.getcwd()  # or your desired output path
label_desc_path = os.path.join(output_folder, "skeletons_combined_labels.txt")

with open(label_desc_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(
        "# ITK-SNAP Label Description File\n"
        "# LabelIndex Red Green Blue Alpha LabelDescription\n"
        "1 255 255 0 255 ActualSkeleton\n"
        "2 255 0 0 255 PredictedSkeleton\n"
    )



# plot with napari
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(actual_skeletons[crop], name='Actual Skeletons', colormap='gray', interpolation='nearest', rendering='mip')
viewer.add_image(predicted_holes[crop], name='Predicted Skeletons', colormap='red', interpolation='nearest', rendering='mip')
viewer.add_image(combined_data[crop], name='Combined Skeletons', colormap='blue', interpolation='nearest', rendering='mip')

# make a only endpoints mask be taking only the points that are true in predicted_holes and actual_skeletons
endpoints_mask = np.logical_and(predicted_holes > 0, actual_skeletons > 0)
viewer.add_image(endpoints_mask[crop], name='Endpoints', colormap='green', interpolation='nearest', rendering='mip')

napari.run()