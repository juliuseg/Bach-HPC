import Skeleton_model.No_Warn
import os
import sys
import pickle
import time
from multiprocessing import Pool

from Skeleton_model.Evaluation.configs import all_configs
from scipy.ndimage import label
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, label
from skimage.morphology import skeletonize


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


def get_voxel_counts(volume):
    # Label connected components
    structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity
    labeled, num_features = label(volume, structure=structure)

    # Count voxels per component, skip background (label 0)
    voxel_counts = np.bincount(labeled.ravel())[1:]
    return voxel_counts

def plot_voxel_distribution_from_counts(counts1, counts2, cutoff=500, title="Line Plot: Voxel Contribution by Component Size"):
    # Filter by cutoff
    counts1 = counts1[counts1 <= cutoff]
    counts2 = counts2[counts2 <= cutoff]

    hist1 = np.bincount(counts1, minlength=cutoff + 1)
    hist2 = np.bincount(counts2, minlength=cutoff + 1)

    voxel_sum1 = np.arange(cutoff + 1) * hist1
    voxel_sum2 = np.arange(cutoff + 1) * hist2

    total1 = voxel_sum1.sum()
    total2 = voxel_sum2.sum()

    percent1 = 100 * voxel_sum1[1:] / total1
    percent2 = 100 * voxel_sum2[1:] / total2

    print(f"Average component size (Data): {np.mean(counts1):.2f}")
    print(f"Average component size (Data+Pred): {np.mean(counts2):.2f}")

    x = np.arange(1, cutoff + 1)
    plt.plot(x, percent1, label="Data", linestyle='-', marker='')
    plt.plot(x, percent2, label="Data+Pred", linestyle='--', marker='')
    plt.xlabel("Component Size (Voxel Count)")
    plt.ylabel("Voxel Contribution (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_voxel_distribution_from_counts_hist(counts1, counts2, cutoff=500, bin_size=20, title="Histogram: Voxel Contribution by Component Size"):
    counts1 = counts1[counts1 <= cutoff]
    counts2 = counts2[counts2 <= cutoff]

    bins = np.arange(0, cutoff + bin_size, bin_size)

    voxel_contrib1, _ = np.histogram(counts1, bins=bins, weights=counts1)
    voxel_contrib2, _ = np.histogram(counts2, bins=bins, weights=counts2)

    percent1 = 100 * voxel_contrib1 / voxel_contrib1.sum()
    percent2 = 100 * voxel_contrib2 / voxel_contrib2.sum()

    print(f"Average component size (Data): {np.mean(counts1):.2f}")
    print(f"Average component size (Data+Pred): {np.mean(counts2):.2f}")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bin_size * 0.8
    plt.bar(bin_centers - width / 4, percent1, width=width / 2, label="Data", alpha=0.7)
    plt.bar(bin_centers + width / 4, percent2, width=width / 2, label="Data+Pred", alpha=0.7)
    plt.xlabel("Component Size (Voxel Count)")
    plt.ylabel("Voxel Contribution (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()




# ---------------------- Setup ----------------------
if len(sys.argv) < 1:
    raise ValueError("❗ Usage: python -m Skeleton_model.Evaluation.plot_hist_bef_aft <config_name> ")

config_name = sys.argv[1]
num_cores = os.cpu_count()

config = getattr(all_configs, config_name)

# Load inference results
inference_results_folder = "/work3/s204427/inference_results"
path_to_results = os.path.join(inference_results_folder, f"{config_name}_data.pkl")
if not os.path.exists(path_to_results):
    raise FileNotFoundError(f"❌ Inference results not found at: {path_to_results}")

with open(path_to_results, "rb") as f:
    results = pickle.load(f)

num_iterations = config["num_iterations"]

# Add this at the top to create a folder for saving plots
output_dir = f"./bef_aft_plots/{config_name}"
os.makedirs(output_dir, exist_ok=True)

# ---------------------- Worker Function ----------------------

def process_iteration(i):
    actual_skeleton = np.array(results["actual_skeletons"]).astype(np.float32)[i]
    predicted_hole = np.array(results["predicted_skeletons"]).astype(np.float32)[i]

    # Ensure binary
    actual_skeleton = (actual_skeleton > 0).astype(np.uint8)
    predicted_hole = (predicted_hole > 0).astype(np.uint8)

    # Clean
    cleaned_prediction = clean_prediction(actual_skeleton, predicted_hole)

    # Combine and skeletonize
    combined = np.maximum(actual_skeleton, cleaned_prediction)
    skeletonized_combined = skeletonize(combined, )

    # Get voxel counts from skeletonized combined volume
    counts_original = get_voxel_counts(actual_skeleton)
    counts_combined = get_voxel_counts(skeletonized_combined.astype(np.uint8))

    return counts_original, counts_combined


# ---------------------- Run Parallel ----------------------
print(f"🚀 Starting parallel gathering of voxel size distributions with {num_cores} cores...")

start_all = time.time()
with Pool(processes=num_cores) as pool:
    all_outputs = list(pool.map(process_iteration, range(num_iterations)))
print(f"✅ Done in {time.time() - start_all:.2f} seconds.")

# ---------------------- Aggregate All Counts ----------------------
all_actual_counts = []
all_cleaned_counts = []

for actual_counts, cleaned_counts in all_outputs:
    all_actual_counts.extend(actual_counts)
    all_cleaned_counts.extend(cleaned_counts)

all_actual_counts = np.array(all_actual_counts)
all_cleaned_counts = np.array(all_cleaned_counts)

# ---------------------- Plot & Save ----------------------
# Line plot
plt.figure()
plot_voxel_distribution_from_counts(
    all_actual_counts, all_cleaned_counts,
    cutoff=300,
    title="Line Plot: Voxel Contribution by Component Size"
)
plt.savefig(os.path.join(output_dir, "line_voxel_distribution.png"))
plt.close()

# Histogram
plt.figure()
plot_voxel_distribution_from_counts_hist(
    all_actual_counts, all_cleaned_counts,
    cutoff=300,
    bin_size=20,
    title="Histogram: Voxel Contribution by Component Size"
)
plt.savefig(os.path.join(output_dir, "hist_voxel_distribution.png"))
plt.close()