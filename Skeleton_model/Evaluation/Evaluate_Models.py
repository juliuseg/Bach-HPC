import Skeleton_model.No_Warn
import os
import sys
import importlib
import numpy as np
import pickle
import time
from multiprocessing import Pool
from Skeleton_model.Evaluate_utils import (
    get_skeleton_vectors, remove_non_touching_components, perm_test, 
    CalculateTortuosity, CalculatePerforation, compute_kde_overlap, 
    perm_test_values, generate_cut_skeleton, print_labeled_histogram, 
    binary_dilation_no_endpoints,
)
from Skeleton_model.Evaluation.configs import all_configs
from scipy.ndimage import label
from skimage.morphology import skeletonize
from scipy.ndimage import binary_closing

def str2bool(v):
    return v.lower() in ("yes", "true", "True", "t", "1")


# ---------------------- Setup ----------------------
if len(sys.argv) < 2:
    raise ValueError("❗ Usage: python evaluate_model.py <config_name> [num_cores]")

config_name = sys.argv[1]
num_cores = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
full_prediction = str2bool(sys.argv[3]) if len(sys.argv) > 3 else False

print(f"\n🧠 Loading config: {config_name}, using {num_cores} cores")
# config_module = importlib.import_module(f"Skeleton_model.Evaluation.configs.{config_name}")
# config = config_module.config
config = getattr(all_configs, config_name)


# Load inference results
inference_results_folder = "/work3/s204427/inference_results"
path_to_results = os.path.join(inference_results_folder, f"{config_name}_data.pkl")
if not os.path.exists(path_to_results):
    raise FileNotFoundError(f"❌ Inference results not found at: {path_to_results}")

with open(path_to_results, "rb") as f:
    results = pickle.load(f)

num_iterations = config["num_iterations"]

# ---------------------- Worker Function ----------------------
def process_iteration(i):
    out = {}
    actual_skeleton = np.array(results["actual_skeletons"]).astype(np.float32)[i]
    predicted_hole = np.array(results["predicted_skeletons"]).astype(np.float32)[i]

    if full_prediction:
        predicted_hole = skeletonize(predicted_hole > 0.5).astype(np.float32)
        predicted_hole = predicted_hole / np.max(predicted_hole)
        predicted_hole = np.clip(predicted_hole - binary_dilation_no_endpoints(actual_skeleton), 0, 1)
        print (f"predicted_hole shape: {predicted_hole.shape}")

    
    predicted_hole, num_removed, num_features = remove_non_touching_components(predicted_hole, actual_skeleton)
    out["num_removed"] = num_removed
    out["num_features"] = num_features

    # Label and preprocess
    connectivity = np.ones((3, 3, 3))
    actual_labeled, actual_n = label(actual_skeleton, structure=connectivity)
    _, predicted_n = label(actual_skeleton + predicted_hole, structure=connectivity)

    if not config["skeleton"]:
        actual_labeled, _ = label(skeletonize(actual_skeleton), structure=connectivity)
        predicted_skeleton = skeletonize(predicted_hole + actual_skeleton)
        predicted_skeleton = predicted_skeleton * (predicted_hole > 0)
        predicted_labeled, _ = label(predicted_skeleton, structure=connectivity)
    else:
        predicted_labeled, _ = label(predicted_hole, structure=connectivity)

    actual_labeled_for_tortuosity, predicted_labeled_for_tortuosity = generate_cut_skeleton(predicted_labeled, actual_labeled)

    # print_labeled_histogram(actual_labeled_for_tortuosity, predicted_labeled_for_tortuosity)
    actual_tort = CalculateTortuosity(actual_labeled_for_tortuosity, use_tqdm=False, num_labels=int(np.ceil(1000 / num_iterations)))
    predicted_tort = CalculateTortuosity(predicted_labeled_for_tortuosity, use_tqdm=False, num_labels=int(np.ceil(1000 / num_iterations)))
    out["actual_tort"] = actual_tort
    out["predicted_tort"] = predicted_tort

    actual_labeled, num_labels = label(actual_skeleton, structure=np.ones((3, 3, 3)))
    counts = np.bincount(actual_labeled.ravel())
    valid_labels = np.where(counts > 10)[0]
    valid_labels = valid_labels[valid_labels != 0]
    filtered_skeleton = np.isin(actual_labeled, valid_labels).astype(actual_skeleton.dtype)

    sv, pv = get_skeleton_vectors(actual_skeleton, predicted_hole + actual_skeleton, predicted_hole)
    out["sv"] = sv
    out["pv"] = pv
    out["actual_n"] = actual_n
    out["predicted_n"] = predicted_n
    out["label_ratio"] = np.max([predicted_n, 1e-6]) / actual_n
    out["extension"] = (num_features - num_removed) - (actual_n - predicted_n)

    if not config["skeleton"]:
        out["porosity"] = CalculatePerforation(predicted_hole)
    else:
        out["porosity"] = -1

    # Save data for visualization on last iteration
    if i == num_iterations - 1:
        os.makedirs("plots", exist_ok=True)
        np.save(f"plots/{config_name}_actual_skeleton.npy", actual_skeleton)
        np.save(f"plots/{config_name}_predicted_hole.npy", predicted_hole)
        print("📊 Saved last iteration for visualization")

    return out

# ---------------------- Run Parallel ----------------------
print(f"🚀 Starting parallel evaluation with {num_cores} cores...")

start_all = time.time()
with Pool(processes=num_cores) as pool:
    all_outputs = list(pool.map(process_iteration, range(num_iterations)))
print(f"✅ Done in {time.time() - start_all:.2f} seconds.")

# ---------------------- Collate Results ----------------------
porosity = []
num_removed_total = []
num_features_total = []
torturosity_actual = []
torturosity_predicted = []
skeleton_vectors = []
predicted_vectors = []
labels_in_actual = []
labels_in_predicted = []
label_ratios = []
num_extensions = []

for out in all_outputs:
    porosity.append(out["porosity"])
    num_removed_total.append(out["num_removed"])
    num_features_total.append(out["num_features"])
    labels_in_actual.append(out["actual_n"])
    labels_in_predicted.append(out["predicted_n"])
    label_ratios.append(out["label_ratio"])
    num_extensions.append(out["extension"])
    torturosity_actual = np.hstack((torturosity_actual, out["actual_tort"])) if len(torturosity_actual) > 0 else out["actual_tort"]
    torturosity_predicted = np.hstack((torturosity_predicted, out["predicted_tort"])) if len(torturosity_predicted) > 0 else out["predicted_tort"]
    skeleton_vectors = np.vstack((skeleton_vectors, out["sv"])) if len(skeleton_vectors) > 0 else out["sv"]
    predicted_vectors = np.vstack((predicted_vectors, out["pv"])) if len(predicted_vectors) > 0 else out["pv"]

# ---------------------- Analysis ----------------------
kde_overlap = compute_kde_overlap(skeleton_vectors, predicted_vectors)
p_value_az, p_value_al, obs_diff = perm_test(skeleton_vectors, predicted_vectors, 1000, 1000, 10000)
p_value_tort = perm_test_values(torturosity_actual, torturosity_predicted)

results = {
    "p_value_az": p_value_az,
    "p_value_al": p_value_al,
    "obs_diff": obs_diff,
    "kde_overlap": kde_overlap,
    "label_ratios": label_ratios,
    "num_removed_total": num_removed_total,
    "num_features_total": num_features_total,
    "num_extensions": num_extensions,
    "labels_in_actual": labels_in_actual,
    "labels_in_predicted": labels_in_predicted,
    "porosity": porosity,
    "tortuosity_actual": torturosity_actual,
    "tortuosity_predicted": torturosity_predicted,
    "p_value_tort": p_value_tort,
    "skeleton_vectors": skeleton_vectors,
    "predicted_vectors": predicted_vectors,
    # "config": config,
}

# ---------------------- Save & Print ----------------------
print("\n📊 Final Results:")
print(f"🧪 p-value azmimuth: {p_value_az}, p-value altitude {p_value_al}")
print(f"🧪 Observed Difference: {obs_diff}")
print(f"📊 KDE Overlap: {kde_overlap}")
print(f"📊 Avg Label Ratio: {np.mean(label_ratios)}, Std: {np.std(label_ratios)}")
print(f"📏 porosity: {porosity}")
print(f"📏 Tortuosity - Actual: {np.mean(torturosity_actual)}, std {np.std(torturosity_actual)}")
print(f"📏 Tortuosity - Predicted: {np.mean(torturosity_predicted)}, std {np.std(torturosity_predicted)}")
print(f"🧪 Permutation Test p-value Tortuosity: {p_value_tort}")

folder_name = "saved_runs"
os.makedirs(folder_name, exist_ok=True)
with open(f"{folder_name}/{config_name}_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"✅ Results saved to {folder_name}/{config_name}_results.pkl")
