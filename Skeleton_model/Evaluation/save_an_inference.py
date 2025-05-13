import numpy as np
import sys
import os
import pickle


# Get config name from command line
if len(sys.argv) < 2:
    raise ValueError("❗ Usage: python save_an_inference.py <config_name>")
config_name = sys.argv[1]

inference_results_folder = "/work3/s204427/inference_results"
path_to_results = os.path.join(inference_results_folder, f"{config_name}_data.pkl")
if not os.path.exists(path_to_results):
    raise FileNotFoundError(f"❌ Inference results not found at: {path_to_results}")

with open(path_to_results, "rb") as f:
    results = pickle.load(f)

actual_skeleton = np.array(results["actual_skeletons"]).astype(np.float32)[0]
predicted_hole = np.array(results["predicted_skeletons"]).astype(np.float32)[0]

# Save the results as pickle files to the same file. Only the first iteration is saved.
folder_name = "inference_results"
os.makedirs(folder_name, exist_ok=True)

path = f"{folder_name}/sample_data.pkl"

with open(path, "wb") as f:
    pickle.dump({"actual_skeletons": actual_skeleton, "predicted_skeletons": predicted_hole}, f)

print(f"✅ Inference results saved successfully to: {path}")