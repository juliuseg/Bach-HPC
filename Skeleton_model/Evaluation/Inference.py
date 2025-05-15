import Skeleton_model.No_Warn
import os
import sys
import torch
import pickle
import time
import gc

from Skeleton_model.Evaluation.Load_Test_Strucutere import NarwhalDataset
from Skeleton_model.Evaluate_utils import load_model_weights
from Skeleton_model.get_predictions_from_models import model_for_iterations
from Skeleton_model.Baseline_model import SkeletonBaselineModel
from Skeleton_model.Evaluation.configs import all_configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\ndevice:", device)

if len(sys.argv) < 2:
    raise ValueError("❗ Usage: python evaluate_model.py <config_name | all | comma,separated,list>")

input_arg = sys.argv[1]

# Prepare list of config names
if input_arg == "all":
    config_names = [name for name in dir(all_configs) if not name.startswith("__") and isinstance(getattr(all_configs, name), dict)]
else:
    config_names = [name.strip() for name in input_arg.split(",")]

# === Determine dataset loading strategy ===
use_shared_dataset = len(config_names) > 1
dataset_skel = None
dataset_noskel = None

if use_shared_dataset:
    print("📦 Loading shared datasets for all configs...")
    skeleton_configs = [name for name in config_names if getattr(all_configs, name)["skeleton"]]
    nonskeleton_configs = [name for name in config_names if not getattr(all_configs, name)["skeleton"]]

    if skeleton_configs and nonskeleton_configs:
        print("Shared datasets loading for both skeleton=True and skeleton=False")
        dataset_skel = NarwhalDataset(num_samples=32, patch_size=(256, 256, 256), skeleton=True)
        dataset_noskel = NarwhalDataset(num_samples=32, patch_size=(256, 256, 256), skeleton=False)
    elif skeleton_configs:
        print("Shared dataset loaded, only skeleton=True")
        dataset_skel = NarwhalDataset(num_samples=32, patch_size=(256, 256, 256), skeleton=True)
    elif nonskeleton_configs:
        print("Shared dataset loaded, only skeleton=False")
        dataset_noskel = NarwhalDataset(num_samples=32, patch_size=(256, 256, 256), skeleton=False)
    else:
        raise ValueError("❌ No valid configs found for dataset loading.")
    print("✅ Shared datasets loaded (32 samples, 256³, both skeleton=True/False)")

def run_config(config_name):
    print(f"\n🚀 Running config: {config_name}")
    config = getattr(all_configs, config_name, None)

    if config is None:
        print(f"❌ Config '{config_name}' not found, skipping.")
        return

    if config["model_class"] == SkeletonBaselineModel:
        print(f"Using baseline model with search radius: {config['search_radius']}")
        model = SkeletonBaselineModel(search_radius=config["search_radius"])
    else:
        model_path = config["model_path"]
        print(f"📥 Loading model from {model_path}")
        if not os.path.exists(model_path):
            print(f"❌ Skipping {config_name}: model not found at {model_path}")
            return

        model = config["model_class"]()
        model = load_model_weights(model, model_path, device)
        print("✅ Model loaded")

    # === Select appropriate dataset ===
    if use_shared_dataset:
        dataset = dataset_skel if config["skeleton"] else dataset_noskel
        num_iterations = 32
    else:
        dataset = NarwhalDataset(
            num_samples=config["num_iterations"],
            patch_size=config["patch_size"],
            skeleton=config["skeleton"]
        )
        num_iterations = config["num_iterations"]

    actual_skeletons = []
    predicted_skeletons = []

    start_time = time.time()
    for i in range(num_iterations):
        sample = dataset[i]
        actual_skeleton = sample["image"].numpy()

        with torch.no_grad():
            predicted_hole = model_for_iterations(
                actual_skeleton,
                model,
                config["skeleton"],
                config["transform"],
                device,
                iterations=config["predict_iterations"]
            )

        actual_skeletons.append(actual_skeleton)
        predicted_skeletons.append(predicted_hole)

        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()

    print(f"⏱ Finished {num_iterations} iterations in {time.time() - start_time:.2f} seconds")

    results = {
        "actual_skeletons": actual_skeletons,
        "predicted_skeletons": predicted_skeletons,
    }

    folder_name = "/work3/s204427/inference_results"
    os.makedirs(folder_name, exist_ok=True)

    output_path = f"{folder_name}/{config_name}_data.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"✅ Results saved to {output_path}")

# === Run all selected configs ===
for name in config_names:
    run_config(name)
