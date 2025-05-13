from Data_generation.datasets import generate_skeleton, generate_skeleton_with_thickness
import numpy as np
from multiprocessing import Pool, cpu_count
import os
from time import perf_counter as time
import sys

def generate_skeleton_wrapper(args):
    shape = args["shape"]
    size = args["size"]
    index = args["index"]
    gap_size = args["gap_threshold"]
    skeleton = args["skeleton"]
    gap_chance = args["gap_chance"]
    padding = args["padding"]
    num_lines = args["num_lines"]
    wobble = args["wobble"]

    # Call function with gap size if needed
    if skeleton:
        skeleton_data, broken_skeleton, long_holes = generate_skeleton(shape, gap_threshold=gap_size, gap_chance=gap_chance, num_lines=num_lines, wobble=wobble)
    else:
        skeleton_data, broken_skeleton, long_holes = generate_skeleton_with_thickness(shape, gap_threshold=gap_size, gap_chance=gap_chance, num_lines=num_lines, wobble=wobble)
    
    # remove padding
    skeleton_data = skeleton_data[padding:-padding, padding:-padding, padding:-padding]
    broken_skeleton = broken_skeleton[padding:-padding, padding:-padding, padding:-padding]
    long_holes = long_holes[padding:-padding, padding:-padding, padding:-padding]

    base_dir = "/work3/s204427"
    subfolder = "skeleton_data" if skeleton else "segmentation_data"
    gap_chance_str = str(args["gap_chance"]).replace(".", "")
    output_dir = os.path.join(base_dir, subfolder, f"s{size}_n{args['total']}_g{gap_size}_gs{gap_chance_str}_l{num_lines}_w{wobble}")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f"skeleton_data_{index}.npy"), skeleton_data)
    np.save(os.path.join(output_dir, f"broken_skeleton_{index}.npy"), broken_skeleton)
    np.save(os.path.join(output_dir, f"long_holes_{index}.npy"), long_holes)

    return index

if __name__ == "__main__":
    # === Handle command-line arguments ===
    default_size = 256
    default_jobs = cpu_count()
    default_threshold = 20
    default_skeleton = 1
    default_gap_chance = 0.3
    default_num_lines = 15

    # Parse command-line arguments
    size = int(sys.argv[1]) if len(sys.argv) > 1 else default_size
    num_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else default_jobs
    num_jobs = min(num_jobs, cpu_count())  # Limit to available CPU cores
    num_jobs = default_jobs if num_jobs == -1 else num_jobs
    gap_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else default_threshold
    skeleton = bool(int(sys.argv[4])) if len(sys.argv) > 4 else default_skeleton
    gap_chance = float(sys.argv[5]) if len(sys.argv) > 5 else default_gap_chance
    num_lines = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    wobble = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5

    padding = 32
    shape = (size + padding*2, size + padding*2, size + padding*2)

    print(f"Generating {'skeletons' if skeleton else 'segmentations'} with shape={shape}, "
          f"gap_size={gap_threshold}, using {num_jobs} parallel jobs and gap chance={gap_chance}")

    args_list = [{
        "shape": shape,
        "index": i,
        "gap_threshold": gap_threshold,
        "skeleton": skeleton,
        "total": num_jobs,
        "gap_chance": gap_chance,
        "padding": padding,
        "size": size,
        "num_lines": num_lines,
        "wobble": wobble,
    } for i in range(num_jobs)]

    t = time()
    with Pool(processes=num_jobs) as pool:
        results = pool.map(generate_skeleton_wrapper, args_list)
    elapsed = time() - t

    print(f"Finished generating all {num_jobs} datasets in {elapsed:.2f} seconds")
