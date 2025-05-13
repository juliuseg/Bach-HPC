import numpy as np
from Skeleton_model.st3d import structure_tensor, eig_special
from scipy.ndimage import label, binary_dilation, binary_erosion
from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Baseline_model import SkeletonBaselineModel
import torch
import time
from tqdm import tqdm
from scipy.ndimage import binary_closing, generate_binary_structure
from collections import Counter

import pandas as pd

# def mean_direction(vecs):
#     """Compute mean direction of a set of unit vectors."""
#     mean_vec = np.mean(vecs, axis=0)
#     return mean_vec / np.linalg.norm(mean_vec)  # Normalize to get a unit vector

def print_align_vectors_to_mean(vecs, mean, max_vectors=1000, seed=42):
    """
    Flip vectors so they point in same hemisphere as the mean axis.
    Returns a string: 'v1 = np.array([[x, y, z], ...])'
    """
    vecs = np.array(vecs)

    # Random subset if too many
    if len(vecs) > max_vectors:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(vecs), size=max_vectors, replace=False)
        vecs = vecs[indices]

    aligned = align_vectors_to_mean(vecs, mean)

    # Format as one-line numpy array string
    vector_str = ", ".join([f"[{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]" for v in aligned])
    return f"np.array([{vector_str}])"


def align_vectors_to_mean(vectors, mean):
    """
    Align vectors to the same hemisphere as the mean vector.
    """
    vectors = np.array(vectors)
    mean = mean / np.linalg.norm(mean)

    dots = np.dot(vectors, mean)  # shape (N,)
    signs = np.where(dots < 0, -1, 1).reshape(-1, 1)

    aligned = vectors * signs
    aligned = aligned / np.linalg.norm(aligned, axis=1, keepdims=True)
    return aligned


def mean_axis(vecs):
    """
    Computes the mean axis from a set of 3D unit vectors, treating (v) and (-v) as the same direction.
    This is done using the principal eigenvector of the orientation matrix.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    vecs = vecs / norms
    # Compute orientation matrix
    orientation_matrix = np.einsum('ni,nj->ij', vecs, vecs)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(orientation_matrix)

    # Return the eigenvector with the largest eigenvalue
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    return principal_axis

def angular_distance(vec1, vec2):
    """Compute the angular distance between two unit vectors."""
    dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)  # Clip to avoid numerical issues
    if dot_product < 0:
        dot_product = -dot_product
    return np.arccos(dot_product)  # Returns angle in radians

def perm_test(vecs_orig, vecs_hole, n_orig=None, n_hole=None, n_permutations=10000):
    """
    Perform a permutation test on two sets of 3D unit vectors by comparing
    the angular distance between their mean directions.
    """
    mean_orig = mean_axis(vecs_orig)
    mean_pred = mean_axis(vecs_hole)

    # difference between the two mean directions
    obs_diff = angular_distance(mean_orig, mean_pred)
    # Obs diff to degrees
    obs_diff = np.rad2deg(obs_diff)

    # print (f"vecs_orig = {print_align_vectors_to_mean(vecs_orig,mean_orig)}")
    # print (f"vecs_hole = {print_align_vectors_to_mean(vecs_hole,mean_orig)}")
    
    vecs_orig = align_vectors_to_mean(vecs_orig, mean_orig)
    vecs_hole = align_vectors_to_mean(vecs_hole, mean_orig)

    az_A, alt_A = cartesian_to_spherical(vecs_orig)
    az_B, alt_B = cartesian_to_spherical(vecs_hole)

    # print (f"az_A[0] = {az_A[:10]}")
    # print (f"az_B[0] = {az_B[:10]}")
    # print (f"alt_A[0] = {alt_A[:10]}")
    # print (f"alt_B[0] = {alt_B[:10]}")


    # print ("Doing permutation test for azimuth")
    p_az = perm_test_angles(az_A, az_B)
    # print ("Doing permutation test for altitude")
    p_alt = perm_test_angles(alt_A, alt_B)


    return p_az, p_alt, obs_diff

    # if n_orig is None:
    #     n_orig = len(vecs_orig)
    # if n_hole is None:
    #     n_hole = len(vecs_hole)

    # # Compute observed test statistic: Angular distance between mean directions
    # mean_orig = mean_axis(vecs_orig)
    # mean_hole = mean_axis(vecs_hole)


    # print (f"mean_orig: {mean_orig}")
    # print (f"mean_hole: {mean_hole}")

    # obs_diff = angular_distance(mean_orig, mean_hole)

    # # Permutation test setup
    # pooled = np.vstack([vecs_orig, vecs_hole])
    # n_total = pooled.shape[0]
    # null_distribution = np.zeros(n_permutations)

    # # Permutation loop
    # for i in range(n_permutations):
    #     perm_indices = np.random.choice(n_total, n_orig + n_hole, replace=False)  # Random indices
    #     perm_group1 = pooled[perm_indices[:n_orig]]
    #     perm_group2 = pooled[perm_indices[n_orig:]]
        
    #     # Compute test statistic for permuted data
    #     null_distribution[i] = angular_distance(mean_axis(perm_group1), mean_axis(perm_group2))
    #     #print (f"null_distribution[{i}]: {null_distribution[i]}")

    # print (f"obs_diff: {obs_diff}")
    # # Compute p-value
    # p_value = np.mean(null_distribution >= obs_diff)
    
    # # Print how many permutations were greater than the observed difference
    # print(f"🧪 Permutation Test: {np.sum(null_distribution >= obs_diff)} / {n_permutations} permutations")

    # return p_value

def cartesian_to_spherical(vectors):
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    azimuth = np.arctan2(y, x)
    azimuth = np.mod(azimuth, 2*np.pi)

    altitude = np.arcsin(z)
    altitude = np.mod(altitude, 2*np.pi)

    return azimuth, altitude

def perm_test_angles(angles_A, angles_B, n_permutations=10000):
    """
    Permutation test for angular data. Compares the angular distance between
    the circular means of two sets of angles (in radians).
    """
    n_A = len(angles_A)
    n_B = len(angles_B)

    # Compute circular means
    mean_A = np.angle(np.mean(np.exp(1j * angles_A))) % (2*np.pi)
    mean_B = np.angle(np.mean(np.exp(1j * angles_B))) % (2*np.pi)

    # Observed angular difference
    obs_diff = np.abs((mean_A - mean_B + np.pi) % (2*np.pi) - np.pi)

    # Permutation test
    pooled = np.concatenate([angles_A, angles_B])
    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        permuted = np.random.permutation(pooled)
        perm_A = permuted[:n_A]
        perm_B = permuted[n_A:]

        mean_perm_A = np.angle(np.mean(np.exp(1j * perm_A))) % (2 * np.pi)
        mean_perm_B = np.angle(np.mean(np.exp(1j * perm_B))) % (2 * np.pi)

        diff = np.abs((mean_perm_A - mean_perm_B + np.pi) % (2*np.pi) - np.pi)
        null_distribution[i] = diff
        #print (f"Permutation: {i}, d:{np.rad2deg(diff)} >o:{np.rad2deg(obs_diff)} ? { diff > obs_diff}")

    p_value = np.mean(null_distribution >= obs_diff)

    # print(f"🧪 Permutation Test (angles): {np.sum(null_distribution >= obs_diff)} / {n_permutations}")
    # print(f"Observed angular difference: {np.rad2deg(obs_diff):.2f}°")
    # print(f"P-value: {p_value:.4f}")

    return p_value

# Correct von Mises-Fisher PDF (3D sphere)
def vmf_pdf(x, mu, kappa):
    c_d = kappa / (4 * np.pi * np.sinh(kappa))
    return c_d * np.exp(kappa * (x @ mu))

# Spherical KDE using vMF kernel
def spherical_kde(data, grid_points, kappa):
    kde_vals = np.zeros(len(grid_points))
    for d in data:
        kde_vals += vmf_pdf(grid_points, d, kappa)
    kde_vals /= len(data)
    return kde_vals

# Generate an even spherical grid
def spherical_grid(n_theta=200, n_phi=100):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi)

    x = np.sin(Phi) * np.cos(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Phi)

    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    return points, Theta, Phi

# Compute overlap between KDEs (proper spherical integration)
def kde_overlap(kde1, kde2, Theta, Phi):
    min_density = np.minimum(kde1, kde2)
    
    # Proper spherical differential element
    d_theta = Theta[0, 1] - Theta[0, 0]
    d_phi = Phi[1, 0] - Phi[0, 0]
    
    area_element = np.sin(Phi).ravel() * d_theta * d_phi
    overlap = np.sum(min_density * area_element)
    
    return overlap

def compute_kde_overlap(A, B, kappa=30):
    mean_orig = mean_axis(A)

    A = align_vectors_to_mean(A, mean_orig)
    B = align_vectors_to_mean(B, mean_orig)

    # Generate spherical grid
    points, Theta, Phi = spherical_grid(n_theta=200, n_phi=100)

    # KDE bandwidth parameter (higher kappa = narrower kernel)

    # Compute KDE on grid
    kde_A = spherical_kde(A, points, kappa)
    kde_B = spherical_kde(B, points, kappa)


    # Compute overlap correctly
    overlap = kde_overlap(kde_A, kde_B, Theta, Phi)
    return overlap


import numpy as np

def perm_test_values(values_A, values_B, n_permutations=10000, two_sided=True):
    """
    Permutation test for regular numerical data.
    Compares the difference in means between two groups.
    
    Args:
        values_A (array-like): First sample.
        values_B (array-like): Second sample.
        n_permutations (int): Number of permutations.
        two_sided (bool): If True, tests for difference in either direction.
    
    Returns:
        float: p-value.
    """
    # Convert to numpy arrays and flatten to 1D
    values_A = np.asarray(values_A).ravel()
    values_B = np.asarray(values_B).ravel()
    
    n_A = len(values_A)
    n_B = len(values_B)

    # Observed difference in means
    obs_diff = np.mean(values_A) - np.mean(values_B)
    
    # Permutation test
    pooled = np.concatenate([values_A, values_B])
    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        permuted = np.random.permutation(pooled)
        perm_A = permuted[:n_A]
        perm_B = permuted[n_A:]
        diff = np.mean(perm_A) - np.mean(perm_B)
        null_distribution[i] = diff

    # Compute p-value
    if two_sided:
        p_value = np.mean(np.abs(null_distribution) >= np.abs(obs_diff))
    else:
        p_value = np.mean(null_distribution >= obs_diff)

    return p_value


def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
def convert_skeleton_vectors(skeleton_vectors):
    """
    Converts the dictionary of labeled skeleton vectors into a single (x,3) array of 3D vectors.

    :param skeleton_vectors: Dictionary mapping each label to an array of (position, vector).
    :return: NumPy array of shape (x, 3) containing only the 3D vectors.
    """
    all_vectors = []

    for vectors in skeleton_vectors.values():
        all_vectors.append(vectors[:, 3:])  # Extract only the vector part

    return np.vstack(all_vectors) if all_vectors else np.empty((0, 3), dtype=np.float64)

def convert_prediction(predicted):
    """
    Convert the predicted hole to a binary image.
    
    :param predicted: 3D array of predicted hole values
    :return: Binary 3D array of the predicted hole
    """
    return (predicted > 0.5).astype(np.float64)

import numpy as np
import time
from joblib import Parallel, delayed

from scipy.ndimage import label

def get_skeleton_vectors(skeleton_data, predicted_skeleton, only_prediction, sigma=2, rho=5, n_jobs=1):
    start_time = time.time()

    # --- Compute structure tensors in parallel ---
    st_skeleton, st_predicted = Parallel(n_jobs=n_jobs)(
        delayed(structure_tensor)(vol, sigma, rho, n_jobs)
        for vol in [skeleton_data, predicted_skeleton]
    )
    # print(f"⏱️ Time to compute structure tensors: {time.time() - start_time:.2f}s")

    start_time = time.time()
    _, vec_skeleton = eig_special(st_skeleton, False)
    _, vec_predicted = eig_special(st_predicted, False)
    # print(f"⏱️ Time to compute eigenvectors: {time.time() - start_time:.2f}s")

    # --- Reshape to (X, Y, Z, 3) ---
    start_time = time.time()
    shape = skeleton_data.shape
    vec_skeleton = vec_skeleton.reshape(3, *shape).transpose(1, 2, 3, 0)
    vec_predicted = vec_predicted.reshape(3, *shape).transpose(1, 2, 3, 0)

    # --- Label connected components ---
    labeled_skeleton, n_skel = label(skeleton_data)
    labeled_predicted, n_pred = label(only_prediction)

    # --- Compute mean axis per label efficiently ---
    skeleton_results = compute_label_directions(vec_skeleton, labeled_skeleton, n_skel)
    predicted_results = compute_label_directions(vec_predicted, labeled_predicted, n_pred)

    # print(f"⏱️ Time to label + compute directions: {time.time() - start_time:.2f}s")

    return np.array(skeleton_results), np.array(predicted_results)


def compute_label_directions(vec_field, labeled_volume, num_labels):
    # Get all non-zero voxel coordinates and corresponding labels
    coords = np.argwhere(labeled_volume > 0)
    labels = labeled_volume[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Get all vectors at those coords
    vectors = vec_field[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Prepare results list
    results = []

    # Group vectors by label using unique and mask
    unique_labels = np.arange(1, num_labels + 1)
    for label_id in unique_labels:
        mask = labels == label_id
        if np.sum(mask) < 2:  # Skip small blobs
            continue  
        label_vectors = vectors[mask]
        direction = mean_axis(label_vectors)
        results.append([np.sum(mask), *direction])

    results = np.array(results)
    results = results[:, 1:]
    return results



import numpy as np
from scipy.ndimage import label, binary_dilation

def remove_non_touching_components(predicted_hole, actual_skeleton):
    """
    Removes labeled groups in predicted_hole that do not touch actual_skeleton.
    
    Parameters:
    - predicted_hole (numpy.ndarray): 3D binary array (float values, will be converted to int).
    - actual_skeleton (numpy.ndarray): 3D binary array indicating the reference structure.
    
    Returns:
    - filtered_predicted_hole (numpy.ndarray): 3D array with non-touching components removed.
    """

    # Convert float binary arrays to integer binary
    predicted_hole = (predicted_hole > 0).astype(int)
    actual_skeleton = (actual_skeleton > 0).astype(int)

    # Label connected components in predicted_hole using full 3D connectivity
    connectivity = np.ones((3, 3, 3), dtype=int)
    labeled_pred, num_features = label(predicted_hole, structure=connectivity)


    # Expand actual_skeleton slightly to ensure touching components are detected
    dilated_skeleton = binary_dilation(actual_skeleton, structure=connectivity)

    # Find unique labels that touch actual_skeleton (or its dilated version)
    touching_labels = np.unique(labeled_pred[dilated_skeleton > 0])

    # Remove non-touching labels
    mask = np.isin(labeled_pred, touching_labels)
    filtered_predicted_hole = np.where(mask, predicted_hole, 0)

    # Count removed components
    num_removed = num_features - len(touching_labels)

    return filtered_predicted_hole, num_removed, num_features

from scipy.ndimage import find_objects
import numpy as np
from scipy.ndimage import label, binary_dilation

def count_hole_components_touching_one_skeleton(predicted_hole, actual_skeleton):
    """
    Efficiently counts how many predicted_hole components touch exactly one actual_skeleton component.

    Parameters:
    - predicted_hole (ndarray): 3D binary array.
    - actual_skeleton (ndarray): 3D binary array.

    Returns:
    - count_touching_one (int): Number of predicted_hole components touching exactly one skeleton.
    - total_predicted_components (int): Total number of predicted_hole components.
    """
    predicted_hole = (predicted_hole > 0).astype(np.uint8)
    actual_skeleton = (actual_skeleton > 0).astype(np.uint8)

    connectivity = np.ones((3, 3, 3), dtype=np.uint8)

    # Label the predicted components and skeleton
    labeled_pred, num_pred = label(predicted_hole, structure=connectivity)
    labeled_skel, _ = label(actual_skeleton, structure=connectivity)

    # Dilate the entire labeled prediction mask
    # This keeps labels intact (won't merge them)
    dilated = binary_dilation(labeled_pred > 0, structure=connectivity)
    
    # We'll use a mask to keep only regions around the predicted components
    touching_zone = np.zeros_like(labeled_pred)
    touching_zone[dilated] = labeled_pred[dilated]

    count_touching_one = 0

    for label_id in range(1, num_pred + 1):
        # Find where this predicted component's dilated region is
        mask = (touching_zone == label_id)

        # Find which skeleton labels are present here
        skel_labels = np.unique(labeled_skel[mask])
        skel_labels = skel_labels[skel_labels > 0]
        # print (f"skel_labels: {skel_labels}")
        if len(skel_labels) == 1:
            count_touching_one += 1

    # print(f"count_touching_one: {count_touching_one} out of {num_pred}")
    return count_touching_one, num_pred




import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage.graph import route_through_array
from scipy.spatial.distance import cdist

def find_endpoints_fast(labeled_skeleton):
    """
    Finds endpoints in a labeled 3D skeletonized image in a single pass using convolution.
    
    Parameters:
        labeled_skeleton (np.ndarray): 3D array where different structures have unique labels.
        
    Returns:
        np.ndarray: 3D array with the same shape where endpoints are marked with their label values.
    """
    start_time = time.time()
    # Define 3D connectivity kernel (26-connectivity)
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0  # Exclude center voxel

    # Count neighbors for each voxel
    neighbor_count = convolve((labeled_skeleton > 0).astype(int), kernel, mode='constant', cval=0)

    # Endpoints: voxels with exactly 1 neighbor
    endpoints_mask = (neighbor_count == 1) & (labeled_skeleton > 0)

    # Assign endpoints based on their single connected neighbor's value
    labeled_neighbors = convolve(labeled_skeleton, kernel, mode='constant', cval=0)
    labeled_endpoints = endpoints_mask * labeled_neighbors  # Assign labels
    # print (f"Time taken to find endpoints: {time.time()-start_time}")
    return labeled_endpoints


import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from tqdm import tqdm
from collections import Counter
import random

import numpy as np
from scipy.ndimage import label
from collections import Counter

import numpy as np
from scipy.ndimage import label

def generate_cut_skeleton(source_labeled_skeleton, cut_source_labeled_skeleton, min_length=6, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # === 1. Get target segment lengths from source ===
    unique_labels, lengths = np.unique(source_labeled_skeleton, return_counts=True)
    filtered_lengths = [l for lbl, l in zip(unique_labels, lengths) if lbl > 0 and l >= min_length]
    desired_segments = np.array(sorted(filtered_lengths, reverse=True))  # biggest first

    # Remove skeletons in source shorter than min_length
    filtered_source_labeled = np.zeros_like(source_labeled_skeleton, dtype=source_labeled_skeleton.dtype)
    for lbl, l in zip(unique_labels, lengths):
        if lbl > 0 and l >= min_length:
            filtered_source_labeled[source_labeled_skeleton == lbl] = lbl

    # === 2. Binarize and relabel the cut source ===
    cut_binary = (cut_source_labeled_skeleton > 0).astype(np.uint8)
    cut_labeled, _ = label(cut_binary, structure=np.ones((3, 3, 3)))

    # === 3. Efficient label -> voxel mapping ===
    coords = np.argwhere(cut_labeled > 0)
    labels_at_coords = cut_labeled[tuple(coords.T)]
    label_to_coords = {}

    for coord, label_val in zip(coords, labels_at_coords):
        if label_val not in label_to_coords:
            label_to_coords[label_val] = []
        label_to_coords[label_val].append(coord)

    # Keep only skeletons long enough to be useful
    skeleton_chunks = []
    for label_val, coord_list in label_to_coords.items():
        coords_array = np.array(coord_list)
        if len(coords_array) >= min_length:
            skeleton_chunks.append((label_val, coords_array))

    # Sort largest to smallest
    skeleton_chunks.sort(key=lambda x: len(x[1]), reverse=True)

    # === 4. Fulfill each desired segment ===
    output_binary = np.zeros_like(cut_labeled, dtype=np.uint8)
    next_label = 1

    for segment_length in desired_segments:
        found = False
        for i in range(len(skeleton_chunks)):
            entry = skeleton_chunks[i]
            if entry is None:
                continue
            label_val, coords = entry
            if len(coords) >= segment_length:
                selected = coords[:segment_length]
                output_binary[tuple(np.array(selected).T)] = next_label
                next_label += 1

                remaining = coords[segment_length:]
                if len(remaining) >= min_length:
                    skeleton_chunks[i] = (label_val, remaining)
                else:
                    skeleton_chunks[i] = None  # mark for removal

                found = True
                break

        # if not found:
        #     print(f"[INFO] Skipping segment of length {segment_length} — no available skeleton.")

        # Clean up any exhausted skeletons
        skeleton_chunks = [s for s in skeleton_chunks if s is not None]
        

    return output_binary, filtered_source_labeled


def print_labeled_histogram(actual_labeled, predicted_labeled):
    
    # Calculate the unique labels in the labeled_skeleton together with their lengths for each skeleton
    unique_labels, lengths = np.unique(actual_labeled, return_counts=True)

    # Filter: skip label 0 (background), and only keep skeletons longer than 5
    filtered_lengths = [length for label, length in zip(unique_labels, lengths) if label > 0 and length > 0]

    # Count how many skeletons have each length
    hist_actual = dict(Counter(filtered_lengths))

    # Sort the histogram by length
    hist_actual = dict(sorted(hist_actual.items()))


    # Calculate the unique labels in the labeled_skeleton together with their lengths for each skeleton
    unique_labels, lengths = np.unique(predicted_labeled, return_counts=True)

    # Filter: skip label 0 (background), and only keep skeletons longer than 5
    filtered_lengths = [length for label, length in zip(unique_labels, lengths) if label > 0 and length > 0]

    # Count how many skeletons have each length
    hist_predicted = dict(Counter(filtered_lengths))

    # Sort the histogram by length
    hist_predicted = dict(sorted(hist_predicted.items()))

    print(f"skeleton_actual = {hist_actual}\nskeleton_pred = {hist_predicted}")


def CalculateTortuosity(labeled_skeleton, use_tqdm=True, n_jobs=-1, num_labels=None):
    """
    Computes tortuosity in parallel by detecting endpoints and shortest paths.

    Parameters:
        labeled_skeleton (np.ndarray): 3D array where different structures have unique labels.
        use_tqdm (bool): Whether to show progress bar with tqdm.
        n_jobs (int): Number of parallel jobs (default: -1 = all available cores)
        num_labels (int): Optional. Randomly selects up to this number of unique labels to process.

    Returns:
        list: Tortuosity values for each labeled structure.
    """


    # Find endpoints efficiently
    labeled_endpoints = find_endpoints_fast(labeled_skeleton)

    start_time = time.time()    
    # Step 1: Get all endpoint coordinates (non-zero values in labeled_endpoints)
    endpoint_coords = np.array(np.nonzero(labeled_endpoints)).T  # Shape (N, 3)

    # Step 2: Get corresponding labels (same shape as coords)
    endpoint_labels = labeled_endpoints[tuple(endpoint_coords.T)]  # Shape (N,)

    # Step 3: Use Pandas to group by label
    df = pd.DataFrame(endpoint_coords, columns=["z", "y", "x"])
    df["label"] = endpoint_labels

    # Step 4: Group by label and convert to dictionary of NumPy arrays
    endpoint_dict = {
        label: group[["z", "y", "x"]].to_numpy()
        for label, group in df.groupby("label")
    }
    # print (f"Time taken to get endpoint_dict: {time.time()-start_time}")

    # Get unique labels (excluding background)
    labels = np.unique(labeled_skeleton)
    labels = labels[labels > 0]

    # Get label positions only once
    flat_labels = labeled_skeleton.ravel()
    unique, counts = np.unique(flat_labels[flat_labels > 0], return_counts=True)

    # Map label → voxel count (skeleton length)
    skeleton_lengths = dict(zip(unique, counts))

    # Subsample if requested
    # if num_labels is not None and len(labels) > num_labels:
    #     # print (f"Subsampling labels: {num_labels} out of {len(labels)}")
    #     labels = np.random.choice(labels, size=num_labels, replace=False)



    # print (f"Preprocess time: {time.time()-preprocess_time}")

    def compute_tortuosity(label_value):
        local_times = {
            "endpoint": 0,
            "skeleton_length": 0,
            "distant_points": 0,
            "rest": 0,
            "total": 0,
        }
        t_total = time.perf_counter()

        t = time.perf_counter()
        endpoints = endpoint_dict.get(label_value, np.empty((0, 3)))        
        local_times["endpoint"] += time.perf_counter() - t

        if len(endpoints) < 2:
            return None, local_times

        t = time.perf_counter()
        skeleton_length = skeleton_lengths.get(label_value, 0)
        local_times["skeleton_length"] += time.perf_counter() - t

        t = time.perf_counter()
        dists = pdist(endpoints)
        if len(dists) == 0:
            return None, local_times
        local_times["distant_points"] += time.perf_counter() - t

        t = time.perf_counter()
        idx = np.argmax(dists)
        n = len(endpoints)
        tri_idx = np.triu_indices(n, 1)
        p1, p2 = endpoints[tri_idx[0][idx]], endpoints[tri_idx[1][idx]]
        shortest_path_length = np.linalg.norm(p1 - p2)
        local_times["rest"] += time.perf_counter() - t

        if shortest_path_length <= 0:
            return 1.0, local_times

        local_times["total"] = time.perf_counter() - t_total
        return skeleton_length / shortest_path_length, local_times


    # Run in parallel
    # if use_tqdm:
    #     results = Parallel(n_jobs=n_jobs)(
    #         delayed(compute_tortuosity)(label) for label in tqdm(labels, desc="Calculating Tortuosity", unit="structure")
    #     )
    # else:
    # Use tqdm if requested
    label_iter = tqdm(labels, desc="Calculating Tortuosity", unit="structure") if use_tqdm else labels

    t_results = time.perf_counter()
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(compute_tortuosity)(label) for label in label_iter
    # )
    results = []
    for label in label_iter:
        result = compute_tortuosity(label)
        results.append(result)

    # Separate out results and timing
    tortuosity_values = []
    times = {"endpoint": 0, "skeleton_length": 0, "distant_points": 0, "rest": 0, "total": 0}

    for r, t in results:
        if r is not None:
            tortuosity_values.append(r)
        for k in times:
            times[k] += t[k]

    # print (f"Times: {times}")
    return tortuosity_values



def CalculatePerforation(skeleton):
    # dilated_skeleton = binary_dilation(skeleton, structure=np.ones((3, 3, 3)))
    # eroded_skeleton = binary_erosion(dilated_skeleton, structure=np.ones((3, 3, 3))).astype(skeleton.dtype)
    
    # eroded_label = labelComparison(eroded_skeleton)
    morph_closing = binary_closing(skeleton, generate_binary_structure(3, 3))
    
    skeleton_label = labelComparison(skeleton)
    morph_label = labelComparison(morph_closing)
    # print (f"eroded_labelComparison: {morph_label}, skeleton_labelComparison: {skeleton_label}")

    perforation = morph_label / skeleton_label
    
    return 1/perforation

def labelComparison(skeleton):
    labeled, ncomponents_1 = label(skeleton, generate_binary_structure(3, 1))

    # label with 26
    labeled, ncomponents_3 = label(skeleton, generate_binary_structure(3, 3))

    perforation = ncomponents_3/ncomponents_1


    return perforation

def load_model_weights(model, path, device):
    checkpoint = torch.load(path, map_location=torch.device(device))

    # Handle various checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            state_dict = checkpoint['model']  # model is already a state_dict
        else:
            state_dict = checkpoint  # assume it's already the raw state_dict
    else:
        raise ValueError("❌ Unexpected checkpoint format.")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


