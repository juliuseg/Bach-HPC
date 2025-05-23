import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
from scipy.ndimage import convolve
from st3d import structure_tensor, eig_special
import time
from skimage.draw import line_nd

from scipy.spatial import cKDTree




def mean_direction(vecs):
    """Compute mean direction of a set of unit vectors."""
    mean_vec = np.mean(vecs, axis=0)
    return mean_vec / np.linalg.norm(mean_vec)  # Normalize to get a unit vector

def angular_distance(vec1, vec2):
    """Compute the angular distance between two unit vectors."""
    dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)  # Clip to avoid numerical issues
    if dot_product < 0:
        dot_product = -dot_product
    return np.arccos(dot_product)  # Returns angle in radians

def perm_test(vecs_orig, vecs_hole, n_orig=None, n_hole=None, n_permutations=1000):
    """
    Perform a permutation test on two sets of 3D unit vectors by comparing
    the angular distance between their mean directions.
    """

    if n_orig is None:
        n_orig = len(vecs_orig)
    if n_hole is None:
        n_hole = len(vecs_hole)

    # Compute observed test statistic: Angular distance between mean directions
    mean_orig = mean_direction(vecs_orig)
    mean_hole = mean_direction(vecs_hole)
    obs_diff = angular_distance(mean_orig, mean_hole)

    # Permutation test setup
    pooled = np.vstack([vecs_orig, vecs_hole])
    n_total = pooled.shape[0]
    null_distribution = np.zeros(n_permutations)

    # Permutation loop
    for i in range(n_permutations):
        perm_indices = np.random.choice(n_total, n_orig + n_hole, replace=False)  # Random indices
        perm_group1 = pooled[perm_indices[:n_orig]]
        perm_group2 = pooled[perm_indices[n_orig:]]
        
        # Compute test statistic for permuted data
        null_distribution[i] = angular_distance(mean_direction(perm_group1), mean_direction(perm_group2))

    # Compute p-value
    p_value = np.mean(null_distribution >= obs_diff)

    return p_value



def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
def score(n_skeletons_orig, n_skeletons_new, struc_pval):
    struc_score = 1 if struc_pval > 0.05 else map_range(struc_pval, 0, 0.05, 0, 1)
    return n_skeletons_orig/n_skeletons_new * struc_score



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

    return filtered_predicted_hole


def find_endpoints(skeleton):
    """
    Finds endpoints in a 3D skeletonized binary image.
    
    Parameters:
        skeleton (np.ndarray): A 3D binary skeleton (1s for skeleton, 0s for background).
        
    Returns:
        np.ndarray: A 3D binary array of the same shape with only endpoints marked as 1.
    """
    # Define a 3D connectivity kernel (26-connectivity)
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0  # Exclude the center voxel

    # Count the number of neighbors for each voxel
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)

    # Endpoints are voxels with exactly **one** neighbor
    endpoints = (neighbor_count <= 1) & (skeleton == 1)

    print(f"Amount of endpoints: {np.sum(endpoints)}")
    return endpoints.astype(np.float16)  # Return a binary 3D array with only endpoints

def extract_skeleton_vectors(skeleton_data, mask=None, sigma=2, rho=5):
    """
    Computes structure tensors and extracts dominant eigenvectors for a given skeletonized dataset.

    Parameters:
        skeleton_data (np.ndarray): 3D binary array of the skeleton.
        sigma (float): Gaussian smoothing parameter for local structure tensor computation.
        rho (float): Gaussian smoothing parameter for tensor regularization.

    Returns:
        tuple: (coords, vectors)
            - coords (np.ndarray): Array of shape (N, 3) containing voxel coordinates where skeleton exists.
            - vectors (np.ndarray): Array of shape (N, 3) containing dominant eigenvectors at those locations.
    """
    if mask is None:
        mask = skeleton_data
    # Compute structure tensor
    st_data = skeleton_data.copy()
    st_tensor = structure_tensor(np.swapaxes(st_data, 0, 2), sigma, rho)

    # Compute dominant eigenvectors
    _, vec = eig_special(st_tensor)

    # Reshape vectors into (X, Y, Z, 3) format
    shape = st_data.shape
    vec = vec.reshape(3, *shape).transpose(1, 2, 3, 0)

    # Extract voxel positions where skeleton exists
    coords = np.argwhere(mask)


    # Extract corresponding direction vectors
    vectors = vec[coords[:, 0], coords[:, 1], coords[:, 2]]

    vectors = correct_endpoint_directions (vectors, coords, skeleton_data)
    #vectors = compute_outward_vectors(coords, skeleton_data)

    # print (f"coords: {coords.shape}, vectors: {vectors.shape}, coords_sd: {coords_sd.shape}")
    return coords, vectors



def filter_valid_vectors(coords, vectors):
    norms = np.linalg.norm(vectors, axis=1)
    valid = (norms > 1e-6) & np.all(np.isfinite(vectors), axis=1)  # Remove NaNs and zero-length vectors
    vectors[valid] /= norms[valid, np.newaxis]  # Normalize valid vectors
    return coords[valid], vectors[valid]

import numpy as np
import time

def correct_endpoint_directions(vectors, endpoints, skeleton):
    """
    Corrects endpoint directions to ensure they point outward from the skeleton.
    Includes detailed debug information for each flipped and non-flipped vector.

    Parameters:
        vectors (np.ndarray): (N, 3) array of endpoint direction vectors.
        endpoints (np.ndarray): (N, 3) array of endpoint coordinates.
        skeleton (np.ndarray): 3D binary array representing the skeleton structure (1 = skeleton, 0 = background).

    Returns:
        np.ndarray: (N, 3) array of corrected vectors.
    """
    # Print debug info
    print(f"Endpoints shape: {endpoints.shape}")
    print(f"Vectors shape: {vectors.shape}")
    print(f"Skeleton shape: {skeleton.shape}")
                             
    start_time = time.time()
    corrected_vectors = vectors.copy()

    # Define 26-connected neighborhood offsets
    offsets = np.array([
        (i, j, k) for i in [-1, 0, 1]
        for j in [-1, 0, 1]
        for k in [-1, 0, 1]
        if not (i == 0 and j == 0 and k == 0)  # Exclude center voxel
    ])

    for i, (x, y, z) in enumerate(endpoints):
        # Generate potential neighbor coordinates
        neighbor_coords = np.array([x, y, z]) + offsets

        # Keep only valid coordinates within the volume
        valid_mask = (
            (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < skeleton.shape[0]) &
            (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < skeleton.shape[1]) &
            (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < skeleton.shape[2])
        )
        neighbor_coords = neighbor_coords[valid_mask]

        # Get neighbors that are part of the skeleton
        neighbor_mask = skeleton[neighbor_coords[:, 0], neighbor_coords[:, 1], neighbor_coords[:, 2]] == 1
        valid_neighbors = neighbor_coords[neighbor_mask]

        if len(valid_neighbors) > 0:
            # Compute direction vectors to neighbors
            neighbor_directions = valid_neighbors - np.array([x, y, z])

            # Compute the average direction
            avg_direction = np.mean(neighbor_directions, axis=0)

            # Handle zero-length vectors before normalization
            norm = np.linalg.norm(avg_direction)
            if norm > 1e-6:
                avg_direction /= norm  # Normalize

                # Compute dot product and check if we need to flip
                dot_product = np.dot(vectors[i], avg_direction)
                should_flip = dot_product < 0

                # Print debug information
                print("\n--- DEBUG: Checking Endpoint ---")
                print(f"Endpoint at: {x, y, z}")
                print(f"Original Vector: {vectors[i]}")
                print(f"Average Neighbor Direction: {avg_direction}")
                print(f"Dot Product: {dot_product}")
                print(f"Should Flip: {should_flip}")

                # Print 3x3 local area
                x_min, x_max = max(0, x - 1), min(skeleton.shape[0], x + 2)
                y_min, y_max = max(0, y - 1), min(skeleton.shape[1], y + 2)
                z_min, z_max = max(0, z - 1), min(skeleton.shape[2], z + 2)
                print("3x3 neighborhood:")
                print(skeleton[x_min:x_max, y_min:y_max, z_min:z_max])

                # Flip vector if needed
                if should_flip:
                    corrected_vectors[i] *= -1  # Flip the vector
                    print("✅ Vector was flipped!\n")
                else:
                    print("❌ No flipping needed.\n")

    print(f"Corrected {len(endpoints)} endpoints in {time.time() - start_time:.2f} seconds.")
    return corrected_vectors
