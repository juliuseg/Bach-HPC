import numpy as np
from Skeleton_model.st3d import structure_tensor, eig_special
from scipy.ndimage import label, binary_dilation, binary_erosion

from Skeleton_model.model import CustomUNet, transform
from Skeleton_model.Baseline_model import SkeletonBaselineModel
import torch
import time
from tqdm import tqdm


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
    return (predicted > 0.1).astype(np.float64)

def get_skeleton_vectors(skeleton_data, predicted_skeleton):
    # Define sigma and rho for structure tensor computation
    sigma = 2
    rho = 5

    st_skeleton_data = skeleton_data.copy()

    st_predicted_data = skeleton_data.copy() + predicted_skeleton.copy()

#     S = structure_tensor_3d(volume, sigma, rho)
# val, vec = eig_special_3d(S)

# # Convert from cupy to numpy. Moves data from GPU to CPU.
# val = cp.asnumpy(val)
# vec = cp.asnumpy(vec)

    # Compute structure tensors
    st_skeleton = structure_tensor(st_skeleton_data, sigma, rho)
    st_predicted = structure_tensor(st_predicted_data, sigma, rho)


    # Compute dominant eigenvectors using eig_special
    _, vec_skeleton = eig_special(st_skeleton)
    _, vec_predicted = eig_special(st_predicted)


    # Reshape vectors into (X, Y, Z, 3) format
    shape = st_skeleton_data.shape
    vec_skeleton = vec_skeleton.reshape(3, *shape).transpose(1, 2, 3, 0)
    vec_predicted = vec_predicted.reshape(3, *shape).transpose(1, 2, 3, 0)


    # Extract voxel positions where skeleton exists
    skeleton_coords = np.argwhere(skeleton_data)
    predicted_coords = np.argwhere(predicted_skeleton)

    # Extract corresponding direction vectors
    skeleton_vectors = vec_skeleton[skeleton_coords[:, 0], skeleton_coords[:, 1], skeleton_coords[:, 2]]
    predicted_vectors = vec_predicted[predicted_coords[:, 0], predicted_coords[:, 1], predicted_coords[:, 2]]
    
    # Define permutation tests
    # perm_test_skeleton_predicted = perm_test(skeleton_vectors, predicted_vectors, n_permutations=10000)

    return skeleton_vectors, predicted_vectors


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

def model_for_iterations(actual_skeleton, model, transform, device, iterations=1):
    accumulated_prediction = np.zeros_like(actual_skeleton[0])  # Initialize an empty array to sum predictions
    if (isinstance(model, SkeletonBaselineModel)):
        iterations = 1

    for _ in range(iterations):
        print ("Iteration: ", _)
        if (isinstance(model, SkeletonBaselineModel)):
            
            predicted_hole = model.get_prediction(actual_skeleton[0])[0]
        else :
            # Apply the correct transform
            actual_skeleton_tensor = transform(actual_skeleton).unsqueeze(0).to(device)  # Move to device

            # Predict the gaps
            with torch.no_grad():
                predicted_hole = model(actual_skeleton_tensor)

            predicted_hole = predicted_hole.cpu().squeeze().numpy()


        # Convert prediction
        predicted_hole = convert_prediction(predicted_hole)

        # Accumulate predictions
        accumulated_prediction += predicted_hole  # Add to accumulated predictions

        # Update actual_skeleton with the predicted_hole for the next iteration
        actual_skeleton = np.maximum(actual_skeleton, predicted_hole)

    return accumulated_prediction  # Return the summed predictions
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

    return labeled_endpoints


def CalculateTortuosity(labeled_skeleton):
    """
    Computes tortuosity in a single pass by detecting endpoints and shortest paths.

    Parameters:
        labeled_skeleton (np.ndarray): 3D array where different structures have unique labels.

    Returns:
        dict: Mapping of label -> tortuosity.
    """
    tortuosity_values = {}
    
    # Find endpoints efficiently
    labeled_endpoints = find_endpoints_fast(labeled_skeleton)
    
    # Get unique labels
    labels = np.unique(labeled_skeleton)
    labels = labels[labels > 0]  # Remove background (label 0)
    # print (len(labels))
    # Use tqdm for progress bar

    with tqdm(total=len(labels), desc="Calculating Tortuosity", unit="structure") as pbar:
        for label_value in labels:
            # Extract endpoints for this label
            endpoints = np.array(np.where(labeled_endpoints == label_value)).T

            if len(endpoints) < 2:
                pbar.update(1)
                continue  # Skip if fewer than 2 endpoints

            # Compute total skeleton length
            skeleton_length = np.sum(labeled_skeleton == label_value)

            # Find the two most distant endpoints
            dist_matrix = cdist(endpoints, endpoints)
            idx1, idx2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            p1, p2 = endpoints[idx1], endpoints[idx2]

            # Use Euclidean distance instead of pathfinding
            shortest_path_length = np.linalg.norm(p1 - p2)

            # Compute Tortuosity
            tortuosity = skeleton_length / shortest_path_length if shortest_path_length > 0 else 1
            tortuosity_values[label_value] = tortuosity

            # Update progress bar
            pbar.update(1)

    # Get the mean:
    mean_tortuosity = np.mean(list(tortuosity_values.values()))
    print (mean_tortuosity)
    return mean_tortuosity

def CalculateCompactness(skeleton):
    """surface^1.5 / volume
       edited by s204427 but got from: https://github.com/jmlipman/RatLesNetv2/blob/master/lib/metric.py#L78
    """
    pred = skeleton
    surface = np.sum(border_np(pred))
    volume = np.sum(pred)
    result = (surface**1.5)/volume
    return result

def border_np(y):
    """Calculates the border of a 3D binary map.
       From NiftyNet.
    """
    return y - binary_erosion(y)