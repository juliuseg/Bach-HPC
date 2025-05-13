import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time
from skimage.draw import line_nd

from scipy.spatial import cKDTree

class SkeletonBaselineModel:
    """
    A baseline model for connecting skeleton endpoints in 3D space.

    This model:
    - Finds the closest endpoint within a search radius.
    - Connects endpoints using a simple distance-based heuristic.
    - Uses a basic connection strategy (e.g., nearest neighbor).
    """

    def __init__(self, search_radius=12, dilation=0):
        """
        Initializes the baseline model.

        Parameters:
            search_radius (int): The maximum distance to search for endpoint connections.
        """
        self.search_radius = search_radius
        self.dilation = dilation

    def get_prediction(self, data):
        start_time = time.time()
        #print(f"{data.shape} skeletonized, unique values: {np.unique(data)}")
        endpoints = self.find_endpoints(data)
        #print (f"{endpoints.shape} endpoints ")

        skeleton_coords, skeleton_vectors = self.extract_skeleton_vectors_simple(data.astype(np.float32), endpoints.astype(np.float32))
        skeleton_coords, skeleton_vectors = self.filter_valid_vectors(skeleton_coords, skeleton_vectors)


        # Connect endpoints
        connected_endpoints = self.connect_endpoints(skeleton_coords, skeleton_vectors, shape=data.shape)
        #print(f"Connected endpoints in {time.time() - start_time} seconds")
        return connected_endpoints, endpoints, skeleton_coords, skeleton_vectors



    def find_endpoints(self, skeleton):
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

        #print(f"Amount of endpoints: {np.sum(endpoints)}")
        return endpoints.astype(np.float16)  # Return a binary 3D array with only endpoints


    def connect_endpoints(self, endpoints, vectors, shape, search_radius=10):
        """
        Connects endpoint vectors by finding nearby matching vectors and drawing connections.

        Conditions:
        - Two vectors should be facing each other (dot product < 0).
        - The vector connecting the two endpoints should be aligned with the direction (dot product > 0).

        Parameters:
            endpoints (np.ndarray): (N, 3) array of endpoint coordinates.
            vectors (np.ndarray): (N, 3) array of endpoint direction vectors.
            shape (tuple): Shape of the 3D space (X, Y, Z).
            search_radius (float): Max distance to search for other endpoints.

        Returns:
            np.ndarray: 3D binary array (1 where new connections are made, 0 elsewhere).
        """
        # Initialize output connection array
        connections = np.zeros(shape, dtype=np.uint8)

        # Timing dictionary
        timing = {
            "KDTree_build": 0,
            "query_ball_point": 0,
            "filter_indices": 0,
            "find_closest": 0,
            "vector_computation": 0,
            "dot_product_checks": 0,
            "draw_line": 0
        }

        # Build a KD-Tree for fast nearest-neighbor searching
        start_time = time.time()
        tree = cKDTree(endpoints)
        timing["KDTree_build"] = time.time() - start_time

        # Keep track of which endpoints have been used
        used = np.zeros(len(endpoints), dtype=bool)

        for i, (p1, v1) in enumerate(zip(endpoints, vectors)):
            if used[i]:
                continue  # Skip if already used

            # Find nearby endpoints within search radius
            start_time = time.time()
            indices = tree.query_ball_point(p1, search_radius)
            timing["query_ball_point"] += time.time() - start_time

            # Filter out already used points and self-matches
            start_time = time.time()
            indices = [j for j in indices if j != i and not used[j]]
            timing["filter_indices"] += time.time() - start_time

            if not indices:
                continue  # No valid matches

            # Pick the closest valid match
            start_time = time.time()
            # Pick the closest valid match# Get multiple closest neighbors
            _, candidates = tree.query(p1, k=min(5, len(endpoints)))  # Get up to 5 nearest

            # Ensure candidates are valid and not already used
            valid_candidates = [j for j in candidates if j in indices and not used[j]]

            if valid_candidates:
                # Pick the closest from the filtered candidates
                closest_idx = min(valid_candidates, key=lambda j: np.linalg.norm(endpoints[j] - p1))
            else:
                continue  # Skip if no valid candidates
            timing["find_closest"] += time.time() - start_time

            p2 = endpoints[closest_idx]
            v2 = vectors[closest_idx]

            # Compute vector between p1 and p2
            start_time = time.time()
            connection_vector = p2.astype(np.float32) - p1.astype(np.float32)
            connection_vector /= np.linalg.norm(connection_vector)  # Normalize
            timing["vector_computation"] += time.time() - start_time

            # Check conditions:
            start_time = time.time()
            if np.dot(v1, v2) >= 0:
                # print (f"skipped with: {np.dot(v1, v2)}: v1, v2", v1, v2)
                continue  # Skip this pair

            if np.dot(v1, connection_vector) <= 0.0:
                # print (f"skipped{np.dot(v1, connection_vector)}: v1, connection_vector", v1, connection_vector)
                continue  # Skip this pair
            timing["dot_product_checks"] += time.time() - start_time

            # Mark both endpoints as used
            used[i] = used[closest_idx] = True

            # Draw a connection between p1 and p2
            start_time = time.time()
            self.draw_line(connections, p1, p2)
            timing["draw_line"] += time.time() - start_time

        # Print timing results
        # print("\n--- Performance Timing ---")
        # for key, value in timing.items():
        #     print(f"{key}: {value:.6f} seconds")

        # print(f"Amount of connections: {np.sum(connections)}")
        return connections.astype(np.float32)
    
    def compute_outward_vectors(self, endpoints, skeleton):
        """
        Computes outward-facing vectors for endpoints in a 3D skeleton structure.

        Parameters:
            endpoints (np.ndarray): (N, 3) array of endpoint coordinates.
            skeleton (np.ndarray): 3D binary array representing the skeleton (1 = skeleton, 0 = background).

        Returns:
            np.ndarray: (N, 3) array of outward-pointing vectors.
        """
        # Define 26-connected neighborhood offsets
        offsets = np.array([
            (i, j, k) for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            for k in [-1, 0, 1]
            if not (i == 0 and j == 0 and k == 0)  # Exclude center voxel
        ])

        outward_vectors = np.zeros((len(endpoints), 3))  # Initialize vectors

        for i, (x, y, z) in enumerate(endpoints):
            # Generate potential neighbor coordinates
            neighbor_coords = np.array([x, y, z]) + offsets

            # Keep only valid coordinates within volume bounds
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

                # Compute the **negative** average direction (away from the skeleton)
                avg_direction = -np.mean(neighbor_directions, axis=0)

                # Normalize the vector
                norm = np.linalg.norm(avg_direction)
                if norm > 1e-6:
                    avg_direction /= norm  # Normalize

                outward_vectors[i] = avg_direction  # Store the outward vector

        return outward_vectors


    def extract_skeleton_vectors_simple(self, skeleton_data, mask=None, sigma=2, rho=5):
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
        # Extract voxel positions where skeleton exists
        coords = np.argwhere(mask)

        vectors = self.compute_outward_vectors(coords, skeleton_data)

        return coords, vectors

    
    def draw_line(self, volume, p1, p2):
        """
        Draws a 3D line between two points using `skimage.draw.line_nd`.

        Parameters:
            volume (np.ndarray): 3D binary array where the line is drawn.
            p1 (tuple): (x, y, z) coordinates of the first point.
            p2 (tuple): (x, y, z) coordinates of the second point.
        """
        p1 = np.array(p1, dtype=int)
        p2 = np.array(p2, dtype=int)

        # Generate 3D line coordinates
        line_coords = line_nd(p1, p2, endpoint=True)

        # Set the voxels in the volume
        volume[line_coords[0], line_coords[1], line_coords[2]] = 1

    def filter_valid_vectors(self, coords, vectors):
        norms = np.linalg.norm(vectors, axis=1)
        valid = (norms > 1e-6) & np.all(np.isfinite(vectors), axis=1)  # Remove NaNs and zero-length vectors
        vectors[valid] /= norms[valid, np.newaxis]  # Normalize valid vectors
        return coords[valid], vectors[valid]
