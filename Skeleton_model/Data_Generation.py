# import numpy as np
# from noise import pnoise1 
# from skimage.draw import line_nd
# from scipy.ndimage import binary_dilation


# class ArtificialSkeletonGenerator:
#     """Class to generate 3D artificial skeleton datasets with wobbly line patterns."""

#     def __init__(self, shape=(128, 128, 128)):
#         self.shape = shape  # Define the shape of the 3D volume
        

#         # Wobble strength & frequency (adjust these for different noise effects)
#         self.wobble_strengths = [7, 4, 1]
#         self.wobble_scales = [0.03, 0.06, 0.09]

#     def get_random_border_start(self):
#         """Returns a random start point along the top (y=max) or left (z=0) border."""
#         side = np.random.choice(["y=max", "z=0"])
#         if side == "y=max":
#             return np.random.randint(0, self.shape[0]), self.shape[1] - 1, np.random.randint(0, self.shape[2])
#         else:
#             return np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1]), 0

#     def connect_voxels(self, start, end,  art_data):
#         """Ensures connectivity between discrete points by interpolating."""
#         rr, cc, zz = line_nd(start, end)
#         art_data[rr, cc, zz] = 1.0
#         return rr, cc, zz

#     def calculate_radius(self, center, start):
#         """Calculate the radius of the circle given a center and a start point."""
#         return np.sqrt((start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2)

#     def get_next_point(self, center, current, step_size, clockwise=True):
#         """Find the next point moving around the circle."""
#         radius = self.calculate_radius(center, current)
#         angle = np.arctan2(current[1] - center[1], current[0] - center[0])
#         angle += -step_size if clockwise else step_size
#         new_x = center[0] + radius * np.cos(angle)
#         new_y = center[1] + radius * np.sin(angle)
#         return (new_x, new_y)

#     def size_distribution(self):
#         loaded_data = np.load("component_sizes.npy")

#         # Split into unique_sizes and counts
#         unique_sizes, counts = loaded_data[:, 0], loaded_data[:, 1]
#         probabilities = counts / counts.sum()  # Normalize to create probabilities

#         # Define a discrete random variable based on observed sizes
#         size_distribution = rv_discrete(name="component_size_dist", values=(unique_sizes, probabilities))
#         return size_distribution

#     def generate_skeleton_data(self, num_lines):
#         """Generates a 3D artificial dataset with wobbly skeleton structures."""
#         art_data = np.zeros(self.shape, dtype=np.float32)  # Initialize empty dataset
#         skeletons = []  # Store the generated skeletons
#         for i in range(num_lines):
#             x, y, z = self.get_random_border_start()
#             seed_x, seed_yz = np.random.randint(0, 100000, size=2)
#             noise_step, prev_x, prev_y, prev_z = 0, x, y, z
#             line_length = self.shape[0]*2  
#             line_index = 0

#             center = ((self.shape[1]-1)*3, (self.shape[2]-1)*2.4)
#             start = (y, z)
#             clockwise = True
#             radius = self.calculate_radius(center, start)
#             num_steps = int(2 * np.pi * radius) * 0.25
#             step_size = 2 * np.pi / num_steps
#             next_point = start
#             line_indices = []

#             while 0 <= y < self.shape[1] and 0 <= z < self.shape[2] and line_index < line_length:
#                 line_index += 1
#                 total_dx, total_dy, total_dz = 0, 0, 0
#                 for strength, scale in zip(self.wobble_strengths, self.wobble_scales):
#                     noise_x = pnoise1((noise_step + seed_x) * scale) * strength
#                     noise_yz = pnoise1((noise_step + seed_yz) * scale) * strength
#                     total_dx += noise_x / np.sqrt(2)
#                     total_dy += noise_yz / np.sqrt(2)
#                     total_dz += noise_yz / np.sqrt(2)

#                 dx, dy, dz = int(total_dx), int(total_dy), int(total_dz)
#                 next_point = self.get_next_point(center, next_point, step_size, clockwise)
#                 y, z = int(next_point[0]), int(next_point[1])

#                 y_set = np.clip(y + dy, 0, self.shape[1] - 1)
#                 z_set = np.clip(z + dz, 0, self.shape[2] - 1)
#                 x_set = np.clip(x + dx, 0, self.shape[0] - 1)

                

#                 if 0 <= y_set < self.shape[1]:
#                     if (x_set, y_set, z_set) != (prev_x, prev_y, prev_z):
#                         rr, cc, zz = self.connect_voxels((prev_x, prev_y, prev_z), (x_set, y_set, z_set),art_data)
#                         for r, c, z in zip(rr, cc, zz):
#                             line_indices.append((r, c, z))

#                     art_data[x_set, y_set, z_set] = 1.0
#                     line_indices.append((x_set, y_set, z_set))
#                     prev_x, prev_y, prev_z = x_set, y_set, z_set

#                     if (z_set >= self.shape[2] - 1):
#                         break  # Stop the line from moving further
#                 else:
#                     break

#                 noise_step += 1

#             skeletons.append(line_indices)

#         return art_data, skeletons


# import numpy as np
# import random
# from noise import pnoise1
# from skimage.draw import line_nd
# from scipy.stats import rv_discrete

# class ArtificialSkeletonGeneratorExtended(ArtificialSkeletonGenerator):
#     """Extended class that can generate skeletons and also break them into segments."""

#     def break_skeletons(self, skeletons, total_sum=1000):
#         """
#         Breaks skeletons into smaller disconnected parts based on a size distribution.

#         Args:
#             skeletons (list of lists): Each skeleton is a list of indices (tuples).
#             total_sum (int): Target number of voxels to include in split skeletons.

#         Returns:
#             np.ndarray: Array representing the broken skeleton.
#         """
#         # Load the size distribution
#         size_distribution = self.size_distribution()

#         # Generate split sizes
#         split_sizes = []
#         accumulated_sum = 0
#         while accumulated_sum < total_sum:
#             sample_size = size_distribution.rvs(size=1)[0]  # Randomly sample a segment size
#             accumulated_sum += sample_size
#             split_sizes.append(sample_size)

#         split_sizes.sort(reverse=True)  # Start with the largest segments

#         # Perform the split
#         splits = self.split_skeletons(skeletons, split_sizes)

#         # Generate new array with the selected splits
#         skeletons_splitted = self.draw_splits_to_array(self.shape, splits)

#         return skeletons_splitted

#     def split_skeletons(self, skeletons, split_sizes):
#         """
#         Splits skeletons into groups of specific lengths by randomly selecting starting points
#         within skeletons while ensuring no index is reused.

#         Args:
#             skeletons (list of lists): Each skeleton is a list of indices (tuples).
#             split_sizes (list of int): Each entry represents the number of indices required in a group.

#         Returns:
#             list of lists: Each inner list contains the indices for that split.
#         """
#         used_indices = set()  # Track indices already assigned to a split
#         splits = []

#         for size in split_sizes:
#             attempt_count = 0
#             while attempt_count < 100:  # Avoid infinite loops if space runs out
#                 # Pick a random skeleton
#                 skel = random.choice(skeletons)
#                 if len(skel) < size:
#                     attempt_count += 1
#                     continue  # Skip if skeleton is too small

#                 # Pick a random start index within the skeleton
#                 start_idx = random.randint(0, len(skel) - size)

#                 # Extract ordered segment
#                 split = skel[start_idx:start_idx + size]

#                 # Ensure no indices overlap with existing splits
#                 if any(idx in used_indices for idx in split):
#                     attempt_count += 1
#                     continue  # Retry if there's overlap

#                 # Add split to results
#                 splits.append(split)
#                 used = skel[max(start_idx-3,0):min(start_idx + size+3,len(skel)-1)]
#                 used_indices.update(split)  # Mark these indices as used
#                 break  # Move to the next split size

#             else:
#                 continue  # If no valid split found, continue with next

#         return splits

#     def draw_splits_to_array(self, shape, splits):
#         """
#         Creates a new array where selected indices from the splits are marked.

#         Args:
#             shape (tuple): Shape of the output array.
#             splits (list of lists): Each list contains (x, y, z) indices to be marked.

#         Returns:
#             np.ndarray: New array with marked indices.
#         """
#         new_array = np.zeros(shape, dtype=np.float32)

#         for split in splits:
#             for x, y, z in split:
#                 new_array[x, y, z] = 1.0  # Mark selected indices

#         return new_array

#     def generate_broken_skeleton_data(self, total_sum=1000, num_lines=15):
#         """
#         Generates both the full skeleton data and a broken version.

#         Args:
#             total_sum (int): The total number of voxels to include in the broken skeleton.

#         Returns:
#             tuple: (full skeleton, broken skeleton)
#         """
#         # Generate full skeleton first
#         full_skeleton, skeletons = self.generate_skeleton_data(num_lines=num_lines)

#         # Generate the broken skeleton
#         broken_skeleton = self.break_skeletons(skeletons, total_sum)

#         return full_skeleton, broken_skeleton


# class ArtificialSkeletonGeneratorMiddleBreak(ArtificialSkeletonGenerator):
#     """Extended class that generates a single skeleton and breaks the middle part."""

#     def generate_single_skeleton_broken_middle(self, hole_length=4, num_lines=1):
#         """
#         Generates a single skeleton and breaks a segment in the middle.

#         Args:
#             hole_length (int): Number of voxels to remove around the center.

#         Returns:
#             tuple: (full skeleton, broken skeleton)
#         """
        
#         # Generate a single skeleton
#         # set full_skeleton_all to zeros in the shape of num_lines, self.shape
#         full_skeleton_all = np.zeros((num_lines, *self.shape), dtype=np.float32)
#         skeletons_all = []
#         broken_skeleton_all = np.zeros((num_lines, *self.shape), dtype=np.float32)

#         for i in range(num_lines):
#             max_iter = 100
#             full_skeleton, skeletons = self.generate_skeleton_data(num_lines=1)  # Generate one skeleton at a time
#             while np.sum(full_skeleton) < (hole_length + 4) and max_iter > 0:
#                 full_skeleton, skeletons = self.generate_skeleton_data(num_lines=1)
#                 max_iter -= 1
            
#             skeletons = skeletons[0]

            
#             full_skeleton_all[i] = full_skeleton
#             skeletons_all.append(skeletons)
#             broken_skeleton_all[i] = self.break_middle_section(full_skeleton_all[i], skeletons_all[i], hole_length)

#         # Make hole_skeleton: full_skeleton-broken_skeleton, so we only get the hole in the middle
#         hole_skeleton_all = full_skeleton_all.copy()

#         for i in range(num_lines):
#             hole_skeleton_all[i][broken_skeleton_all[i] == 1] = 0
#             hole_skeleton_all[i] = binary_dilation(hole_skeleton_all[i], iterations=1)

#         final_hole_skeleton = np.sum(hole_skeleton_all, axis=0)
#         final_broken_skeleton = np.sum(broken_skeleton_all, axis=0)

#         return final_hole_skeleton, final_broken_skeleton
    
#     def break_middle_section(self, skeleton_array, skeleton, hole_length):
#         """
#         Removes a segment around the middle of the skeleton.

#         Args:
#             skeleton_array (np.ndarray): Full skeleton array.
#             skeleton (list of tuples): List of (x, y, z) indices forming the skeleton.
#             hole_length (int): Number of voxels to remove in total.

#         Returns:
#             np.ndarray: Skeleton array with a missing middle segment.
#         """
        
#         # Find the middle index
#         mid_index = len(skeleton) // 2

#         # Determine the range to remove
#         start_index = max(mid_index - hole_length // 2, 0)
#         end_index = min(mid_index + hole_length // 2, len(skeleton))

#         # Remove the middle part from the array
#         broken_skeleton = skeleton_array.copy()
#         for i in range(start_index, end_index):
#             x, y, z = skeleton[i]
#             broken_skeleton[x, y, z] = 0  # Set removed voxels to zero

#         return broken_skeleton
    

#     def scale_line_size(self,arr):
#         """
#         Modify a 3D array where a line of `1`s exists, increasing its thickness up to the middle
#         and then decreasing back down symmetrically.

#         Parameters:
#         arr (numpy.ndarray): A 3D numpy array containing a line represented by `1`s.

#         Returns:
#         numpy.ndarray: A new 3D numpy array with the modified line thickness.
#         """
#         # Find coordinates of the line
#         line_coords = np.argwhere(arr == 1)
        
#         if len(line_coords) == 0:
#             return arr  # No line found, return unchanged array

#         # Compute the middle index
#         mid_idx = len(line_coords) // 2
        
#         # Compute maximum radius at middle (can be tuned)
#         max_radius = max(arr.shape) // 10  # Adjust the divisor to control thickness

#         new_arr = np.zeros_like(arr)

#         for i, (x, y, z) in enumerate(line_coords):
#             # Compute thickness based on position in the line
#             if i <= mid_idx:
#                 radius = 1 + int((max_radius - 1) * (i / mid_idx))  # Scale up
#             else:
#                 radius = 1 + int((max_radius - 1) * ((len(line_coords) - i - 1) / mid_idx))  # Scale down
            
#             # Draw a small sphere around the point to increase thickness
#             for dx in range(-radius, radius + 1):
#                 for dy in range(-radius, radius + 1):
#                     for dz in range(-radius, radius + 1):
#                         if 0 <= x + dx < arr.shape[0] and 0 <= y + dy < arr.shape[1] and 0 <= z + dz < arr.shape[2]:
#                             if dx**2 + dy**2 + dz**2 <= radius**2:  # Approximate sphere
#                                 new_arr[x + dx, y + dy, z + dz] = 1
        
#         return new_arr


# def generate_skeleton_based_data(shape=(64, 64, 64), num_lines=4, hole_length=20):
#     """Wrapper function that creates a 3D artificial dataset with skeleton-like structures."""
#     generator = ArtificialSkeletonGeneratorMiddleBreak(shape=shape) # 0.000005 times shape^3
#     skeleton_data, broken_skeleton = generator.generate_single_skeleton_broken_middle(hole_length=hole_length, num_lines=num_lines)
    
#     skeleton_data = skeleton_data[np.newaxis, ...]  # Adds a new channel dimension at axis 0
#     broken_skeleton = broken_skeleton[np.newaxis, ...]  # Same for broken skeleton
    
#     return skeleton_data, broken_skeleton

# import numpy as np
# from noise import pnoise1 
# from skimage.draw import line_nd
# from scipy.ndimage import binary_dilation
# from skimage.morphology import skeletonize

# class ArtificialSkeletonGenerator:
#     """Class to generate 3D artificial skeleton datasets with wobbly line patterns."""

#     def __init__(self, shape=(128, 128, 128)):
#         self.shape = shape  # Define the shape of the 3D volume
        

#         # Wobble strength & frequency (adjust these for different noise effects)
#         self.wobble_strengths = [7, 4, 1]
#         self.wobble_scales = [0.03, 0.06, 0.09]

#     def get_random_border_start(self):
#         """Returns a random start point along the top (y=max) or left (z=0) border."""
#         side = np.random.choice(["y=max", "z=0"])
#         if side == "y=max":
#             return np.random.randint(0, self.shape[0]), self.shape[1] - 1, np.random.randint(0, self.shape[2])
#         else:
#             return np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1]), 0

#     def connect_voxels(self, start, end,  art_data):
#         """Ensures connectivity between discrete points by interpolating."""
#         rr, cc, zz = line_nd(start, end)
#         art_data[rr, cc, zz] = 1.0
#         return rr, cc, zz

#     def calculate_radius(self, center, start):
#         """Calculate the radius of the circle given a center and a start point."""
#         return np.sqrt((start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2)

#     def get_next_point(self, center, current, step_size, clockwise=True):
#         """Find the next point moving around the circle."""
#         radius = self.calculate_radius(center, current)
#         angle = np.arctan2(current[1] - center[1], current[0] - center[0])
#         angle += -step_size if clockwise else step_size
#         new_x = center[0] + radius * np.cos(angle)
#         new_y = center[1] + radius * np.sin(angle)
#         return (new_x, new_y)

#     def size_distribution(self):
#         loaded_data = np.load("component_sizes.npy")

#         # Split into unique_sizes and counts
#         unique_sizes, counts = loaded_data[:, 0], loaded_data[:, 1]
#         probabilities = counts / counts.sum()  # Normalize to create probabilities

#         # Define a discrete random variable based on observed sizes
#         size_distribution = rv_discrete(name="component_size_dist", values=(unique_sizes, probabilities))
#         return size_distribution

#     def generate_skeleton_data(self, num_lines):
#         """Generates a 3D artificial dataset with wobbly skeleton structures."""
#         art_data = np.zeros(self.shape, dtype=np.float32)  # Initialize empty dataset
#         skeletons = []  # Store the generated skeletons
#         for i in range(num_lines):
#             x, y, z = self.get_random_border_start()
#             seed_x, seed_yz = np.random.randint(0, 100000, size=2)
#             noise_step, prev_x, prev_y, prev_z = 0, x, y, z
#             line_length = self.shape[0]*2  
#             line_index = 0

#             center = ((self.shape[1]-1)*3, (self.shape[2]-1)*2.4)
#             start = (y, z)
#             clockwise = True
#             radius = self.calculate_radius(center, start)
#             num_steps = int(2 * np.pi * radius) * 0.25
#             step_size = 2 * np.pi / num_steps
#             next_point = start
#             line_indices = []

#             while 0 <= y < self.shape[1] and 0 <= z < self.shape[2] and line_index < line_length:
#                 line_index += 1
#                 total_dx, total_dy, total_dz = 0, 0, 0
#                 for strength, scale in zip(self.wobble_strengths, self.wobble_scales):
#                     noise_x = pnoise1((noise_step + seed_x) * scale) * strength
#                     noise_yz = pnoise1((noise_step + seed_yz) * scale) * strength
#                     total_dx += noise_x / np.sqrt(2)
#                     total_dy += noise_yz / np.sqrt(2)
#                     total_dz += noise_yz / np.sqrt(2)

#                 dx, dy, dz = int(total_dx), int(total_dy), int(total_dz)
#                 next_point = self.get_next_point(center, next_point, step_size, clockwise)
#                 y, z = int(next_point[0]), int(next_point[1])

#                 y_set = np.clip(y + dy, 0, self.shape[1] - 1)
#                 z_set = np.clip(z + dz, 0, self.shape[2] - 1)
#                 x_set = np.clip(x + dx, 0, self.shape[0] - 1)

                

#                 if 0 <= y_set < self.shape[1]:
#                     if (x_set, y_set, z_set) != (prev_x, prev_y, prev_z):
#                         rr, cc, zz = self.connect_voxels((prev_x, prev_y, prev_z), (x_set, y_set, z_set),art_data)
#                         for r, c, z in zip(rr, cc, zz):
#                             line_indices.append((r, c, z))

#                     art_data[x_set, y_set, z_set] = 1.0
#                     line_indices.append((x_set, y_set, z_set))
#                     prev_x, prev_y, prev_z = x_set, y_set, z_set

#                     if (z_set >= self.shape[2] - 1):
#                         break  # Stop the line from moving further
#                 else:
#                     break

#                 noise_step += 1

#             skeletons.append(line_indices)

#         return art_data, skeletons


# import numpy as np
# import random
# from noise import pnoise1
# from skimage.draw import line_nd
# from scipy.stats import rv_discrete

# class ArtificialSkeletonGeneratorExtended(ArtificialSkeletonGenerator):
#     """Extended class that can generate skeletons and also break them into segments."""

#     def break_skeletons(self, skeletons, total_sum=1000):
#         """
#         Breaks skeletons into smaller disconnected parts based on a size distribution.

#         Args:
#             skeletons (list of lists): Each skeleton is a list of indices (tuples).
#             total_sum (int): Target number of voxels to include in split skeletons.

#         Returns:
#             np.ndarray: Array representing the broken skeleton.
#         """
#         # Load the size distribution
#         size_distribution = self.size_distribution()

#         # Generate split sizes
#         split_sizes = []
#         accumulated_sum = 0
#         while accumulated_sum < total_sum:
#             sample_size = size_distribution.rvs(size=1)[0]  # Randomly sample a segment size
#             if sample_size > self.shape[0]:  # Skip if too large
#                 continue
#             accumulated_sum += sample_size
#             split_sizes.append(sample_size)
        

#         split_sizes.sort(reverse=True)  # Start with the largest segments

#         # Perform the split
#         splits = self.split_skeletons(skeletons, split_sizes)

#         # Generate new array with the selected splits
#         skeletons_splitted = self.draw_splits_to_array(self.shape, splits)

#         return skeletons_splitted

#     def split_skeletons(self, skeletons, split_sizes):
#         """
#         Splits skeletons into groups of specific lengths by randomly selecting starting points
#         within skeletons while ensuring no index is reused.

#         Args:
#             skeletons (list of lists): Each skeleton is a list of indices (tuples).
#             split_sizes (list of int): Each entry represents the number of indices required in a group.

#         Returns:
#             list of lists: Each inner list contains the indices for that split.
#         """
#         used_indices = set()  # Track indices already assigned to a split
#         splits = []

#         for size in split_sizes:
#             attempt_count = 0
#             while attempt_count < 100:  # Avoid infinite loops if space runs out
#                 # Pick a random skeleton
#                 skel = random.choice(skeletons)
#                 if len(skel) < size:
#                     attempt_count += 1
#                     continue  # Skip if skeleton is too small

#                 # Pick a random start index within the skeleton
#                 start_idx = random.randint(0, len(skel) - size)

#                 # Extract ordered segment
#                 split = skel[start_idx:start_idx + size]

#                 # Ensure no indices overlap with existing splits
#                 if any(idx in used_indices for idx in split):
#                     attempt_count += 1
#                     continue  # Retry if there's overlap

#                 # Add split to results
#                 splits.append(split)
#                 used = skel[max(start_idx-3,0):min(start_idx + size+3,len(skel)-1)]
#                 used_indices.update(used)  # Mark these indices as used
#                 break  # Move to the next split size

#             else:
#                 continue  # If no valid split found, continue with next

#         return splits

#     def draw_splits_to_array(self, shape, splits):
#         """
#         Creates a new array where selected indices from the splits are marked.

#         Args:
#             shape (tuple): Shape of the output array.
#             splits (list of lists): Each list contains (x, y, z) indices to be marked.

#         Returns:
#             np.ndarray: New array with marked indices.
#         """
#         new_array = np.zeros(shape, dtype=np.float32)

#         for split in splits:
#             for x, y, z in split:
#                 new_array[x, y, z] = 1.0  # Mark selected indices

#         return new_array

#     def generate_broken_skeleton_data(self, total_sum=1000, num_lines=15):
#         """
#         Generates both the full skeleton data and a broken version.

#         Args:
#             total_sum (int): The total number of voxels to include in the broken skeleton.

#         Returns:
#             tuple: (full skeleton, broken skeleton)
#         """
#         # Generate full skeleton first
#         full_skeleton, skeletons = self.generate_skeleton_data(num_lines=num_lines)

#         # Generate the broken skeleton
#         broken_skeleton = self.break_skeletons(skeletons, total_sum)

#         hole_skeleton = full_skeleton.copy()    
#         hole_skeleton[broken_skeleton == 1] = 0
#         hole_skeleton = binary_dilation(hole_skeleton, iterations=1)
#         return hole_skeleton, broken_skeleton



# class ArtificialSkeletonGeneratorMiddleBreak(ArtificialSkeletonGenerator):
#     """Extended class that generates a single skeleton and breaks the middle part."""

#     def generate_single_skeleton_broken_middle(self, hole_length=4, num_lines=1):
#         """
#         Generates a single skeleton and breaks a segment in the middle.

#         Args:
#             hole_length (int): Number of voxels to remove around the center.

#         Returns:
#             tuple: (full skeleton, broken skeleton)
#         """
        
#         # Generate a single skeleton
#         # set full_skeleton_all to zeros in the shape of num_lines, self.shape
#         full_skeleton_all = np.zeros((num_lines, *self.shape), dtype=np.float32)
#         skeletons_all = []
#         broken_skeleton_all = np.zeros((num_lines, *self.shape), dtype=np.float32)

#         for i in range(num_lines):
#             max_iter = 100
#             full_skeleton, skeletons = self.generate_skeleton_data(num_lines=1)  # Generate one skeleton at a time
#             while np.sum(full_skeleton) < (hole_length + 4) and max_iter > 0:
#                 full_skeleton, skeletons = self.generate_skeleton_data(num_lines=1)
#                 max_iter -= 1
#                 print("max_iter", max_iter)
            
#             skeletons = skeletons[0]

#             print ("ended with sum", np.sum(full_skeleton))
            
#             full_skeleton_all[i] = full_skeleton
#             skeletons_all.append(skeletons)
#             print ("skeletons: ", skeletons)
#             broken_skeleton_all[i] = self.break_middle_section(full_skeleton_all[i], skeletons_all[i], hole_length)

#         # Make hole_skeleton: full_skeleton-broken_skeleton, so we only get the hole in the middle
#         hole_skeleton_all = full_skeleton_all.copy()

#         for i in range(num_lines):
#             hole_skeleton_all[i][broken_skeleton_all[i] == 1] = 0
#             hole_skeleton_all[i] = binary_dilation(hole_skeleton_all[i], iterations=1)

#         final_hole_skeleton = np.sum(hole_skeleton_all, axis=0)
#         final_broken_skeleton = np.sum(broken_skeleton_all, axis=0)

#         return final_hole_skeleton, final_broken_skeleton
    
#     def break_middle_section(self, skeleton_array, skeleton, hole_length):
#         """
#         Removes a segment around the middle of the skeleton.

#         Args:
#             skeleton_array (np.ndarray): Full skeleton array.
#             skeleton (list of tuples): List of (x, y, z) indices forming the skeleton.
#             hole_length (int): Number of voxels to remove in total.

#         Returns:
#             np.ndarray: Skeleton array with a missing middle segment.
#         """
        
#         # Find the middle index
#         mid_index = len(skeleton) // 2

#         # Determine the range to remove
#         start_index = max(mid_index - hole_length // 2, 0)
#         end_index = min(mid_index + hole_length // 2, len(skeleton))

#         # Remove the middle part from the array
#         broken_skeleton = skeleton_array.copy()
#         for i in range(start_index, end_index):
#             x, y, z = skeleton[i]
#             broken_skeleton[x, y, z] = 0  # Set removed voxels to zero

#         return broken_skeleton
    

#     def scale_line_size(self,arr):
#         """
#         Modify a 3D array where a line of `1`s exists, increasing its thickness up to the middle
#         and then decreasing back down symmetrically.

#         Parameters:
#         arr (numpy.ndarray): A 3D numpy array containing a line represented by `1`s.

#         Returns:
#         numpy.ndarray: A new 3D numpy array with the modified line thickness.
#         """
#         # Find coordinates of the line
#         line_coords = np.argwhere(arr == 1)
        
#         if len(line_coords) == 0:
#             return arr  # No line found, return unchanged array

#         # Compute the middle index
#         mid_idx = len(line_coords) // 2
        
#         # Compute maximum radius at middle (can be tuned)
#         max_radius = max(arr.shape) // 10  # Adjust the divisor to control thickness

#         new_arr = np.zeros_like(arr)

#         for i, (x, y, z) in enumerate(line_coords):
#             # Compute thickness based on position in the line
#             if i <= mid_idx:
#                 radius = 1 + int((max_radius - 1) * (i / mid_idx))  # Scale up
#             else:
#                 radius = 1 + int((max_radius - 1) * ((len(line_coords) - i - 1) / mid_idx))  # Scale down
            
#             # Draw a small sphere around the point to increase thickness
#             for dx in range(-radius, radius + 1):
#                 for dy in range(-radius, radius + 1):
#                     for dz in range(-radius, radius + 1):
#                         if 0 <= x + dx < arr.shape[0] and 0 <= y + dy < arr.shape[1] and 0 <= z + dz < arr.shape[2]:
#                             if dx**2 + dy**2 + dz**2 <= radius**2:  # Approximate sphere
#                                 new_arr[x + dx, y + dy, z + dz] = 1
        
#         return new_arr


# def generate_skeleton_based_data(shape=(32, 32, 32)):
#     """Wrapper function that creates a 3D artificial dataset with skeleton-like structures."""
#     # generator = ArtificialSkeletonGeneratorMiddleBreak(shape=shape) # 0.000005 times shape^3
#     # skeleton_data, broken_skeleton = generator.generate_single_skeleton_broken_middle(hole_length=20, num_lines=4)
    
#     generator = ArtificialSkeletonGeneratorExtended(shape=shape) # 0.000005 times shape^3
#     skeleton_data, broken_skeleton = generator.generate_broken_skeleton_data(total_sum=100, num_lines=6) # For 64: 600, 20

#     skeleton_data = skeleton_data[np.newaxis, ...]  # Adds a new channel dimension at axis 0
#     broken_skeleton = broken_skeleton[np.newaxis, ...]  # Same for broken skeleton
    
#     return skeleton_data, broken_skeleton


import numpy as np
from noise import pnoise1 
from skimage.draw import line_nd
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize
from scipy import ndimage
import time
import os


class ArtificialSkeletonGenerator:
    """Class to generate 3D artificial skeleton datasets with wobbly line patterns."""

    def __init__(self, shape=(128, 128, 128)):
        self.shape = shape  # Define the shape of the 3D volume
        

        # Wobble strength & frequency (adjust these for different noise effects)
        self.wobble_strengths = [7, 4, 1]
        self.wobble_scales = [0.03, 0.06, 0.09]

    def get_random_border_start(self):
        """Returns a random start point along the top (y=max) or left (z=0) border."""
        side = np.random.choice(["y=max", "z=0"])
        if side == "y=max":
            return np.random.randint(0, self.shape[0]), self.shape[1] - 1, np.random.randint(0, self.shape[2])
        else:
            return np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1]), 0

    def connect_voxels(self, start, end,  art_data):
        """Ensures connectivity between discrete points by interpolating."""
        rr, cc, zz = line_nd(start, end)
        art_data[rr, cc, zz] = 1.0
        return rr, cc, zz

    def calculate_radius(self, center, start):
        """Calculate the radius of the circle given a center and a start point."""
        return np.sqrt((start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2)

    def get_next_point(self, center, current, step_size, clockwise=True):
        """Find the next point moving around the circle."""
        radius = self.calculate_radius(center, current)
        angle = np.arctan2(current[1] - center[1], current[0] - center[0])
        angle += -step_size if clockwise else step_size
        new_x = center[0] + radius * np.cos(angle)
        new_y = center[1] + radius * np.sin(angle)
        return (new_x, new_y)

    def size_distribution(self):
        loaded_data = np.load("component_sizes.npy")

        # Split into unique_sizes and counts
        unique_sizes, counts = loaded_data[:, 0], loaded_data[:, 1]
        probabilities = counts / counts.sum()  # Normalize to create probabilities

        # Define a discrete random variable based on observed sizes
        size_distribution = rv_discrete(name="component_size_dist", values=(unique_sizes, probabilities))
        return size_distribution

    def generate_skeleton_data(self, num_lines):
        """Generates a 3D artificial dataset with wobbly skeleton structures."""
        art_data = np.zeros(self.shape, dtype=np.float32)  # Initialize empty dataset
        skeletons = []  # Store the generated skeletons
        for i in range(num_lines):
            x, y, z = self.get_random_border_start()
            seed_x, seed_yz = np.random.randint(0, 100000, size=2)
            noise_step, prev_x, prev_y, prev_z = 0, x, y, z
            line_length = self.shape[0]*2  
            line_index = 0

            center = ((self.shape[1]-1)*3, (self.shape[2]-1)*2.4)
            start = (y, z)
            clockwise = True
            radius = self.calculate_radius(center, start)
            num_steps = int(2 * np.pi * radius) * 0.25
            step_size = 2 * np.pi / num_steps
            next_point = start
            line_indices = []

            while 0 <= y < self.shape[1] and 0 <= z < self.shape[2] and line_index < line_length:
                line_index += 1
                total_dx, total_dy, total_dz = 0, 0, 0
                for strength, scale in zip(self.wobble_strengths, self.wobble_scales):
                    noise_x = pnoise1((noise_step + seed_x) * scale) * strength
                    noise_yz = pnoise1((noise_step + seed_yz) * scale) * strength
                    total_dx += noise_x / np.sqrt(2)
                    total_dy += noise_yz / np.sqrt(2)
                    total_dz += noise_yz / np.sqrt(2)

                dx, dy, dz = int(total_dx), int(total_dy), int(total_dz)
                next_point = self.get_next_point(center, next_point, step_size, clockwise)
                y, z = int(next_point[0]), int(next_point[1])

                y_set = np.clip(y + dy, 0, self.shape[1] - 1)
                z_set = np.clip(z + dz, 0, self.shape[2] - 1)
                x_set = np.clip(x + dx, 0, self.shape[0] - 1)

                

                if 0 <= y_set < self.shape[1]:
                    if (x_set, y_set, z_set) != (prev_x, prev_y, prev_z):
                        rr, cc, zz = self.connect_voxels((prev_x, prev_y, prev_z), (x_set, y_set, z_set),art_data)
                        for r, c, z in zip(rr, cc, zz):
                            line_indices.append((r, c, z))

                    art_data[x_set, y_set, z_set] = 1.0
                    line_indices.append((x_set, y_set, z_set))
                    prev_x, prev_y, prev_z = x_set, y_set, z_set

                    if (z_set >= self.shape[2] - 1):
                        break  # Stop the line from moving further
                else:
                    break

                noise_step += 1

            skeletons.append(line_indices)

        return art_data, skeletons


import numpy as np
import random
from noise import pnoise1
from skimage.draw import line_nd
from scipy.stats import rv_discrete

class ArtificialSkeletonGeneratorExtended(ArtificialSkeletonGenerator):
    """Extended class that can generate skeletons and also break them into segments."""

    def break_skeletons(self, full_skeleton, skeletons, total_sum=1000):
        """
        Breaks skeletons into smaller disconnected parts based on a size distribution.

        Args:
            skeletons (list of lists): Each skeleton is a list of indices (tuples).
            total_sum (int): Target number of voxels to include in split skeletons.

        Returns:
            np.ndarray: Array representing the broken skeleton.
        """
        # Load the size distribution
        size_distribution = self.size_distribution()

        

        # Generate split sizes
        split_sizes = []
        accumulated_sum = 0
        while accumulated_sum < total_sum:
            sample_size = size_distribution.rvs(size=1)[0]  # Randomly sample a segment size
            if sample_size > self.shape[0]:  # Skip if too large
                continue
            accumulated_sum += sample_size
            split_sizes.append(sample_size)
        

        split_sizes.sort(reverse=True)  # Start with the largest segments

        # Perform the split
        splits = self.split_skeletons(skeletons, split_sizes)

        # Generate new array with the selected splits
        skeletons_splitted = np.zeros(self.shape, dtype=np.float32)
        for split in splits:
            for x, y, z in split:
                skeletons_splitted[x, y, z] = 1

        hole_skeleton = full_skeleton.copy()    
        hole_skeleton[skeletons_splitted == 1] = 0

        # Now we have hole skeleton and we just need to label it



        
        return hole_skeleton, 1

    def split_skeletons(self, skeletons, split_sizes):
        """
        Splits skeletons into groups of specific lengths by randomly selecting starting points
        within skeletons while ensuring no index is reused.

        Args:
            skeletons (list of lists): Each skeleton is a list of indices (tuples).
            split_sizes (list of int): Each entry represents the number of indices required in a group.

        Returns:
            list of lists: Each inner list contains the indices for that split.
        """
        used_indices = set()  # Track indices already assigned to a split
        splits = []

        for size in split_sizes:
            attempt_count = 0
            while attempt_count < 100:  # Avoid infinite loops if space runs out
                # Pick a random skeleton
                skel = random.choice(skeletons)
                if len(skel) < size:
                    attempt_count += 1
                    continue  # Skip if skeleton is too small

                # Pick a random start index within the skeleton
                start_idx = random.randint(0, len(skel) - size)

                # Extract ordered segment
                split = skel[start_idx:start_idx + size]

                # Ensure no indices overlap with existing splits
                if any(idx in used_indices for idx in split):
                    attempt_count += 1
                    continue  # Retry if there's overlap

                # Add split to results
                splits.append(split)
                used = skel[max(start_idx-3,0):min(start_idx + size+3,len(skel)-1)]
                used_indices.update(used)  # Mark these indices as used
                break  # Move to the next split size

            else:
                continue  # If no valid split found, continue with next

        return splits

    def draw_splits_to_array(self, splits):
        """
        Creates a new array where selected indices from the splits are marked.

        Args:
            shape (tuple): Shape of the output array.
            splits (list of lists): Each list contains (x, y, z) indices to be marked.

        Returns:
            np.ndarray: New array with marked indices.
        """
        new_array = np.zeros(self.shape, dtype=np.float32)

        label = 1
        for split in splits:
            for x, y, z in split:
                new_array[x, y, z] = label
            label += 1  

        return new_array, label

    def generate_broken_skeleton_data(self, total_sum=1000, num_lines=15,dilation=1):
        """
        Generates both the full skeleton data and a broken version.

        Args:
            total_sum (int): The total number of voxels to include in the broken skeleton.

        Returns:
            tuple: (full skeleton, broken skeleton)
        """
        # Generate full skeleton first
        full_skeleton, skeletons = self.generate_skeleton_data(num_lines=num_lines)
        
        # Generate the broken skeleton
        hole_skeleton, num_labels = self.break_skeletons(full_skeleton.copy(), skeletons, total_sum) # We gonna make break_skeletons return like label.
       
        broken_skeleton = full_skeleton.copy()
        broken_skeleton[hole_skeleton > 0] = 0

        hole_skeleton = self.remove_border_holes(hole_skeleton)

        if(dilation>0):
            hole_skeleton = binary_dilation(hole_skeleton, iterations=dilation)



        return broken_skeleton, hole_skeleton

    def remove_border_holes(self, hole_skeleton):
        connectivity = np.ones((3, 3, 3))
        labeled_holes, num_holes = ndimage.label(hole_skeleton, structure=connectivity)
        
        # Create a single 3D border mask
        border = np.zeros_like(hole_skeleton, dtype=bool)
        border[0, :, :] = True
        border[-1, :, :] = True
        border[:, 0, :] = True
        border[:, -1, :] = True
        border[:, :, 0] = True
        border[:, :, -1] = True

        # Find labels that touch the border
        border_labels = np.unique(labeled_holes[border])  # Get all hole labels that touch the border
        border_labels = border_labels[border_labels > 0]  # Remove background (0)

        # Create a mask for all holes to remove
        remove_mask = np.isin(labeled_holes, border_labels)

        # Remove holes touching the border
        hole_skeleton[remove_mask] = 0

        return hole_skeleton



class ArtificialSkeletonGeneratorMiddleBreak(ArtificialSkeletonGenerator):
    """Extended class that generates a single skeleton and breaks the middle part."""

    def generate_single_skeleton_broken_middle(self, hole_length=4, num_lines=1):
        """
        Generates a single skeleton and breaks a segment in the middle.

        Args:
            hole_length (int): Number of voxels to remove around the center.

        Returns:
            tuple: (full skeleton, broken skeleton)
        """
        
        # Generate a single skeleton
        # set full_skeleton_all to zeros in the shape of num_lines, self.shape
        full_skeleton_all = np.zeros((num_lines, *self.shape), dtype=np.float32)
        skeletons_all = []
        broken_skeleton_all = np.zeros((num_lines, *self.shape), dtype=np.float32)

        for i in range(num_lines):
            max_iter = 100
            full_skeleton, skeletons = self.generate_skeleton_data(num_lines=1)  # Generate one skeleton at a time
            while np.sum(full_skeleton) < (hole_length + 4) and max_iter > 0:
                full_skeleton, skeletons = self.generate_skeleton_data(num_lines=1)
                max_iter -= 1
                print("max_iter", max_iter)
            
            skeletons = skeletons[0]

            print ("ended with sum", np.sum(full_skeleton))
            
            full_skeleton_all[i] = full_skeleton
            skeletons_all.append(skeletons)
            print ("skeletons: ", skeletons)
            broken_skeleton_all[i] = self.break_middle_section(full_skeleton_all[i], skeletons_all[i], hole_length)

        # Make hole_skeleton: full_skeleton-broken_skeleton, so we only get the hole in the middle
        hole_skeleton_all = full_skeleton_all.copy()

        for i in (num_lines):
            hole_skeleton_all[i][broken_skeleton_all[i] == 1] = 0
            hole_skeleton_all[i] = binary_dilation(hole_skeleton_all[i], iterations=1)

        final_hole_skeleton = np.sum(hole_skeleton_all, axis=0)
        final_broken_skeleton = np.sum(broken_skeleton_all, axis=0)

        return final_hole_skeleton, final_broken_skeleton
    
    def break_middle_section(self, skeleton_array, skeleton, hole_length):
        """
        Removes a segment around the middle of the skeleton.

        Args:
            skeleton_array (np.ndarray): Full skeleton array.
            skeleton (list of tuples): List of (x, y, z) indices forming the skeleton.
            hole_length (int): Number of voxels to remove in total.

        Returns:
            np.ndarray: Skeleton array with a missing middle segment.
        """
        
        # Find the middle index
        mid_index = len(skeleton) // 2

        # Determine the range to remove
        start_index = max(mid_index - hole_length // 2, 0)
        end_index = min(mid_index + hole_length // 2, len(skeleton))

        # Remove the middle part from the array
        broken_skeleton = skeleton_array.copy()
        for i in range(start_index, end_index):
            x, y, z = skeleton[i]
            broken_skeleton[x, y, z] = 0  # Set removed voxels to zero

        return broken_skeleton
    

    def scale_line_size(self,arr):
        """
        Modify a 3D array where a line of `1`s exists, increasing its thickness up to the middle
        and then decreasing back down symmetrically.

        Parameters:
        arr (numpy.ndarray): A 3D numpy array containing a line represented by `1`s.

        Returns:
        numpy.ndarray: A new 3D numpy array with the modified line thickness.
        """
        # Find coordinates of the line
        line_coords = np.argwhere(arr == 1)
        
        if len(line_coords) == 0:
            return arr  # No line found, return unchanged array

        # Compute the middle index
        mid_idx = len(line_coords) // 2
        
        # Compute maximum radius at middle (can be tuned)
        max_radius = max(arr.shape) // 10  # Adjust the divisor to control thickness

        new_arr = np.zeros_like(arr)

        for i, (x, y, z) in enumerate(line_coords):
            # Compute thickness based on position in the line
            if i <= mid_idx:
                radius = 1 + int((max_radius - 1) * (i / mid_idx))  # Scale up
            else:
                radius = 1 + int((max_radius - 1) * ((len(line_coords) - i - 1) / mid_idx))  # Scale down
            
            # Draw a small sphere around the point to increase thickness
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if 0 <= x + dx < arr.shape[0] and 0 <= y + dy < arr.shape[1] and 0 <= z + dz < arr.shape[2]:
                            if dx**2 + dy**2 + dz**2 <= radius**2:  # Approximate sphere
                                new_arr[x + dx, y + dy, z + dz] = 1
        
        return new_arr

def generate_skeleton_based_data(shape=(64,64,64), total_sum=500, num_lines=20):
#def generate_skeleton_based_data(shape=(128,128,128), total_sum=500*8, num_lines=20*8):
    """Wrapper function that creates a 3D artificial dataset with skeleton-like structures."""
    start_time = time.time()

    generator = ArtificialSkeletonGeneratorExtended(shape=shape) # 0.000005 times shape^3
    skeleton_data, broken_skeleton = generator.generate_broken_skeleton_data(total_sum=total_sum, num_lines=num_lines, dilation=0) # For 64: 500, 20
    skeleton_data = skeleton_data[np.newaxis, ...]  # Adds a new channel dimension at axis 0

    broken_skeleton = broken_skeleton[np.newaxis, ...]  # Same for broken skeleton
    #print(f"Data generation took: {time.time() - start_time:.2f} seconds")
    return skeleton_data, broken_skeleton


if __name__ == "__main__":
    skeleton_data, broken_skeleton = generate_skeleton_based_data()

    # Show with napari
    import napari
    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(broken_skeleton, name='Broken Skeleton Data', colormap='green')
    viewer.add_image(skeleton_data, name='Skeleton Data', colormap='blue')

    napari.run()
