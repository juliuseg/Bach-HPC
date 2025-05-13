import numpy as np
from multiprocessing import Pool, cpu_count
from noise import pnoise3, pnoise1
from skimage.morphology import skeletonize
from scipy import ndimage
from skimage.morphology import ball
import numpy as np
import time
import os
from scipy.ndimage import find_objects
from scipy.stats import rankdata, rv_discrete
import bisect
import random
from skimage.draw import line_nd
from scipy.ndimage import label, binary_dilation, generate_binary_structure,find_objects
from collections import defaultdict


class ArtificialSkeletonGenerator:
    """Class to generate 3D artificial skeleton datasets with wobbly line patterns."""

    def __init__(self, shape=(128, 128, 128), wobble=1.5):
        self.shape = shape  # Define the shape of the 3D volume
        

        # Wobble strength & frequency (adjust these for different noise effects)
        self.wobble_strengths = [7.0, 4.0, 1.0]
        self.wobble_scales = [0.03, 0.06, 0.09]

        self.wobble_strengths = [x * wobble for x in self.wobble_strengths]
        self.wobble_scales = [x * wobble for x in self.wobble_scales]

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
        art_data[rr, cc, zz] = 1
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
        start_time = time.time()
        p_noise_time = 0
        connect_voxels_time = 0
        clip_time = 0
        get_next_point_time = 0
        art_data = np.zeros(self.shape, dtype=np.uint8)  # Initialize empty dataset
        skeletons = []  # Store the generated skeletons

        mod_size = num_lines//100
        for i in range(num_lines):
            # if i % (mod_size*10) == 0:
            #     #print (f"Generating, percentage: {i//mod_size} out of {num_lines//mod_size}, time: {time.time()-start_time}")

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
                    p_noise_start = time.time()
                    noise_x = pnoise1((noise_step + seed_x) * scale) * strength
                    noise_yz = pnoise1((noise_step + seed_yz) * scale) * strength
                    p_noise_time += time.time() - p_noise_start
                    total_dx += noise_x / np.sqrt(2)
                    total_dy += noise_yz / np.sqrt(2)
                    total_dz += noise_yz / np.sqrt(2)

                dx, dy, dz = int(total_dx), int(total_dy), int(total_dz)

                get_next_point_start = time.time()
                next_point = self.get_next_point(center, next_point, step_size, clockwise)
                get_next_point_time += time.time() - get_next_point_start
                y, z = int(next_point[0]), int(next_point[1])

                clip_start = time.time()
                y_set = np.clip(y + dy, 0, self.shape[1] - 1)
                z_set = np.clip(z + dz, 0, self.shape[2] - 1)
                x_set = np.clip(x + dx, 0, self.shape[0] - 1)
                clip_time += time.time() - clip_start

                

                if 0 <= y_set < self.shape[1]:
                    if (x_set, y_set, z_set) != (prev_x, prev_y, prev_z):
                        connect_voxels_start = time.time()
                        rr, cc, zz = self.connect_voxels((prev_x, prev_y, prev_z), (x_set, y_set, z_set),art_data)
                        connect_voxels_time += time.time() - connect_voxels_start
                        for r, c, z in zip(rr, cc, zz):
                            line_indices.append((r, c, z))

                    art_data[x_set, y_set, z_set] = 1
                    line_indices.append((x_set, y_set, z_set))
                    prev_x, prev_y, prev_z = x_set, y_set, z_set

                    if (z_set >= self.shape[2] - 1):
                        break  # Stop the line from moving further
                else:
                    break

                noise_step += 1

            skeletons.append(line_indices)

        return art_data, skeletons


class ArtificialSkeletonGeneratorExtended(ArtificialSkeletonGenerator):
    """Extended class that can generate skeletons and also break them into segments."""

    def break_skeletons(self, full_skeleton, skeletons, gap_threshold = 10, gap_min=1, gap_chance=0.3):
        """
        Breaks skeletons into smaller disconnected parts using a size distribution.
        Inserts gaps between them (some large, some small).

        Returns:
            skeletons_splitted: volume with retained skeleton segments
            small_hole_skeleton: volume with small gaps (1–10)
            large_hole_skeleton: volume with large gaps (30–50)
        """
        size_distribution = self.size_distribution()

        splits, small_gaps, large_gaps = self.split_skeletons(skeletons, size_distribution, gap_chance, gap_threshold, gap_min=gap_min)

        shape = self.shape
        skeletons_splitted = np.zeros(shape, dtype=np.uint8)
        small_hole_skeleton = np.zeros(shape, dtype=np.uint8)
        large_hole_skeleton = np.zeros(shape, dtype=np.uint8)

        for split in splits:
            for x, y, z in split:
                skeletons_splitted[x, y, z] = 1

        for gap in small_gaps:
            for x, y, z in gap:
                small_hole_skeleton[x, y, z] = 1

        for gap in large_gaps:
            for x, y, z in gap:
                large_hole_skeleton[x, y, z] = 1

        return skeletons_splitted, small_hole_skeleton, large_hole_skeleton


    def split_skeletons(self, skeletons, size_distribution, large_gap_chance=0.3, gap_threshold=10, gap_min=1):
        splits = []
        small_gaps = []
        large_gaps = []

        n_small = [0, 0, 0]

        for skel in skeletons:
            i = 0
            fail_count = 0

            # Pre-sample initial split size
            split_size = size_distribution.rvs(size=1)[0]

            while i < len(skel) and fail_count < 5:
                # Skip tiny segments sometimes
                if split_size <= 3 and random.random() < 0.5:
                    n_small[split_size - 1] += 1
                    split_size = size_distribution.rvs(size=1)[0]
                    continue

                # Sample gap
                if random.random() < large_gap_chance:
                    gap_size = random.randint(gap_threshold, 50)
                    is_large = True
                else:
                    gap_size = random.randint(gap_min, gap_threshold)
                    is_large = False

                # Sample the NEXT split size in advance
                next_split_size = size_distribution.rvs(size=1)[0]

                total_needed = split_size + gap_size + next_split_size
                if i + total_needed >= len(skel):
                    fail_count += 1
                    break

                # Add current split
                split = skel[i:i + split_size]
                splits.append(split)
                i += split_size

                # Add gap
                gap = skel[i:i + gap_size]
                (large_gaps if is_large else small_gaps).append(gap)
                i += gap_size

                # Prepare for next loop
                split_size = next_split_size
                fail_count = 0  # reset fail count

        return splits, small_gaps, large_gaps#, n_small





    def draw_splits_to_array(self, splits):
        """
        Creates a new array where selected indices from the splits are marked.

        Args:
            shape (tuple): Shape of the output array.
            splits (list of lists): Each list contains (x, y, z) indices to be marked.

        Returns:
            np.ndarray: New array with marked indices.
        """
        new_array = np.zeros(self.shape, dtype=np.uint8)

        label = 1
        for split in splits:
            for x, y, z in split:
                new_array[x, y, z] = label
            label += 1  

        return new_array, label

    def generate_broken_skeleton_data(self, num_lines=15, max_gap=20, gap_threshold=10, gap_min=1, gap_chance=0.3, thickness=0, save_full_skeleton=False):
        """
        Generates both the full skeleton data and a broken version.

        Args:
            total_sum (int): The total number of voxels to include in the broken skeleton.

        Returns:
            tuple: (full skeleton, broken skeleton)
        """
        # Generate full skeleton first
        start_time = time.time()
        full_skeleton, skeletons = self.generate_skeleton_data(num_lines=num_lines)
        #print(f"Generated full skeleton in {time.time() - start_time:.2f} seconds.")
        start_time = time.time()
        #print(f"Removed close by dilation in {time.time() - start_time:.2f} seconds.")
        start_time = time.time()

        if save_full_skeleton:
            # Define the folder path
            file_path = "FullSkeletonData"

            # Ensure the directory exists
            os.makedirs(file_path, exist_ok=True)

            # Save the full skeleton to a file
            np.save(os.path.join(file_path, "full_skeleton.npy"), full_skeleton)

            #print(f"Saved full skeleton to {file_path}/full_skeleton.npy")
        
        # Generate the broken skeleton
        skeletons_splitted, small_hole_skeleton, large_hole_skeleton = self.break_skeletons(full_skeleton.copy(), skeletons, gap_threshold = gap_threshold, gap_min=gap_min, gap_chance=gap_chance)#, total_sum=total_sum) # We gonna make break_skeletons return like label.

        hole_skeleton = small_hole_skeleton.copy()
        # broken_skeleton = full_skeleton.copy()
        # broken_skeleton[hole_skeleton > 0] = 0

        # skeletons_splitted_new = self.remove_close_by_dilation(skeletons_splitted, hole_skeleton, dilation_radius=1)
        # skeletons_splitted = skeletons_splitted_new
        
        # hole_skeleton = full_skeleton.copy() 
        # hole_skeleton[skeletons_splitted == 1] = 0

        # hole_addtion = self.find_short_skeletons_as_holes(skeletons_splitted, hole_skeleton)
        # hole_skeleton = hole_skeleton + hole_addtion


        #print(f"Generated broken skeleton in {time.time() - start_time:.2f} seconds.")
        start_time = time.time()

        start_time = time.time()
        # hole_skeleton = self.remove_skeleton_under_length(hole_skeleton, max_gap)
        # #print(f"Removed under lenght holes in {time.time() - start_time:.2f} seconds.")
        start_time = time.time()


        # if (thickness >= 1):
        #     hole_skeleton = self.spherical_dilation(hole_skeleton, radius=thickness)

        #print (f"Time taken for dilation: {time.time()-start_time}")



        return skeletons_splitted, hole_skeleton, large_hole_skeleton
    
    def spherical_dilation(self, volume, radius):
        # Create the spherical (ball) structuring element
        struct_elem = ball(radius)

        # Perform binary dilation
        dilated_volume = binary_dilation(volume, structure=struct_elem)

        return dilated_volume.astype(np.uint8)
    
    def create_float_ball(self, radius, size = None):
        if size is None:
            size = int(np.ceil(radius * 2)) + 1
        center = (np.array([size, size, size]) - 1) / 2
        zz, yy, xx = np.indices((size, size, size))
        distance = np.sqrt((xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2)
        ball = (distance <= radius).astype(np.uint8)
        return ball


    def variable_spherical_dilation(self, volume, noise_scale=0.1):
        start_time = time.time()

        # Label connected skeletons
        labeled_volume, num_features = label(volume, structure=np.ones((3, 3, 3)))
        label_time = time.time() - start_time

        # Precompute slices for each skeleton
        slices = find_objects(labeled_volume)

        dilated_volume = np.zeros_like(volume)
        grid_shape = volume.shape

        perlin_noise_time = 0
        argwhere_time = 0
        boundary_check_time = 0
        dilation_time = 0
        interp_time = 0
        ball_time = 0

        #print(f"Setup time: {label_time}, number of slices: {len(slices)}")

        mod_size = max(num_features // 100, 1)

        # Sort the width distribution once for efficient lookup
        widths = np.array([2.0, 2.82842712, 3.46410162, 4.0, 4.47213595, 4.89897949])
        size = 6
        balls = [self.create_float_ball(r/2, size) for r in widths]
        # Precompute CDF once at the top
        counts = np.array([51964, 48412, 27655, 11158, 6832, 134])
        probabilities = counts / counts.sum()
        cdf = np.cumsum(probabilities)

        # === Precompute Perlin noise empirical CDF mapping ===
        #print("Precomputing Perlin noise CDF...")
        perlin_samples = 10000
        perlin_noise_samples = np.array([pnoise3(i * noise_scale, i * noise_scale, i * noise_scale) for i in range(perlin_samples)])
        perlin_noise_normalized = (perlin_noise_samples + 1) / 2  # [0, 1]
        # Rank-based mapping
        ranks = rankdata(perlin_noise_normalized, method="average")
        perlin_cdf_map = ranks / perlin_samples
        # Sorted noise samples for lookup
        perlin_sorted = np.sort(perlin_noise_normalized)
        #print("Perlin noise CDF ready.")

        ball_offsets = [np.array(np.where(b)).T - size // 2 for b in balls]

        # Get all (x, y, z) you will loop over
        all_coords = np.vstack([coords for skeleton_label, skeleton_slice in enumerate(slices, start=1)
                                for coords in (np.argwhere(labeled_volume[skeleton_slice] == skeleton_label) +
                                            np.array([s.start for s in skeleton_slice]))])

        # Now compute Perlin noise just for those
        perlin_vals = np.array([
            pnoise3(x * noise_scale, y * noise_scale, z * noise_scale)
            for x, y, z in all_coords
        ])

        coord_to_noise = {(x, y, z): val for (x, y, z), val in zip(all_coords, perlin_vals)}


        # Iterate over each skeleton
        for skeleton_label, skeleton_slice in enumerate(slices, start=1):
            #if skeleton_label % mod_size == 0:
                ##print(f"Processing skeleton {skeleton_label // mod_size} out of {num_features // mod_size}")

            sub_volume = labeled_volume[skeleton_slice]

            argwhere_start = time.time()
            coords_in_sub = np.argwhere(sub_volume == skeleton_label)
            argwhere_time += time.time() - argwhere_start

            offset = np.array([s.start for s in skeleton_slice])
            coords = coords_in_sub + offset

            for (x, y, z) in coords:
                perlin_noise_start = time.time()
                noise_val = coord_to_noise[(x, y, z)]
                t = (noise_val + 1) / 2  # Normalize to [0, 1]
                idx = np.searchsorted(perlin_sorted, t)
                if idx >= len(perlin_cdf_map):
                    idx = len(perlin_cdf_map) - 1
                t_uniform = perlin_cdf_map[idx]

                perlin_noise_time += time.time() - perlin_noise_start

                # Map noise value through the empirical width distribution
                interp_start = time.time()
                index = bisect.bisect_left(cdf, t_uniform)
                interp_time += time.time() - interp_start
                
                ball_start = time.time()
                offsets = ball_offsets[index]
                bx, by, bz = (offsets + [x, y, z]).T
                ball_time += time.time() - ball_start

                boundary_check_start = time.time()
                valid = (bx >= 0) & (bx < grid_shape[0]) & \
                        (by >= 0) & (by < grid_shape[1]) & \
                        (bz >= 0) & (bz < grid_shape[2])
                boundary_check_time += time.time() - boundary_check_start

                dilation_start = time.time()
                dilated_volume[bx[valid], by[valid], bz[valid]] = 1
                dilation_time += time.time() - dilation_start

        #print times:
        # print(f"Perlin noise generation took {perlin_noise_time:.2f} seconds.")
        # print(f"Argwhere took {argwhere_time:.2f} seconds.")
        # print(f"Boundary check took {boundary_check_time:.2f} seconds.")
        # print(f"Dilation took {dilation_time:.2f} seconds.")
        # print(f"Interpolation took {interp_time:.2f} seconds.")
        # print(f"Ball creation took {ball_time:.2f} seconds.")

        return dilated_volume
    
    

    def remove_close_by_dilation(self, volume, holes, dilation_radius=3):

        # Step 1: Label the original structures
        original_labels, num_features = label(volume, structure=np.ones((3, 3, 3)))
        # print(f"Original structures found: {num_features}")

        # Step 2: Dilate holes
        dilated_holes = binary_dilation(holes, structure=np.ones((3, 3, 3)), iterations=dilation_radius)

        # Step 3: Precompute structure coordinates
        # print("Precomputing coordinates...")
        structure_first_coords = {}
        structure_coords = defaultdict(list)
        fully_eaten_structures = set()

        it = np.nditer(original_labels, flags=['multi_index'])
        for value in it:
            idx = value.item()
            if idx > 0:
                coord = it.multi_index
                structure_coords[idx].append(coord)
                if idx not in structure_first_coords:
                    structure_first_coords[idx] = coord

        # Step 4: Check if structures are fully eaten by holes
        for label_idx, coords in structure_coords.items():
            if all(dilated_holes[x, y, z] for x, y, z in coords):
                fully_eaten_structures.add(label_idx)

        # print(f"Found {len(fully_eaten_structures)} fully eaten structures.")
        # print(f"Precomputing done. Time: {time.time() - start_time:.2f}s")

        # Step 5: Dilate the volume
        dilated_volume = binary_dilation(volume, structure=np.ones((3, 3, 3)), iterations=dilation_radius)
        dilated_volume[dilated_holes > 0] = 0

        # Step 6: Label the dilated volume
        dilated_labels, num_dilated = label(dilated_volume, structure=np.ones((3, 3, 3)))
        # print(f"Dilated merged structures: {num_dilated}")

        kept_regions = set()
        cleaned_volume = volume.copy()

        # Step 7: Remove overlapping structures, except fully eaten ones
        for label_idx in range(1, num_features + 1):
            #if label_idx % 100 == 1 or label_idx == num_features:
                #print(f"Processing structure {label_idx} out of {num_features}, time: {time.time() - start_time:.2f}s")

            if label_idx in fully_eaten_structures:
                # Always keep fully eaten structures
                continue

            x, y, z = structure_first_coords[label_idx]
            dilated_region = dilated_labels[x, y, z]

            if dilated_region in kept_regions:
                # Remove structure directly via coordinates
                for cx, cy, cz in structure_coords[label_idx]:
                    cleaned_volume[cx, cy, cz] = 0
            else:
                kept_regions.add(dilated_region)

        # print(f"Kept {total_kept} structures, removed {num_features - total_kept}.")
        # print(f"Total time: {time.time() - start_time:.2f}s")
        return cleaned_volume
    



    def find_short_skeletons_as_holes(self, skeleton, holes):
        start_time = time.time()
        
        structure = np.ones((3, 3, 3))
        
        # Label holes and skeletons
        labeled_holes, num_holes = label(holes, structure=structure)
        labeled_skeleton, num_skeletons = label(skeleton, structure=structure)
        #print(f"Labeling done in {time.time() - start_time:.2f}s.")
        #print(f"Found {num_skeletons} skeletons and {num_holes} holes.")

        hole_addition = np.zeros_like(skeleton, dtype=np.uint8)
        
        objects_slices = find_objects(labeled_skeleton)
        loop_start = time.time()

        skeletons_under_length = 0
        
        for label_idx, obj_slice in enumerate(objects_slices, start=1):
            #if label_idx % 1000 == 1 or label_idx == num_skeletons:
                ##print(f"Processing skeleton {label_idx}/{num_skeletons} - time: {time.time() - loop_start:.2f}s")

            sub_skeleton = (labeled_skeleton[obj_slice] == label_idx)
            coords = np.argwhere(sub_skeleton)
            length = len(coords)
            
            if length >= 3:
                continue  # Skip long skeletons

            skeletons_under_length += 1

            sub_labeled_holes = labeled_holes[obj_slice]
            neighbor_hole_labels = set()

            for (sx, sy, sz) in coords:
                #print
                # Check 26 neighbors of each voxel for hole labels
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == dy == dz == 0:
                                continue
                            nx, ny, nz = sx + dx, sy + dy, sz + dz
                            if (
                                0 <= nx < sub_labeled_holes.shape[0] and
                                0 <= ny < sub_labeled_holes.shape[1] and
                                0 <= nz < sub_labeled_holes.shape[2]
                            ):
                                neighbor_label = sub_labeled_holes[nx, ny, nz]
                                if neighbor_label > 0:
                                    neighbor_hole_labels.add(neighbor_label)

            if len(neighbor_hole_labels) >= 1:
                # Add skeleton voxels as holes
                for offset in coords:
                    x, y, z = offset + np.array([s.start for s in obj_slice])
                    hole_addition[x, y, z] = 1

        total_time = time.time() - start_time
        # #print(f"Completed in {total_time:.2f}s. Added {np.sum(hole_addition)} new hole voxels. Skeletons under length: {skeletons_under_length}")

        return hole_addition











def generate_skeleton(shape, gap_threshold, gap_chance=0.3, num_lines=15, wobble=1.5):
    """Wrapper function that creates a 3D artificial dataset with skeleton-like structures."""
    size_for_pow = np.cbrt(shape[0]*shape[1]*shape[2]).astype(np.float32) / 100
    # #print (f"Size for pow: {size_for_pow}")
    generator = ArtificialSkeletonGeneratorExtended(shape=shape, wobble=wobble) # 0.000005 times shape^3
    skeleton_data, broken_skeleton, long_holes = generator.generate_broken_skeleton_data(num_lines=np.round(num_lines*pow(size_for_pow,3)).astype(np.int32), gap_threshold=gap_threshold, gap_min=1, gap_chance=gap_chance)
    
    # dilate long holes:
    long_holes = binary_dilation(long_holes, structure=np.ones((3, 3, 3)), iterations=1)
    
    return skeleton_data, broken_skeleton, long_holes

def generate_skeleton_with_thickness(shape, gap_threshold, gap_chance=0.3, num_lines=15, wobble=1.5):
    """Wrapper function that creates a 3D artificial dataset with skeleton-like structures."""
    size_for_pow = np.cbrt(shape[0]*shape[1]*shape[2]).astype(np.float32) / 100
    # #print (f"Size for pow: {size_for_pow}")
    generator = ArtificialSkeletonGeneratorExtended(shape=shape, wobble=wobble) # 0.000005 times shape^3
    skeleton_data, broken_skeleton, long_holes = generator.generate_broken_skeleton_data(num_lines=np.round(num_lines*pow(size_for_pow,3)).astype(np.int32), gap_threshold=gap_threshold, gap_min=3, gap_chance=gap_chance)
    
    # generator = ArtificialSkeletonGeneratorExtended(shape=shape) # 0.000005 times shape^3
    # skeleton_data, broken_skeleton = generator.generate_broken_skeleton_data(total_sum=500*512*8, num_lines=20*512*8) # For 64: 500, 20
    # skeleton_data = skeleton_data
    # skeleton_data = generator.spherical_dilation(skeleton_data, radius=3)
    
    # dilation_time = time.time()
    skeleton_data_dilated = generator.variable_spherical_dilation(skeleton_data, noise_scale=0.002)
    all_skeleton_short = generator.variable_spherical_dilation(skeleton_data+broken_skeleton, noise_scale=0.002)
    all_skeleton_long = generator.variable_spherical_dilation(skeleton_data+long_holes, noise_scale=0.002)
    broken_skeleton_dilated = all_skeleton_short - skeleton_data_dilated
    long_holes_dilated = all_skeleton_long - skeleton_data_dilated
    long_holes_dilated = binary_dilation(long_holes_dilated, structure=np.ones((3, 3, 3)), iterations=1)
    #print (f"Dilation time: {time.time()-dilation_time}")
    
    
    # skeleton_data_dilated = skeleton_data_dilated[np.newaxis, ...]  # Adds a new channel dimension at axis 0
    # broken_skeleton_dilated = broken_skeleton_dilated[np.newaxis, ...].astype(np.uint8)  # Same for broken skeleton
    # skeleton_data = skeleton_data[np.newaxis, ...]
    # hole_addtion = hole_addtion[np.newaxis, ...]

    #print (f"Time taken: {time.time()-startTime}")
    
    return skeleton_data_dilated, broken_skeleton_dilated, long_holes_dilated



