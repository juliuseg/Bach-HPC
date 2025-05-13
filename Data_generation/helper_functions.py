import numpy as np
from scipy.ndimage import map_coordinates
from skimage.draw import line_nd
from noise import pnoise1
from collections import defaultdict

def get_random_border_start(shape):
    side = np.random.choice(["y=max", "z=0"])
    if side == "y=max":
        return np.random.randint(0, shape[0]), shape[1] - 1, np.random.randint(0, shape[2])
    else:
        return np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), 0

def connect_voxels(start, end, art_data):
    rr, cc, zz = line_nd(start, end)
    art_data[rr, cc, zz] = 1
    return rr, cc, zz

def calculate_radius(center, start):
    return np.sqrt((start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2)

def get_next_point(center, current, step_size, clockwise=True):
    radius = calculate_radius(center, current)
    angle = np.arctan2(current[1] - center[1], current[0] - center[0])
    angle += -step_size if clockwise else step_size
    new_x = center[0] + radius * np.cos(angle)
    new_y = center[1] + radius * np.sin(angle)
    return (new_x, new_y)


def generate_line_task(args):
    shape, wobble_strengths, wobble_scales, seeds = args

    art_data = np.zeros(shape, dtype=np.uint8)
    all_line_indices = []

    for seed_x, seed_yz in seeds:
        x, y, z = get_random_border_start(shape)
        noise_step, prev_x, prev_y, prev_z = 0, x, y, z
        line_length = shape[0] * 2
        line_index = 0

        center = ((shape[1] - 1) * 3, (shape[2] - 1) * 2.4)
        start = (y, z)
        clockwise = True
        radius = calculate_radius(center, start)
        num_steps = int(2 * np.pi * radius) * 0.25
        step_size = 2 * np.pi / num_steps
        next_point = start
        line_indices = []

        while 0 <= y < shape[1] and 0 <= z < shape[2] and line_index < line_length:
            line_index += 1
            total_dx = total_dy = total_dz = 0
            for strength, scale in zip(wobble_strengths, wobble_scales):
                noise_x = pnoise1((noise_step + seed_x) * scale) * strength
                noise_yz = pnoise1((noise_step + seed_yz) * scale) * strength
                total_dx += noise_x / np.sqrt(2)
                total_dy += noise_yz / np.sqrt(2)
                total_dz += noise_yz / np.sqrt(2)

            dx, dy, dz = int(total_dx), int(total_dy), int(total_dz)
            next_point = get_next_point(center, next_point, step_size, clockwise)
            y, z = int(next_point[0]), int(next_point[1])

            y_set = np.clip(y + dy, 0, shape[1] - 1)
            z_set = np.clip(z + dz, 0, shape[2] - 1)
            x_set = np.clip(x + dx, 0, shape[0] - 1)

            if 0 <= y_set < shape[1]:
                if (x_set, y_set, z_set) != (prev_x, prev_y, prev_z):
                    rr, cc, zz = connect_voxels((prev_x, prev_y, prev_z), (x_set, y_set, z_set), art_data)
                    for r, c, z in zip(rr, cc, zz):
                        line_indices.append((r, c, z))
                        art_data[r, c, z] = 1

                art_data[x_set, y_set, z_set] = 1
                line_indices.append((x_set, y_set, z_set))
                prev_x, prev_y, prev_z = x_set, y_set, z_set

                if z_set >= shape[2] - 1:
                    break
            else:
                break

            noise_step += 1

        all_line_indices.append(line_indices)

    return art_data, all_line_indices



def process_label_chunk(chunk_indices, labels_shape, labels_flat):
    local_coords = defaultdict(list)
    local_first = {}

    for flat_idx in chunk_indices:
        idx = labels_flat[flat_idx]
        if idx == 0:
            continue

        # Convert flat index to (x, y, z)
        z = flat_idx % labels_shape[2]
        y = (flat_idx // labels_shape[2]) % labels_shape[1]
        x = flat_idx // (labels_shape[1] * labels_shape[2])
        coord = (x, y, z)

        local_coords[idx].append(coord)
        if idx not in local_first:
            local_first[idx] = coord

    return local_coords, local_first







    # def size_distribution(self):
    #     loaded_data = np.load("component_sizes.npy")

    #     # Split into unique_sizes and counts
    #     unique_sizes, counts = loaded_data[:, 0], loaded_data[:, 1]
    #     probabilities = counts / counts.sum()  # Normalize to create probabilities

    #     # Define a discrete random variable based on observed sizes
    #     size_distribution = rv_discrete(name="component_size_dist", values=(unique_sizes, probabilities))
    #     return size_distribution
        
    
    
    # def generate_skeleton_data(self, num_lines):
    #     num_cores = os.cpu_count()
    #     print(f"Using {num_cores} cores split across {num_lines} lines.")

    #     shape = self.shape
    #     wobble_strengths = self.wobble_strengths
    #     wobble_scales = self.wobble_scales

    #     # Make a big list of seed pairs
    #     all_seeds = [(np.random.randint(0, 100000), np.random.randint(0, 100000)) for _ in range(num_lines)]

    #     # Split seeds across cores
    #     chunk_size = len(all_seeds) // num_cores
    #     seed_chunks = [all_seeds[i:i + chunk_size] for i in range(0, len(all_seeds), chunk_size)]

    #     args_list = [(shape, wobble_strengths, wobble_scales, chunk) for chunk in seed_chunks]

    #     with Pool(processes=num_cores) as pool:
    #         results = pool.map(generate_line_task, args_list)

    #     # Merge the results
    #     art_data = np.zeros(self.shape, dtype=np.uint8)
    #     skeletons = []
    #     for partial_data, line_indices_list in results:
    #         art_data |= partial_data
    #         skeletons.extend(line_indices_list)

    #     return art_data, skeletons