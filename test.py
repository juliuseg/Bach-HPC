import numpy as np
from scipy.stats import rv_discrete


# Load the saved array
loaded_data = np.load("Bach/component_sizes.npy")

# Split into unique_sizes and counts
unique_sizes, counts = loaded_data[:, 0], loaded_data[:, 1]
probabilities = counts / counts.sum()  # Normalize to create probabilities

# Define a discrete random variable based on observed sizes
size_distribution = rv_discrete(name="component_size_dist", values=(unique_sizes, probabilities))

# Print to verify
print("✅ Loaded component sizes:")
for size, count in zip(unique_sizes, counts):
    print(f"Component size: {size}, Count: {count}")

random_sample = size_distribution.rvs(size=1)
print(f"Randomly sampled component size: {random_sample}")