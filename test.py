import os

# Define the target directory and file name
directory = "/work3/s204427"
file_name = "example.txt"

# Check if the directory exists
if os.path.exists(directory):
    print(f"Listing files in {directory}:")
    print(os.listdir(directory))  # List files in the directory
else:
    print(f"Directory {directory} does not exist or is not accessible.")

# Define the full path to the file
file_path = os.path.join(directory, file_name)

# Attempt to write to the file
try:
    with open(file_path, "w") as file:
        file.write("Hello, this is a simple text file saved in /work3/s204427.")
    print(f"File saved successfully at: {file_path}")
except PermissionError:
    print(f"Permission denied: Cannot write to {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
