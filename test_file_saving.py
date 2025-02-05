import os

# Define the directory and file path
save_dir = "Bach/model_checkpoints"
save_path = os.path.join(save_dir, "test_file.txt")

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Write to the file
with open(save_path, "w") as f:
    f.write("This is a test file.\n")

# Verify if the file exists
if os.path.exists(save_path):
    print(f"✅ File successfully saved at: {save_path}")
else:
    print("❌ File was NOT saved correctly!")

# Read the file to check content
with open(save_path, "r") as f:
    content = f.read()
    print("File content:", content)
