# This script downloads the Qwen2.5-Math-1.5B model from Hugging Face to the specified path.
# Requirements: Install huggingface_hub if not already installed (pip install huggingface_hub)

from huggingface_hub import snapshot_download
import os

# Model repository ID on Hugging Face
repo_id = "Qwen/Qwen2.5-Math-1.5B"  # You can change to "Qwen/Qwen2.5-Math-1.5B-Instruct" if needed

# Specified local directory to save the model
local_dir = "/kun-data/assignment5-alignment/models"
model_dir = f"{local_dir}/{repo_id}"
# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Download the model snapshot
print(f"Downloading model from {repo_id} to {model_dir}...")
snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False)

print("Download completed!")