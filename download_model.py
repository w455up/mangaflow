from simple_lama_inpainting import SimpleLama
import os

print("Pre-downloading LaMa model...")
try:
    _ = SimpleLama()
    print("Model downloaded successfully.")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)
