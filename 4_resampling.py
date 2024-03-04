import numpy as np
import os
import sys
from scipy.ndimage import zoom

def load_volume_from_slices(input_dir):
    slices = [np.load(os.path.join(input_dir, f)) for f in sorted(os.listdir(input_dir)) if f.endswith(".npy")]
    return np.stack(slices, axis=0)

def resample_volume(volume, original_spacing, target_spacing):
    resampling_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(volume.shape * resampling_factor).astype(int)
    resampled_volume = zoom(volume, resampling_factor, order=1)
    return resampled_volume

def process_volume(input_dir, output_dir, original_spacing, target_spacing):
    os.makedirs(output_dir, exist_ok=True)
    volume = load_volume_from_slices(input_dir)
    resampled_volume = resample_volume(volume, original_spacing, target_spacing)
    # Save each slice of the resampled volume as a separate .npy file
    for i, slice in enumerate(resampled_volume):
        output_path = os.path.join(output_dir, f"slice_{i:04d}.npy")
        np.save(output_path, slice)
        print(f"Slice {i} resampled and saved to {output_path}")

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    original_spacing = tuple(map(float, sys.argv[3:6]))  # Expecting three spacing values: depth, height, width
    target_spacing = tuple(map(float, sys.argv[6:9]))  # Similarly for target spacing
    process_volume(input_dir, output_dir, original_spacing, target_spacing)
