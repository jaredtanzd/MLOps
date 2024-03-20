import os
import numpy as np
import nibabel as nib
from PIL import Image

def normalize(volume):
    """Normalize the volume to 0-1 scale"""
    min_val = np.min(volume)
    max_val = np.max(volume)
    return (volume - min_val) / (max_val - min_val)

def generate_slices_flat_structure_tiff_and_rotate(src_dir, base_tgt_dir):
    """
    Generate 2D slices from 3D NIfTI grey matter masks, rotate them by 90 degrees,
    and place them into train, test, or val directories based on the patient ID.
    The output filenames are formatted as 'KCLXXXX_Y.tiff', maintaining the 0-1 normalization.
    """
    os.makedirs(base_tgt_dir, exist_ok=True)
    
    for filename in os.listdir(src_dir):
        if filename.endswith(".nii.gz"):
            id = filename.split("_")[0].replace("sub-", "")
            id_num = int(id)
            filepath = os.path.join(src_dir, filename)
            
            if 1 <= id_num <= 20:
                sub_dir = "train"
            elif 21 <= id_num <= 30:
                sub_dir = "test"
            elif 31 <= id_num <= 37:
                sub_dir = "val"
            else:
                continue

            tgt_dir = os.path.join(base_tgt_dir, sub_dir)
            os.makedirs(tgt_dir, exist_ok=True)

            nifti = nib.load(filepath)
            volume_norm = normalize(nifti.get_fdata())
            
            for i in range(volume_norm.shape[-1]):
                slice = volume_norm[:, :, i]
                slice_image = Image.fromarray(slice.astype(np.float32))
                rotated_slice = slice_image.rotate(90, expand=True)  # Rotate by 90 degrees
                
                slice_filename = f"KCL{id}_{i+1}.tiff"
                rotated_slice.save(os.path.join(tgt_dir, slice_filename), format='TIFF')
                print(f"Saved: {os.path.join(tgt_dir, slice_filename)}")

if __name__ == "__main__":
    src_dir = "./GM"  # Your source directory for NIfTI files
    base_tgt_dir = "./data/processed/sorted_masks"  # Your base target directory for TIFF files
    generate_slices_flat_structure_tiff_and_rotate(src_dir, base_tgt_dir)
