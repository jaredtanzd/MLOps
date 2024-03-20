import os
import numpy as np
from PIL import Image

def ensure_scale_to_1(image):
    """Ensure the image is in the 0-1 scale."""
    if np.max(image) > 1:
        image = image / 255.0
    return image

def process_directory_flat_structure(src_base, tgt_base):
    """Process all .tiff files in the source base directory, ensuring they are in the 0-1 range,
    and save them in the corresponding train, test, or val directories with filenames formatted
    as 'KCLXXXX_Y.tiff', where Y has no leading zeros."""
    if not os.path.exists(tgt_base):
        os.makedirs(tgt_base, exist_ok=True)

    for root, dirs, files in os.walk(src_base):
        for filename in files:
            if filename.endswith('.tiff'):
                patient_number, slice_number = filename.split('_')
                patient_number = patient_number.replace('KCL', '')
                slice_number = slice_number.replace('.tiff', '')
                
                try:
                    patient_number_int = int(patient_number)
                    slice_number_int = int(slice_number)
                except ValueError:
                    continue  # Skip files that don't match expected format
                
                if 1 <= patient_number_int <= 20:
                    sub_dir = 'train'
                elif 21 <= patient_number_int <= 30:
                    sub_dir = 'test'
                elif 31 <= patient_number_int <= 37:
                    sub_dir = 'val'
                else:
                    continue  # Skip files outside the specified ranges
                
                target_dir = os.path.join(tgt_base, sub_dir)
                os.makedirs(target_dir, exist_ok=True)
                
                img_path = os.path.join(root, filename)
                img = Image.open(img_path)
                img_array = np.array(img)
                scaled_img = ensure_scale_to_1(img_array)
                img_to_save = Image.fromarray((scaled_img * 255).astype(np.uint8))
                output_filename = f"KCL{patient_number}_{slice_number_int}.tiff"
                img_to_save.save(os.path.join(target_dir, output_filename), format='TIFF')
                print(f"Saved: {os.path.join(target_dir, output_filename)}")

if __name__ == "__main__":
    src_base = 'data/processed/brain_extracted'
    tgt_base = 'data/processed/sorted_imgs'
    process_directory_flat_structure(src_base, tgt_base)
