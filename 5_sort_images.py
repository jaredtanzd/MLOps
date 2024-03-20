import os
import numpy as np
from PIL import Image

def ensure_scale_to_1(image):
    """Ensure the image is in the 0-1 scale."""
    if np.max(image) > 1:
        image = image / 255.0
    return image

def process_directory_flat_structure(src_base, tgt_base):
    """
    Process all NumPy files in the source base directory, ensuring they are in the 0-1 range,
    and save them in the corresponding train, test, or val directories with filenames formatted
    as 'KCLXXXX_Y.tiff', where Y has no leading zeros.
    """
    if not os.path.exists(tgt_base):
        os.makedirs(tgt_base, exist_ok=True)

    for root, dirs, files in os.walk(src_base):
        base_folder_name = os.path.basename(root)
        # Skip processing if the directory does not match the expected patient folder format
        if not base_folder_name.startswith('KCL_'):
            continue

        patient_number = base_folder_name.replace('KCL_', '')
        try:
            patient_number_int = int(patient_number)
        except ValueError:
            continue  # Skip directories that don't have an integer after 'KCL_'

        # Determine the target subdirectory based on patient number
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

        for filename in files:
            if filename.endswith('.npy'):
                img_data = np.load(os.path.join(root, filename))
                scaled_img_1 = ensure_scale_to_1(img_data)
                slice_number = int(filename.replace('IMG', '').split('.')[0])
                output_filename = f"KCL{patient_number}_{slice_number}.tiff"
                img_to_save = Image.fromarray(scaled_img_1.astype(np.float32))
                img_to_save.save(os.path.join(target_dir, output_filename), format='TIFF')
                print(f"Saved: {os.path.join(target_dir, output_filename)}")

if __name__ == "__main__":
    src_base = 'data/processed/brain_extracted'
    tgt_base = 'data/processed/sorted_imgs'
    process_directory_flat_structure(src_base, tgt_base)
