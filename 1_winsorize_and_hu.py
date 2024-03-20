import pydicom
import numpy as np
import sys
import os
import re  # For regex matching

def process_dicom_file(dicom_path, output_path):
    raw_dcm = pydicom.dcmread(dicom_path)
    # Winsorize
    raw_data = raw_dcm.pixel_array.copy()
    raw_data[raw_data == -2000] = 0
    # Convert to Hounsfield Units
    hu_data = (raw_data * raw_dcm.RescaleSlope) + raw_dcm.RescaleIntercept
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the processed data
    np.save(output_path, hu_data)

def winsorize_and_convert_to_hu(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            if re.match(r'KCL_\d{4}$', dir_name):
                full_path = os.path.join(root, dir_name)
                # Process DICOM files in each matching directory
                for root_inner, dirs_inner, files_inner in os.walk(full_path):
                    for file in files_inner:
                        if file.endswith(".dcm"):
                            dicom_path = os.path.join(root_inner, file)
                            relative_path = os.path.relpath(root_inner, input_dir)
                            output_path = os.path.join(output_dir, relative_path, file.replace('.dcm', '.npy'))
                            process_dicom_file(dicom_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    winsorize_and_convert_to_hu(input_dir, output_dir)
