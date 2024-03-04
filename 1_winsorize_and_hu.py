import pydicom
import numpy as np
import sys
import os

def process_dicom_file(dicom_path, output_path):
    raw_dcm = pydicom.dcmread(dicom_path)
    wl = 40  # Default value
    ww = 80  # Default value

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
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.relpath(root, input_dir), file.replace('.dcm', '.npy'))
                process_dicom_file(dicom_path, output_path)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    winsorize_and_convert_to_hu(input_dir, output_dir)

