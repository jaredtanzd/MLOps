import numpy as np
import os
import sys
import re
from skimage import morphology
from scipy.ndimage import label, binary_fill_holes

def extract_brain(input_file, output_file):
    img = np.load(input_file)
    
    segmentation = morphology.dilation(img, np.ones((1, 1)))
    labels, label_nb = label(segmentation)
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0
    
    mask = labels == label_count.argmax()
    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    
    brain = img * mask
    np.save(output_file, brain)

def process_directory_recursive(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        # Check if the current directory matches the "KCL_XXXX" pattern or is a subdirectory of a matching directory
        if re.search(r'KCL_\d{4}', root):
            current_input_dir = root
            relative_path = os.path.relpath(current_input_dir, input_dir)
            current_output_dir = os.path.join(output_dir, relative_path)

            # Ensure the output directory exists
            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)

            # Process all .npy files in the current directory
            for file in files:
                if file.endswith(".npy"):
                    input_file = os.path.join(current_input_dir, file)
                    output_file = os.path.join(current_output_dir, file)
                    extract_brain(input_file, output_file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    process_directory_recursive(input_dir, output_dir)
