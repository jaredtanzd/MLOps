import numpy as np
import sys
import os
import re

def apply_windowing(input_file, output_file, window_center, window_width):
    img = np.load(input_file)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    np.save(output_file, img)

def process_directory(input_dir, output_dir, window_center, window_width):
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            if re.match(r'KCL_\d{4}$', dir_name):
                full_dir_path = os.path.join(root, dir_name)
                output_dir_path = os.path.join(output_dir, os.path.relpath(full_dir_path, input_dir))
                process_npy_files(full_dir_path, output_dir_path, window_center, window_width)

def process_npy_files(input_dir, output_dir, window_center, window_width):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            apply_windowing(input_file, output_file, window_center, window_width)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_dir> <output_dir> <window_center> <window_width>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    window_center = int(sys.argv[3])
    window_width = int(sys.argv[4])
    process_directory(input_dir, output_dir, window_center, window_width)
