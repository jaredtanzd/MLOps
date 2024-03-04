import numpy as np
import sys
import os

def apply_windowing(input_file, output_file, window_center, window_width):
    img = np.load(input_file)

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    np.save(output_file, img)

def process_directory(input_dir, output_dir, window_center, window_width):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            apply_windowing(input_file, output_file, window_center, window_width)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    window_center = int(sys.argv[3])
    window_width = int(sys.argv[4])
    process_directory(input_dir, output_dir, window_center, window_width)

