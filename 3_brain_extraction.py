import numpy as np
import os
import sys
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

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            extract_brain(input_file, output_file)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    process_directory(input_dir, output_dir)

