import numpy as np
from skimage import io
import os
import random
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import shutil  # For copying files
import argparse

def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def sample_and_transfer_images(source_dir, target_dir, sample_size=100):
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    clear_directory(target_dir)

    filenames = [filename for filename in os.listdir(source_dir) if filename.endswith('.tiff')]
    sampled_filenames = random.sample(filenames, min(sample_size, len(filenames)))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in sampled_filenames:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
        
    print(f"Transferred {len(sampled_filenames)} images from {source_dir} to {target_dir}.")

def load_images_from_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.tiff'):
            img_path = os.path.join(directory_path, filename)
            img = io.imread(img_path).astype(np.float32) / 255.0
            images.append(img.flatten())
            
    if images:
        return np.array(images)
    else:
        print(f"No images found in {directory_path}.")
        return None

def apply_dimensionality_reduction(data, n_components=None):
    if data is not None:
        if n_components is None or n_components > min(data.shape):
            n_components = min(data.shape)
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)
    else:
        return None

def detect_drift(current_data, new_data, alpha=0.05):
    if current_data is not None and new_data is not None:
        ks_results = [ks_2samp(current_data[:, i], new_data[:, i]).pvalue for i in range(current_data.shape[1])]
        reject_list, corrected_p_values, _, _ = multipletests(ks_results, alpha=alpha, method='bonferroni')
        return reject_list, corrected_p_values
    else:
        return None, None

def main(old_data_dir, new_data_dir):
    temp_current_data_dir = 'temp_current_data'
    temp_new_data_dir = 'temp_new_data'
    
    sample_and_transfer_images(old_data_dir, temp_current_data_dir)
    sample_and_transfer_images(new_data_dir, temp_new_data_dir)

    current_data = load_images_from_directory(temp_current_data_dir)
    new_data = load_images_from_directory(temp_new_data_dir)

    if current_data is not None and new_data is not None:
        current_data_reduced = apply_dimensionality_reduction(current_data)
        new_data_reduced = apply_dimensionality_reduction(new_data)

        reject_list, _ = detect_drift(current_data_reduced, new_data_reduced)

        drift_detected = any(reject_list)
        print("Drift detected:" if drift_detected else "No significant drift detected.")
        print(f"Number of features with drift detected: {np.sum(reject_list)} / {len(reject_list)}")
        
        return drift_detected
    else:
        print("Image loading failed. Drift detection skipped.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect data drift between old and new image datasets.")
    parser.add_argument('--old', type=str, required=True, help="Path to the old data directory.")
    parser.add_argument('--new', type=str, required=True, help="Path to the new data directory.")
    args = parser.parse_args()

    drift_detected = main(args.old, args.new)
    
    # Output for CML to consume
    print(f"::set-output name=drift-detected::{str(drift_detected).lower()}")
