# Project setup guide

This guide outlines the initial setup process for a medical image analysis project, focusing on version control, data storage, and dataset management using Git and DVC (Data Version Control). It also details the process for setting up an AWS S3 bucket for remote data storage.

## Initial Setup

### Prerequisites

- Git
- DVC
- AWS Account

### Steps

1. **Initialize DVC and Git**: Start by setting up DVC and Git in your project directory to enable version control for your datasets, models, and code.
    ```
    dvc init
    git init  
    ```

2. **Create an AWS S3 Bucket**: Use the AWS Management Console to create an S3 bucket for storing your datasets and models. Choose a unique name for your bucket and select a region close to your server for optimized access speeds. Ensure your bucket has the correct security settings for your project's needs.

3. **Create an IAM User**: Generate an IAM User in AWS with read and write access to your S3 bucket. Attach the `AmazonS3FullAccess` policy to this user. Securely store the provided Access Key ID and Secret Access Key.

### Versioning Datasets and Models

1. **Add Your Data to DVC**:
To version control your DICOM datasets and GM masks, use the following commands:

    ```
    dvc add <data_folder>
    git add <data_folder>.dvc
    git commit -m "Tracking <data_folder> with DVC"
    ```
DVC calculates a MD5 checksum for each file, moving the data to `.dvc/cache`. A pointer file is created in Git, allowing for dataset version control alongside your code.

2. **Configure DVC for S3 Storage**:
Set up DVC to use your S3 bucket for remote data storage with the following commands:

    ```
    dvc remote add -d myremote s3://mybucket/data
    dvc remote modify myremote access_key_id <your-access-key-id>
    dvc remote modify myremote secret_access_key <your-secret-access-key>
    ```

Adjust the bucket path as needed. The `-d` flag sets this as the default remote storage.

3. **Push Data to S3**:
After configuring your AWS credentials (using `aws configure`), upload your data to S3 with `dvc push`.

### Sharing and Version Control

1. **Share Remote Storage Configuration**:
Commit your DVC configuration changes to Git to share the remote storage setup with collaborators:


    ```
    git add .dvc/config
    git commit -m "Configured AWS S3 as DVC remote storage"
    git push
    ```

2. **Rolling Back to Previous Versions**:
To revert to earlier versions of your data, use Git and DVC commands:

    ```
    git checkout <commit_sha> <data_folder>.dvc
    dvc checkout
    ```


## Preprocessing Pipeline

The project includes Python scripts for data preprocessing:

- `winsorize_and_hu.py`: Winsorize and convert to Hounsfield Units (HU).
- `windowing.py`: Apply windowing to enhance image features.
- `brain_extraction.py`: Extract the brain from MRI scans.
- `sort_images.py`: Sort images into structured directories.
- `sort_images_no_extraction.py`: Sort images without extraction.
- `process_and_sort_masks.py`: Process and sort NIfTI masks.

### Defining DVC Pipeline

The `dvc.yml` file defines each stage of the preprocessing pipeline, including commands (`cmd`), dependencies (`deps`), and outputs (`outs`). 

### Executing the Pipeline

To execute the defined DVC pipeline, run:
  ```
  dvc repro
  ```

DVC checks for changes in each stage's dependencies and reruns affected stages.

## Tracking Changes with Git

To share and version your pipeline and code, use Git:
  ```
  git add .
  git commit -m "Add preprocessing pipeline"
  git push
  ```

To pull the latest changes and reproduce the pipeline:

  ```
  git pull
  dvc pull
  dvc repro
  ```

## CML Workflow Overview

The CML workflow, triggered by push events, encompasses several stages, including:

1. **Deploying a CML Runner on AWS**: Utilizes an EC2 `p2.xlarge` GPU-equipped instance for intensive computations.

2. **Data Drift Detection**: Evaluates the need for retraining the model by comparing old and new datasets.

3. **Model Retraining**: Conditionally executed based on the drift detection results, utilizing DVC for data versioning and pipeline reproducibility.

4. **Report Publishing**: Summarizes the training outcomes and updates the model in the DVC remote storage, with results communicated through GitHub comments for collaborative review.

## Key Components and Scripts

- **Data Preparation**: Involves preprocessing tasks such as normalization, windowing, and brain extraction, handled by Python scripts referenced in the DVC pipeline.

- **Drift Detection**: A script evaluates changes in data distributions, employing PCA for dimensionality reduction and the Kolmogorov-Smirnov test with Bonferroni correction for statistical analysis.

- **Model Training**: Incorporates early stopping, model layer freezing, and training logs, aiming to fine-tune the U-Net model based on detected data drifts.
