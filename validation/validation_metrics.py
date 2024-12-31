import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
import concurrent.futures

# Image Preprocessing
def preprocess_image(img_path, target_size=(299, 299)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1] range as required by InceptionV3
    return img  # Don't add a new axis here, keep shape as (299, 299, 3)

# Feature Extraction using InceptionV3
def extract_features(images, batch_size=64):
    # Load InceptionV3 model pre-trained on ImageNet
    inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_features = inception_model.predict(np.array(batch))
        features.append(batch_features)
    
    return np.vstack(features)

# Dimensionality Reduction using PCA (Optional)
def reduce_dimensions(features, n_components=128):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

# FID Calculation
def calculate_fid(original_features, augmented_features):
    # Compute the mean and covariance of the original and augmented feature sets
    mu1, sigma1 = np.mean(original_features, axis=0), np.cov(original_features, rowvar=False)
    mu2, sigma2 = np.mean(augmented_features, axis=0), np.cov(augmented_features, rowvar=False)
    
    # Compute the squared difference in means
    diff = mu1 - mu2
    diff_squared = np.sum(diff ** 2)
    
    # Compute the square root of the product of covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Numerical stability check for imaginary values (it may happen due to floating-point errors)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Compute FID score
    fid = diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Parallelize the Feature Extraction for Efficiency (Optional)
def parallel_extract_features(images, batch_size=64, num_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(extract_features, images[i:i+batch_size], batch_size) for i in range(0, len(images), batch_size)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return np.vstack(results)

# Main Validation Process
def validate_images(original_dir, augmented_folders):
    original_images = [os.path.join(original_dir, fname) for fname in os.listdir(original_dir)]
    
    augmented_images = []
    for folder in augmented_folders:
        augmented_images.extend([os.path.join(folder, fname) for fname in os.listdir(folder)])
    
    # Preprocess images
    original_images_preprocessed = [preprocess_image(img) for img in original_images]
    augmented_images_preprocessed = [preprocess_image(img) for img in augmented_images]

    # Extract features (using parallelization if desired)
    original_features = parallel_extract_features(original_images_preprocessed)  # Or use extract_features(original_images_preprocessed)
    augmented_features = parallel_extract_features(augmented_images_preprocessed)  # Or use extract_features(augmented_images_preprocessed)

    # Optional: Apply PCA for dimensionality reduction
    original_features = reduce_dimensions(original_features)
    augmented_features = reduce_dimensions(augmented_features)

    # Calculate FID score
    fid_score = calculate_fid(original_features, augmented_features)
    print(f"FID Score: {fid_score}")
    return fid_score

# Example Usage
if __name__ == "__main__":
    original_dir = "C:/augdata-valid/Test"  # Path to the original images folder
    augmented_folders = [
        "C:/augdata-valid/images/blended",
        "C:/augdata-valid/images/color_adjusted",
        "C:/augdata-valid/images/flipped",
        "C:/augdata-valid/images/noisy",
        "C:/augdata-valid/images/rotated"
    ]  # List of augmented images folders
    
    fid_score = validate_images(original_dir, augmented_folders)
    print(f"FID Score for the augmented images: {fid_score}")
