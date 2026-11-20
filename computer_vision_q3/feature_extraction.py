# File: feature_extraction.py
import os
import cv2
import numpy as np
import glob
from skimage.feature import hog

# Define the classes in a specific order to ensure consistent labeling (0, 1, 2)
CLASSES = ['hole', 'paper' ,'penMark']

def extract_features_from_image(img_rgb):
    # Extracts Color Histogram, Spatial Binning, and HOG from an RGB image.
    # Resize to a standard size (64x64) for consistent feature dimensions
    img_resized = cv2.resize(img_rgb, (64, 64))
    
    # 1. Color Histogram (32 bins per channel)
    hist_features = []
    for i in range(3):
        hist = np.histogram(img_resized[:, :, i], bins=32, range=(0, 256))[0]
        hist_features.append(hist)
    color_hist = np.concatenate(hist_features)
    
    # 2. Spatial Binning (16x16 grid, flattened)
    spatial_features = cv2.resize(img_resized, (16, 16)).ravel()
    
    # 3. Histogram of Oriented Gradients (HOG)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    hog_features = hog(
        img_gray, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=False, 
        feature_vector=True
    )
    
    # Concatenate all features into a single 1D array
    return np.concatenate((color_hist, spatial_features, hog_features))

def load_dataset(dataset_dir='organized_dataset'):
    
    #Loads images from organized directories and extracts features.Returns X_train, y_train, X_val, y_val, X_test, y_test
    splits = ['train', 'val', 'test']
    
    # Dictionary to hold our data
    data = {split: {'X': [], 'y': []} for split in splits}
    
    for split in splits:
        print(f" Processing {split} set...")
        split_dir = os.path.join(dataset_dir, split)
        
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(split_dir, class_name)
            # Find all image files in this class directory
            image_paths = glob.glob(os.path.join(class_dir, '*.*'))
            
            for img_path in image_paths:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"   Warning: Could not read {img_path}. Skipping.")
                    continue
                    
                # Convert BGR to RGB for consistent processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract features
                features = extract_features_from_image(img_rgb)
                
                data[split]['X'].append(features)
                data[split]['y'].append(class_idx)
                
        # Convert lists to numpy arrays
        data[split]['X'] = np.array(data[split]['X'])
        data[split]['y'] = np.array(data[split]['y'])
        print(f"  {split.capitalize()} set: X shape = {data[split]['X'].shape}, y shape = {data[split]['y'].shape}")
        
    return (
        data['train']['X'], data['train']['y'],
        data['val']['X'], data['val']['y'],
        data['test']['X'], data['test']['y']
    )

if __name__ == "__main__":
    print(" Starting feature extraction pipeline...\n")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('organized_dataset')
    print("\n Feature extraction completed successfully!")
    print(f"Total features per image: {X_train.shape[1]}")