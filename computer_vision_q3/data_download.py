# File: data_downloads.py
import os
import shutil
import glob
from sklearn.model_selection import train_test_split

def organize_dataset(raw_dir, target_dir):
    print(f" Searching for images in '{raw_dir}'...")
    
    # Find all images recursively
    image_paths = (glob.glob(os.path.join(raw_dir, '**/*.jpg'), recursive=True) +
                   glob.glob(os.path.join(raw_dir, '**/*.png'), recursive=True) +
                   glob.glob(os.path.join(raw_dir, '**/*.jpeg'), recursive=True))
    
    print(f" Found {len(image_paths)} images.")
    
    if len(image_paths) == 0:
        print(" No images found! Please ensure you extracted the zip file and named the folder 'raw_dataset'.")
        return

    data = []
    for path in image_paths:
        # Normalize path to use forward slashes for easier, cross-platform checking
        norm_path = path.replace('\\', '/').lower()
        filename = os.path.basename(path).lower()
        
        # Check BOTH the parent folder path AND the filename for keywords
        if 'hole' in norm_path:
            label = 'hole'
        elif 'paper' in norm_path:
            label = 'paper'
        else:
            label = 'penMark'
            
        data.append((path, label))
        
    # Debug: print first 3 items to verify labeling is working
    print(" Sample labeling:")
    for p, l in data[:3]:
        print(f"   File: {os.path.basename(p)} -> Label: {l}")

    labels = [x[1] for x in data]
    unique_labels = list(set(labels))
    print(f"🏷️  Unique labels found in dataset: {unique_labels}")
    
    if len(unique_labels) == 1:
        print("⚠️  WARNING: Only ONE class was found!")
        print("   Please check your 'raw_dataset' folder. It should contain subfolders or files named with 'hole', 'mark', or 'pen'.")
        
    # Split: 70% train, 15% val, 15% test
    train_data, temp_data = train_test_split(data, test_size=0.30, random_state=42, stratify=labels)
    temp_labels = [x[1] for x in temp_data]
    val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42, stratify=temp_labels)
    
    splits = {'train': train_data, 'val': val_data, 'test': test_data}
    
    print(f"\nOrganizing data into '{target_dir}'...")
    for split_name, split_data in splits.items():
        for img_path, label in split_data:
            dest_dir = os.path.join(target_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            filename = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(dest_dir, filename))
            
    print(" Data successfully copied into 'organized_dataset'!")
    
    # Robust Verification using os.walk (100% reliable on Windows)
    print("\n Final Verification:")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(target_dir, split)
        if os.path.exists(split_dir):
            # Count all files in the split directory and its subdirectories
            total_files = sum([len(files) for r, d, files in os.walk(split_dir)])
            print(f"   -> {split.capitalize()} set: {total_files} images")
        else:
            print(f"   -> {split.capitalize()} set: 0 images (Directory not created!)")

def main():
    raw_dir = 'raw_dataset'
    target_dir = 'organized_dataset'
    
    if not os.path.exists(raw_dir):
        print(" 'raw_dataset' folder not found.")
        print("ACTION REQUIRED: Please manually download the folder from the Google Drive link,")
        print("   extract the .zip file in this directory, and rename the extracted folder to 'raw_dataset'.")
    else:
        print(" 'raw_dataset' found. Proceeding to organization...")
        organize_dataset(raw_dir, target_dir)

if __name__ == "__main__":
    main()