# File: defect_localization.py
import os
import cv2
import numpy as np
import joblib
import glob
from feature_extraction import extract_features_from_image

# Class mapping based on how we trained the model
CLASSES = [ 'hole','paper', 'penMark']

def contour_based_localization(image_path, model, scaler, min_area=100):
    
    # Uses classical CV (thresholding & contours) to find candidate defect regions,
    # then uses the ML model to classify them.
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f" Could not read image: {image_path}")
        return None, False
        
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Preprocessing: Blur and Threshold to isolate dark spots (defects) from light paper
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Use Otsu's thresholding to automatically find the best cutoff for dark vs light
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Fallback if Otsu fails (e.g., image is completely white/defect-free)
    if np.sum(thresh) == 0:
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological closing: connects broken pixels in a pen mark so it's seen as one object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 2. Find contours (these are our candidate defect regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defect_found = False
    
    # 3. Evaluate each contour region with the trained ML model
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter out tiny noise (dust, sensor noise)
        if area < min_area:
            continue
            
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Add a small margin so we don't cut off the edges of the defect
        margin = 10
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_bgr.shape[1], x + w + margin)
        y2 = min(img_bgr.shape[0], y + h + margin)
        
        # Extract the Region of Interest (ROI)
        roi_bgr = img_bgr[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        
        # If ROI is too small, skip
        if roi_rgb.shape[0] < 10 or roi_rgb.shape[1] < 10:
            continue
            
        # 4. Classify the ROI using our trained model
        features = extract_features_from_image(roi_rgb)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        # 5. If the model confirms it's a hole (1) or mark (2), draw the box!
        if prediction in [1, 2]:
            defect_found = True
            predicted_class = CLASSES[prediction]
            color = (0, 255, 0) # Green box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_bgr, predicted_class.upper(), (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_bgr, defect_found

def main():
    print(" Loading trained model and scaler...")
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    output_dir = 'localized_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear old outputs if they exist
    for f in glob.glob(os.path.join(output_dir, '*.*')):
        os.remove(f)
    
    target_classes = ['hole', 'paper','penMark']
    total_detected = 0
    
    print("\n Starting Defect Localization (Contour-based Region Proposal)...\n")
    
    for cls in target_classes:
        image_paths = glob.glob(os.path.join('organized_dataset', 'test', cls, '*.*'))
        print(f" Processing {len(image_paths)} images from the '{cls}' test set...")
        
        for img_path in image_paths:
            result_img, detected = contour_based_localization(img_path, model, scaler)
            
            if result_img is not None:
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, f"loc_{cls}_{filename}")
                cv2.imwrite(output_path, result_img)
                if detected:
                    total_detected += 1
                
    print(f"\n Localization complete!")
    print(f" Successfully localized defects in {total_detected} images.")
    print(f" Check the '{output_dir}' folder to see your images with green bounding boxes!")

if __name__ == "__main__":
    main()