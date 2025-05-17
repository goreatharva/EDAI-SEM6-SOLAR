import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def generate_mask(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Show the results using matplotlib
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(mask, cmap='gray')
    plt.title('Generated Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Ask for user confirmation
    while True:
        response = input(f"\nAccept mask for {image_path}? (y/n): ").lower()
        if response in ['y', 'n']:
            break
        print("Please enter 'y' for yes or 'n' for no.")
    
    if response == 'y':
        return mask
    return None

def main():
    print("Rooftop Mask Generator (Non-GUI Version)")
    print("=" * 50)
    
    # Find all jpg images
    images = sorted(glob.glob('[0-9]*.jpg'))
    
    if not images:
        print("No numbered jpg files found!")
        return
    
    print(f"Found {len(images)} images to process")
    print("\nStarting mask generation...")
    print("\nFor each image:")
    print("1. Review the three images shown (Original, Grayscale, and Mask)")
    print("2. Enter 'y' to accept the mask or 'n' to skip this image")
    
    for img_path in images:
        # Generate corresponding mask filename
        mask_path = os.path.splitext(img_path)[0] + '.png'
        
        # Skip if mask already exists
        if os.path.exists(mask_path):
            print(f"Mask already exists for {img_path}, skipping...")
            continue
        
        print(f"\nProcessing: {img_path}")
        
        # Generate mask
        mask = generate_mask(img_path)
        
        if mask is not None:
            # Save the mask
            cv2.imwrite(mask_path, mask)
            print(f"✅ Saved mask for {img_path}")
        else:
            print(f"⚠️ Skipped {img_path}")
    
    plt.close('all')
    print("\nMask generation complete!")
    print("You can now run solar_panel_placement.py to analyze the rooftops.")

if __name__ == "__main__":
    main() 