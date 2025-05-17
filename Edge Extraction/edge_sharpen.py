import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import opening, disk, closing
import os
import sys


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def process_image(image_path):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        print("Please place a valid image file (e.g., build1.jpg) in the same directory.")
        sys.exit(1)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        sys.exit(1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create figure for visualization
    plt.figure(figsize=(15, 5))
    
    # Original grayscale
    plt.subplot(131)
    plt.title('Original Grayscale')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    # Apply bilateral filter for edge-preserving smoothing
    blur = cv2.bilateralFilter(gray, 5, 75, 75)
    
    # Define sharpening kernel
    kernel_sharp = np.array([
        [-2, -2, -2],
        [-2, 17, -2],
        [-2, -2, -2]], dtype='int')
    
    # Apply sharpening
    sharpened = cv2.filter2D(blur, -1, kernel_sharp)
    
    # Show sharpened image
    plt.subplot(132)
    plt.title('Sharpened Image')
    plt.imshow(sharpened, cmap='gray')
    plt.axis('off')
    
    # Apply auto-canny edge detection
    canny = auto_canny(sharpened)
    
    # Show edges
    plt.subplot(133)
    plt.title('Edge Detection')
    plt.imshow(canny, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the processed images
    cv2.imwrite('sharpened.png', sharpened)
    cv2.imwrite('edges.png', canny)
    
    return sharpened, canny


if __name__ == "__main__":
    # You can change this to process different images
    image_path = 'build1.jpg'
    try:
        sharpened, edges = process_image(image_path)
        print(f"Processing complete. Check 'sharpened.png' and 'edges.png' for results.")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
