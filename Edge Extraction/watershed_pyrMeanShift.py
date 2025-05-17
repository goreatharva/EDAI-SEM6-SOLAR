# import the necessary packages
from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


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
        print("Please place a valid image file (e.g., der.jpg) in the same directory.")
        sys.exit(1)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        sys.exit(1)

    # Create figure for visualization
    plt.figure(figsize=(20, 5))
    
    # Show original image
    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Apply histogram equalization
    im = equalize(image.copy())
    
    # Apply pyramid mean shift filtering
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    plt.subplot(142)
    plt.title('Thresholded Image')
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    # Compute distance transform and find local maxima
    D = ndimage.distance_transform_edt(thresh)
    coordinates = peak_local_max(D, min_distance=20, labels=thresh)
    localMax = np.zeros_like(D, dtype=bool)
    localMax[tuple(coordinates.T)] = True

    # Apply watershed segmentation
    markers = ndimage.label(localMax)[0]
    labels = watershed(-D, markers, mask=thresh, watershed_line=True)
    
    # Create output image
    output = image.copy()
    
    # Count unique segments
    n_segments = len(np.unique(labels)) - 1
    print(f"[INFO] {n_segments} unique segments found")

    # Process each segment
    for label in np.unique(labels):
        if label == 0:
            continue
            
        # Create mask for current label
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # Find contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Draw contours and labels
        for (i, c) in enumerate(cnts):
            ((x, y), _) = cv2.minEnclosingCircle(c)
            cv2.putText(output, f"#{label}", (int(x) - 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(output, [c], -1, (0, 255, 0), 2)

    # Show segmentation results
    plt.subplot(143)
    plt.title('Distance Transform')
    plt.imshow(D, cmap='jet')
    plt.axis('off')
    
    plt.subplot(144)
    plt.title(f'Final Segmentation ({n_segments} segments)')
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    # Save results
    cv2.imwrite('watershed_result.png', output)
    
    return output, labels


if __name__ == "__main__":
    image_path = 'der.jpg'
    try:
        output, labels = process_image(image_path)
        print("Processing complete. Check 'watershed_result.png' for results.")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
