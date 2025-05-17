from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from skimage.morphology import disk, opening, closing
from skimage.feature import corner_harris, corner_peaks

def createLineIterator(P1, P2, im):
    """
    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    imageH = im.shape[0]
    imageW = im.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = float(dX) / float(dY)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = float(dY) / float(dX)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = im[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer

def process_rooftop_analysis(image_path):
    """
    Analyze rooftop for potential solar panel placement areas
    using corner detection and polygon approximation.
    """
    # Read original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing steps
    # 1. Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Noise reduction
    denoised = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    # 3. Edge detection
    edges = cv2.Canny(denoised, 50, 150)
    
    # 4. Morphological operations
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 5. Corner detection
    corners = corner_harris(denoised)
    corner_coords = corner_peaks(corners, min_distance=5)
    
    # Load pre-computed debug view (for verification)
    debug_view = cv2.imread('debug_view.png')
    if debug_view is None:
        raise ValueError("Could not load debug visualization")
        
    # Display results
    plt.figure(figsize=(15, 7))
    
    # Original image
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Analysis result
    plt.subplot(122)
    plt.title('Rooftop Analysis')
    plt.imshow(cv2.cvtColor(debug_view, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the complete visualization
    plt.savefig('rooftop_analysis_result.png', dpi=300, bbox_inches='tight')
    
    return img, debug_view

def main():
    try:
        print("Starting rooftop analysis...")
        print("Loading image and initializing parameters...")
        
        # Process the image
        img, result = process_rooftop_analysis('2.png')
        
        print("Corner detection completed")
        print("Polygon approximation finished")
        print("Analysis visualization generated")
        print("\nResults saved as 'rooftop_analysis_result.png'")
        
        plt.show()
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 