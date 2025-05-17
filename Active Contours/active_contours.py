import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2
import os
import sys

def process_image(image_path):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        print("Please place a bilate.png file (bilateral filtered and sharpened image) in the same directory.")
        sys.exit(1)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        sys.exit(1)

    # Convert to grayscale if not already
    if len(img.shape) == 3:
        img = rgb2gray(img)

    # Initialize contour
    s = np.linspace(0, 2*np.pi, 1000)
    # Adjust these values based on your image size
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    radius = min(center_x, center_y) - 10

    x = center_x + radius*np.cos(s)
    y = center_y + radius*np.sin(s)
    init = np.array([x, y]).T

    print("Initial contour shape:", init.shape)

    # Apply active contour
    try:
        snake = active_contour(
            gaussian(img, 2),
            init,
            alpha=0.01,   # Reduced to allow more flexibility
            beta=1.0,     # Reduced for less smoothness constraint
            gamma=0.001,  # Time step parameter
            w_line=0,     # Line attraction
            w_edge=5.0    # Increased edge attraction
        )
        
        # Plotting
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.gray()
        ax.imshow(img)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=3, label='Initial contour')
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3, label='Final contour')
        ax.set_title('Active Contour Evolution')
        ax.legend()
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()
        
        # Save result with better visualization
        result_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.polylines(result_img, [init.astype(np.int32)], True, (0, 0, 255), 2)  # Red for initial
        cv2.polylines(result_img, [snake.astype(np.int32)], True, (0, 255, 0), 2)  # Green for final
        cv2.imwrite('active_contour_result.png', result_img)
        
        return snake
    except Exception as e:
        print(f"Error during active contour processing: {str(e)}")
        return None

if __name__ == "__main__":
    image_path = 'bilate.png'
    process_image(image_path)
