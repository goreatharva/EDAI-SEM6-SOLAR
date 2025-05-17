from __future__ import print_function
# import Image
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import os
import sys
from shapely.geometry import Polygon

# Constants for Google Maps calculations
ZOOM = 20
TILE_SIZE = 256
INITIAL_RESOLUTION = 2 * math.pi * 6378137 / TILE_SIZE
ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0
EARTH_CIRCUMFERENCE = 6378137 * 2 * math.pi
FACTOR = math.pow(2, ZOOM)
MAP_WIDTH = 256 * (2 ** ZOOM)

# Added constants for validation
MIN_PANEL_LENGTH = 500  # mm
MAX_PANEL_LENGTH = 2000  # mm
MIN_PANEL_WIDTH = 300  # mm
MAX_PANEL_WIDTH = 1200  # mm
MAX_PANELS_TOGETHER = 10

def validate_image(image_path):
    """Validate image file existence and format"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found")
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image '{image_path}'")
        return img
    except Exception as e:
        raise ValueError(f"Error reading image: {str(e)}")

def grays(im):
    """Convert image to grayscale"""
    if im is None:
        raise ValueError("Input image is None")
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def white_image(im):
    """Create a white image of the same size"""
    if im is None:
        raise ValueError("Input image is None")
    return cv2.bitwise_not(np.zeros(im.shape, np.uint8))

def pixels_per_mm(lat, length):
    """Calculate pixels per mm at given latitude"""
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    return length / math.cos(lat * math.pi / 180) * EARTH_CIRCUMFERENCE * 1000 / MAP_WIDTH

def sharp(gray):
    """Apply sharpening to grayscale image"""
    if gray is None:
        raise ValueError("Input image is None")
    try:
    blur = cv2.bilateralFilter(gray, 5, sigmaColor=7, sigmaSpace=5)
        kernel_sharp = np.array([
        [-2, -2, -2],
        [-2, 17, -2],
            [-2, -2, -2]], dtype='int')
    return cv2.filter2D(blur, -1, kernel_sharp)
    except Exception as e:
        raise ValueError(f"Error in sharpening: {str(e)}")

def get_solar_panel_params():
    """Get solar panel parameters from user input with validation"""
    try:
        while True:
            try:
                panel_lens = int(input("Number of panels together (1-10, default=1): ") or "1")
                if not 1 <= panel_lens <= MAX_PANELS_TOGETHER:
                    print(f"Number of panels must be between 1 and {MAX_PANELS_TOGETHER}")
                    continue
                
                panel_wids = 1  # Fixed width for simplicity
                
                length_s = float(input(f"Enter length of panel in mm ({MIN_PANEL_LENGTH}-{MAX_PANEL_LENGTH}, default=1650): ") or "1650")
                if not MIN_PANEL_LENGTH <= length_s <= MAX_PANEL_LENGTH:
                    print(f"Panel length must be between {MIN_PANEL_LENGTH} and {MAX_PANEL_LENGTH} mm")
                    continue
                
                width = float(input(f"Enter width of panel in mm ({MIN_PANEL_WIDTH}-{MAX_PANEL_WIDTH}, default=992): ") or "992")
                if not MIN_PANEL_WIDTH <= width <= MAX_PANEL_WIDTH:
                    print(f"Panel width must be between {MIN_PANEL_WIDTH} and {MAX_PANEL_WIDTH} mm")
                    continue
                
                angle = float(input("Rotation Angle for Solar Panels (0-360, default=0): ") or "0")
                angle = angle % 360  # Normalize angle to 0-360 range
                
    return panel_lens, panel_wids, length_s, width, angle

            except ValueError:
                print("Please enter valid numeric values")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)

def contours_canny(cnts):
    cv2.drawContours(canny_contours, cnts, -1, 255, 1)

    # Removing the contours detected inside the roof
    for cnt in cnts:
        counters = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []

        if cv2.contourArea(cnt) > 10:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counters += 1
                    pts.append((x, y))

        if counters > 10:
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(canny_polygons, [pts], True, 0)


def contours_img(cnts):
    cv2.drawContours(image_contours, cnts, -1, 255, 1)

    # Removing the contours detected inside the roof
    for cnt in cnts:
        counter = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []
        if cv2.contourArea(cnt) > 5:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counter += 1
                    pts.append((x, y))
        if counter > 10:
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(image_polygons, [pts], True, 0)


def rotation(center_x, center_y, points, ang):
    """Rotate points around center by angle in degrees with numerical stability improvements"""
    try:
        # Normalize angle to 0-360 range
        ang = ang % 360
        angle = math.radians(ang)
        
        # Use more stable sin/cos calculations for common angles
        if ang in [0, 90, 180, 270]:
            sin_ang = [0, 1, 0, -1][int(ang/90)]
            cos_ang = [1, 0, -1, 0][int(ang/90)]
        else:
            sin_ang = math.sin(angle)
            cos_ang = math.cos(angle)
        
    rotated_points = []
        center_x = float(center_x)
        center_y = float(center_y)
        
    for p in points:
            x, y = float(p[0]), float(p[1])
            dx = x - center_x
            dy = y - center_y
            
            # Use more numerically stable rotation formula
            new_x = center_x + (dx * cos_ang - dy * sin_ang)
            new_y = center_y + (dx * sin_ang + dy * cos_ang)
            
            rotated_points.append((int(round(new_x)), int(round(new_y))))
        
    return rotated_points
    except Exception as e:
        raise ValueError(f"Error in rotation calculation: {str(e)}")


def createLineIterator(P1, P2, img):
    imageH = img.shape[0]
    imageW = img.shape[1]
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
            slope = dX.astype(float) / dY.astype(float)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(float) / dX.astype(float)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


def panel_rotation(panels_series, solar_roof_area):

    high_reso = cv2.pyrUp(solar_roof_area)
    rows, cols = high_reso.shape
    high_reso_new = cv2.pyrUp(new_image)

    for _ in range(panels_series - 2):
        for col in range(0, cols, l + 1):
            for row in range(0, rows, w + 1):

                # Rectangular Region of interest for solar panel area
                solar_patch = high_reso[row:row + (w + 1) * pw + 1, col:col + ((l * pl) + 3)]
                r, c = solar_patch.shape

                # Rotation of rectangular patch according to the angle provided
                patch_rotate = np.array([[col, row], [c + col, row], [c + col, r + row], [col, r + row]], np.int32)
                rotated_patch_points = rotation((col + c) / 2, row + r / 2, patch_rotate, solar_angle)
                rotated_patch_points = np.array(rotated_patch_points, np.int32)

                # Check for if rotated points go outside of the image
                if (rotated_patch_points > 0).all():
                    solar_polygon = Polygon(rotated_patch_points)
                    polygon_points = np.array(solar_polygon.exterior.coords, np.int32)

                    # Appending points of the image inside the solar area to check the intensity
                    patch_intensity_check = []

                    # Point polygon test for each rotated solar patch area
                    for j in range(rows):
                        for k in range(cols):
                            if cv2.pointPolygonTest(polygon_points, (k, j), False) == 1:
                                patch_intensity_check.append(high_reso[j, k])

                    # Check for the region available for Solar Panels
                    if np.mean(patch_intensity_check) == 255:

                        # Moving along the length of line to segment solar panels in the patch
                        solar_line_1 = createLineIterator(rotated_patch_points[0], rotated_patch_points[1], high_reso)
                        solar_line_1 = solar_line_1.astype(int)
                        solar_line_2 = createLineIterator(rotated_patch_points[3], rotated_patch_points[2], high_reso)
                        solar_line_2 = solar_line_2.astype(int)
                        line1_points = []
                        line2_points = []
                        if len(solar_line_2) > 10 and len(solar_line_1) > 10:

                            # Remove small unwanted patches
                            cv2.fillPoly(high_reso, [rotated_patch_points], 0)
                            cv2.fillPoly(high_reso_new, [rotated_patch_points], 0)
                            cv2.polylines(high_reso_orig, [rotated_patch_points], 1, 0, 2)
                            cv2.polylines(high_reso_new, [rotated_patch_points], 1, 0, 2)

                            cv2.fillPoly(high_reso_orig, [rotated_patch_points], (0, 0, 255))
                            cv2.fillPoly(high_reso_new, [rotated_patch_points], (0, 0, 255))

                            for i in range(5, len(solar_line_1), 5):
                                line1_points.append(solar_line_1[i])
                            for i in range(5, len(solar_line_2), 5):
                                line2_points.append(solar_line_2[i])

                        # Segmenting Solar Panels in the Solar Patch
                        for points1, points2 in zip(line1_points, line2_points):
                            x1, y1, _ = points1
                            x2, y2, _ = points2
                            cv2.line(high_reso_orig, (x1, y1), (x2, y2), (0, 0, 0), 1)
                            cv2.line(high_reso_new, (x1, y1), (x2, y2), (0, 0, 0), 1)

        # Number of Solar Panels in series (3/4/5)
        panels_series = panels_series - 1
    result = Image.fromarray(high_reso_orig)
    resut_2 = Image.fromarray(high_reso_new)
    result.save('output' + fname )
    resut_2.save('panels' + fname)
    plt.figure()
    plt.axis('off')
    plt.imshow(high_reso_orig)
    plt.figure()
    plt.axis('off')
    plt.imshow(high_reso_new)
    plt.show()


def process_image(image_path, panel_params=None):
    """Process image with improved edge detection and panel placement"""
    try:
        # Validate and load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")

        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Get panel parameters with defaults for testing
        if panel_params is None:
            panel_params = (1, 1, 1650, 992, 0)  # Default values
        panel_lens, panel_wids, length_s, width, angle = panel_params
        
        try:
            # Create working images
            gray = grays(img)
            
            # Enhanced pre-processing with CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Denoise and enhance edges
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 115, 2)
            
            # Morphological operations
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Edge detection with better parameters
            edges = cv2.Canny(binary, 30, 150)
            edges = cv2.dilate(edges, kernel, iterations=2)

            # Create output images
            canny_contours = np.zeros_like(img)
            image_polygons = np.zeros_like(img)
            
            # Find contours with better filtering
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by area
            min_area = img.shape[0] * img.shape[1] * 0.05  # At least 5% of image area
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            if contours:
                # Get largest contour (main roof area)
                roof_contour = contours[0]
                
                # Approximate the contour with less epsilon for more accurate shape
                epsilon = 0.01 * cv2.arcLength(roof_contour, True)
                roof_approx = cv2.approxPolyDP(roof_contour, epsilon, True)
                
                # Draw roof contour
                cv2.drawContours(canny_contours, [roof_approx], 0, (0, 255, 0), 2)
                
                # Create roof mask
                roof_mask = np.zeros_like(gray)
                cv2.drawContours(roof_mask, [roof_approx], 0, 255, -1)
                
                # Get rotated rectangle for better panel alignment
                rect = cv2.minAreaRect(roof_approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Get rectangle dimensions
                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = rect[2]
                
                # Calculate panel dimensions (more realistic)
                panel_width = max(width // 4, 40)  # At least 40 pixels wide
                panel_length = max(height // 3, 60)  # At least 60 pixels high
                
                # Calculate grid dimensions with proper spacing
                spacing = 1.2  # 20% spacing between panels
                n_panels_w = max(1, int(width / (panel_width * spacing)))
                n_panels_h = max(1, int(height / (panel_length * spacing)))
                
                print(f"Debug: Grid size: {n_panels_w}x{n_panels_h}")
                print(f"Debug: Panel size: {panel_width}x{panel_length}")
                
                # Get rotation matrix
                M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
                
                panels_placed = 0
                for i in range(n_panels_h):
                    for j in range(n_panels_w):
                        # Calculate panel position in grid
                        x = rect[0][0] - width/2 + j * panel_width * spacing + panel_width/2
                        y = rect[0][1] - height/2 + i * panel_length * spacing + panel_length/2
                        
                        # Create panel corners
                        panel = np.array([
                            [x - panel_width/2, y - panel_length/2],
                            [x + panel_width/2, y - panel_length/2],
                            [x + panel_width/2, y + panel_length/2],
                            [x - panel_width/2, y + panel_length/2]
                        ], dtype=np.float32)
                        
                        # Rotate panel
                        panel = np.array([np.dot(M, [p[0], p[1], 1]) for p in panel])
                        panel = panel[:, :2].astype(np.int32)
                        
                        # Check if panel center is inside roof area
                        center = np.mean(panel, axis=0, dtype=np.float32)
                        if cv2.pointPolygonTest(roof_approx, (center[0], center[1]), False) >= 0:
                            # Create panel mask
                            panel_mask = np.zeros_like(gray)
                            cv2.fillPoly(panel_mask, [panel], 255)
                            
                            # Check overlap with roof
                            overlap = cv2.bitwise_and(roof_mask, panel_mask)
                            overlap_ratio = cv2.countNonZero(overlap) / cv2.countNonZero(panel_mask)
                            
                            if overlap_ratio > 0.75:  # 75% overlap required
                                panels_placed += 1
                                # Draw panel with enhanced 3D effect
                                cv2.fillPoly(image_polygons, [panel], (139, 69, 19))  # Brown base
                                cv2.polylines(image_polygons, [panel], True, (0, 128, 255), 2)  # Orange border
                                
                                # Add highlight for 3D effect
                                highlight = np.array([
                                    panel[0],
                                    panel[1],
                                    panel[1] + [-3, -3],
                                    panel[0] + [-3, -3]
                                ], dtype=np.int32)
                                cv2.fillPoly(image_polygons, [highlight], (0, 165, 255))
                
                print(f"Debug: Placed {panels_placed} panels")
            
            # Create visualization
            plt.figure(figsize=(20, 5))
            
            plt.subplot(141)
            plt.title('Original Image')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(142)
            plt.title('Edge Detection')
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            
            plt.subplot(143)
            plt.title('Contour Detection')
            plt.imshow(cv2.cvtColor(canny_contours, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(144)
            plt.title(f'Solar Panel Areas ({panels_placed} panels)')
            plt.imshow(cv2.cvtColor(image_polygons, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Save results
            cv2.imwrite('solar_panel_areas.png', image_polygons)
            cv2.imwrite('contour_detection.png', canny_contours)
            cv2.imwrite('edge_detection.png', edges)
            
            return image_polygons
            
        finally:
            # Cleanup
            if 'binary' in locals():
                del binary
            if 'enhanced' in locals():
                del enhanced
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Look for images in current directory
    images = glob.glob('*.jpg') + glob.glob('*.png')
    
    if not images:
        print("No image files found in current directory.")
        print("Please place .jpg or .png files in the directory.")
        sys.exit(1)
        
    print("Available images:", images)
    image_path = input("Enter image name to process (or press Enter for first image): ") or images[0]
    
    try:
        result = process_image(image_path)
        print("\nProcessing complete:")
        print("- Results saved as 'solar_panel_areas.png'")
    except Exception as e:
        print(f"Error processing image: {str(e)}")