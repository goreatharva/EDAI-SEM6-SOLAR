import cv2
import numpy as np
import matplotlib.pyplot as plt

def close_contours(edges, max_gap=20):
    """Aggressively close contours by connecting nearby endpoints"""
    # Dilate edges first to connect very close points
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find endpoints
    skel_endpoints = np.zeros_like(dilated)
    kernel = np.array([[1, 1, 1],
                      [1, 10, 1],
                      [1, 1, 1]])
    
    filtered = cv2.filter2D(dilated.astype(float), -1, kernel)
    endpoints = np.where((filtered == 11) & (dilated == 255))
    
    # Create endpoint image
    for y, x in zip(endpoints[0], endpoints[1]):
        skel_endpoints[y, x] = 255
    
    # Connect endpoints
    result = dilated.copy()
    endpoint_coords = list(zip(endpoints[0], endpoints[1]))
    
    for i, (y1, x1) in enumerate(endpoint_coords):
        min_dist = float('inf')
        closest_point = None
        
        for j, (y2, x2) in enumerate(endpoint_coords[i+1:], i+1):
            dist = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            if dist < min_dist and dist <= max_gap:
                min_dist = dist
                closest_point = (y2, x2)
        
        if closest_point:
            cv2.line(result, (x1, y1), (closest_point[1], closest_point[0]), 255, 1)
    
    # Clean up
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    return result

def enhance_edges(img):
    """Enhanced edge detection with multiple methods"""
    # Normalize and enhance contrast
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_norm)
    
    # Multiple edge detection methods
    edges1 = cv2.Canny(img_clahe, 30, 150)
    
    # Sobel edges
    sobelx = cv2.Sobel(img_clahe, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_clahe, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges2 = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Laplacian edges
    laplacian = cv2.Laplacian(img_clahe, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, edges3 = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine all edges
    edges = cv2.bitwise_or(edges1, edges2)
    edges = cv2.bitwise_or(edges, edges3)
    
    return edges

def find_rectangles(contours, min_area=100, max_area=None):
    """Find rectangular contours with angle checks"""
    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or (max_area and area > max_area):
            continue
            
        # Try to fit a rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Check if it's approximately rectangular
        rect_area = cv2.contourArea(box)
        if rect_area > 0:
            extent = float(area) / rect_area
            if extent > 0.7:  # At least 70% rectangular
                rectangles.append(box)
    
    return rectangles

def process_image(image_path):
    """Process image with improved polygon detection"""
    # Read and validate image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Initial preprocessing
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Get edges using multiple methods
    edges = enhance_edges(img)
    
    # Close contours
    closed_edges = close_contours(edges)
    
    # Create binary image
    _, binary = cv2.threshold(closed_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours with different methods
    contours1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine contours
    all_contours = contours1 + contours2
    
    # Create output images
    polygon_img = np.zeros_like(img)
    filled_polygon_img = np.zeros_like(img)
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Find rectangles
    min_area = img.size * 0.01  # 1% of image size
    max_area = img.size * 0.9   # 90% of image size
    rectangles = find_rectangles(all_contours, min_area, max_area)
    
    # Draw and fill rectangles
    for rect in rectangles:
        # Draw filled polygon
        cv2.fillPoly(filled_polygon_img, [rect], 255)
        
        # Draw outline
        cv2.drawContours(polygon_img, [rect], 0, 255, 2)
        
        # Draw vertices
        for point in rect:
            cv2.circle(polygon_img, tuple(point), 3, 128, -1)
        
        # Debug visualization
        cv2.drawContours(debug_img, [rect], 0, (0, 255, 0), 2)
    
    # If no rectangles found, try with more lenient parameters
    if not rectangles:
        print("No rectangles found, trying with more lenient parameters...")
        # Try to find any significant contours
        for contour in all_contours:
            area = cv2.contourArea(contour)
            if area < min_area * 0.5:
                continue
                
            # Approximate the contour
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:  # At least quadrilateral
                cv2.fillPoly(filled_polygon_img, [approx], 255)
                cv2.drawContours(polygon_img, [approx], 0, 255, 2)
                cv2.drawContours(debug_img, [approx], 0, (0, 0, 255), 2)
                
                for point in approx:
                    cv2.circle(polygon_img, tuple(point[0]), 3, 128, -1)
    
    # Create visualization
    plt.figure(figsize=(20, 5))
    
    # Original image
    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    # Edge detection
    plt.subplot(142)
    plt.title('Closed Edges')
    plt.imshow(closed_edges, cmap='gray')
    plt.axis('off')
    
    # Polygon outline
    plt.subplot(143)
    plt.title('Polygon Outline')
    plt.imshow(polygon_img, cmap='gray')
    plt.axis('off')
    
    # Filled polygon
    plt.subplot(144)
    plt.title('Filled Polygon')
    plt.imshow(filled_polygon_img, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save all intermediate results for debugging
    cv2.imwrite('edges.png', edges)
    cv2.imwrite('closed_edges.png', closed_edges)
    cv2.imwrite('binary.png', binary)
    cv2.imwrite('polygon_approx.png', polygon_img)
    cv2.imwrite('filled_polygon.png', filled_polygon_img)
    cv2.imwrite('debug_visualization.png', debug_img)
    
    return closed_edges, polygon_img, filled_polygon_img

if __name__ == "__main__":
    # Look for images in current directory
    import glob
    images = glob.glob('*.jpg') + glob.glob('*.png')
    
    if not images:
        print("No image files found in current directory")
        exit(1)
        
    print("Available images:", images)
    image_path = input("Enter image name to process (or press Enter for first image): ").strip() or images[0]
    
    try:
        edges, polygon_img, filled_polygon_img = process_image(image_path)
        print("Processing complete. Results saved as PNG files.")
        
        if np.sum(filled_polygon_img) == 0:
            print("\nWarning: No polygons were detected in the image.")
            print("Suggestions:")
            print("1. Check edges.png and closed_edges.png for edge detection quality")
            print("2. Check binary.png for the thresholded image")
            print("3. Check debug_visualization.png for detected shapes")
            print("4. Try adjusting the image contrast or using a different image")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc() 