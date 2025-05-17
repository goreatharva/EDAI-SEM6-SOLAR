import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians
import colorsys
import json
from performance_evaluation import evaluate_model

def create_colorful_visualization(img, alpha=0.7):
    """Create a colorful heatmap-like visualization"""
    # Normalize the input image to 0-255 range
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create a colored heatmap using cv2.COLORMAP_JET
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # Blend with the grayscale image
    gray_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(heatmap, alpha, gray_img, 1-alpha, 0)
    
    return blended

def place_solar_panels(rooftop_mask_path, original_image_path=None):
    """
    Enhanced solar panel placement system with advanced visualization
    """
    # Read and preprocess images
    rooftop_mask = cv2.imread(rooftop_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Store original grayscale for visualization
    original_grayscale = rooftop_mask.copy()
    
    # Improve rooftop detection
    _, binary_mask = cv2.threshold(rooftop_mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    if original_image_path:
        original_img = cv2.imread(original_image_path)
    else:
        # Use the original grayscale values instead of binary mask
        original_img = cv2.cvtColor(original_grayscale, cv2.COLOR_GRAY2BGR)

    # Enhanced parameters
    PANEL_WIDTH = 20
    PANEL_HEIGHT = 10
    MIN_SPACING = 5
    EDGE_MARGIN = 10  # Increased edge margin
    MIN_AREA = 500   # Increased minimum area

    # Panel specifications
    PANEL_POWER = 400  # Watts per panel
    PANEL_EFFICIENCY = 0.20
    SOLAR_IRRADIANCE = 1000  # W/m²
    PERFORMANCE_RATIO = 0.75  # System losses

    def optimize_panel_orientation(contour):
        """Determine optimal panel orientation based on rooftop shape"""
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if angle < -45:
            angle = 90 + angle
        return angle

    def create_panel_mask(width, height, angle):
        """Generate a rotated panel mask with enhanced borders"""
        mask = np.zeros((height * 3, width * 3), dtype=np.uint8)
        center = (width * 1.5, height * 1.5)
        rect = ((center[0], center[1]), (PANEL_WIDTH, PANEL_HEIGHT), angle)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        cv2.drawContours(mask, [box], 0, 255, -1)
        return mask, box

    def check_panel_placement(position, angle, occupied_spaces, contour, binary_mask):
        """Validate panel placement with enhanced constraints"""
        mask, box = create_panel_mask(PANEL_WIDTH, PANEL_HEIGHT, angle)
        box = box + [position[0] - PANEL_WIDTH * 1.5, position[1] - PANEL_HEIGHT * 1.5]
        box = np.array(box, dtype=np.int32)
        
        # Convert position to the correct format for pointPolygonTest
        test_point = (float(position[0]), float(position[1]))
        
        # Check boundary constraints
        if cv2.pointPolygonTest(contour, test_point, False) < 0:
            return False
            
        # Check if the entire panel area is within the rooftop
        panel_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        cv2.drawContours(panel_mask, [box], 0, 255, -1)
        if not np.all(cv2.bitwise_and(panel_mask, binary_mask) == panel_mask):
            return False

        # Convert position to integers for array slicing
        y = int(position[1])
        x = int(position[0])
        
        # Check spacing and overlap
        y_start = max(0, y - PANEL_HEIGHT - MIN_SPACING)
        y_end = min(occupied_spaces.shape[0], y + PANEL_HEIGHT + MIN_SPACING)
        x_start = max(0, x - PANEL_WIDTH - MIN_SPACING)
        x_end = min(occupied_spaces.shape[1], x + PANEL_WIDTH + MIN_SPACING)
        
        roi = occupied_spaces[y_start:y_end, x_start:x_end]
        return not np.any(roi > 0)

    # Process rooftop sections
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = original_img.copy()
    panel_mask = np.zeros_like(binary_mask)
    heat_map = np.zeros_like(binary_mask, dtype=np.float32)
    
    total_panels = 0
    total_area = 0
    panels_info = []

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        # Create a mask for this contour
        contour_mask = np.zeros_like(binary_mask)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        
        angle = optimize_panel_orientation(contour)
        occupied_spaces = np.zeros_like(binary_mask)
        x, y, w, h = cv2.boundingRect(contour)

        # Optimized panel placement grid
        for i in range(x + EDGE_MARGIN, x + w - EDGE_MARGIN, PANEL_WIDTH + MIN_SPACING):
            for j in range(y + EDGE_MARGIN, y + h - EDGE_MARGIN, PANEL_HEIGHT + MIN_SPACING):
                position = np.array([i, j], dtype=np.float32)
                
                if check_panel_placement(position, angle, occupied_spaces, contour, binary_mask):
                    panel_template, box = create_panel_mask(PANEL_WIDTH, PANEL_HEIGHT, angle)
                    box = box + [position[0] - PANEL_WIDTH * 1.5, position[1] - PANEL_HEIGHT * 1.5]
                    box = np.array(box, dtype=np.int32)
                    
                    # Enhanced visualization
                    color = (0, 255, 100)
                    cv2.drawContours(result_img, [box], 0, color, 2)
                    cv2.drawContours(panel_mask, [box], 0, 255, -1)
                    cv2.drawContours(occupied_spaces, [box], 0, 255, -1)
                    
                    # Update heat map
                    cv2.drawContours(heat_map, [box], 0, PANEL_POWER/1000, -1)
                    
                    panels_info.append({
                        'position': position.tolist(),
                        'angle': angle,
                        'power': PANEL_POWER
                    })
                    
                    total_panels += 1
                    total_area += PANEL_WIDTH * PANEL_HEIGHT

    # Calculate solar potential
    total_power_kw = total_panels * PANEL_POWER / 1000
    annual_energy_kwh = total_power_kw * 5.5 * 365 * PERFORMANCE_RATIO  # Assuming 5.5 peak sun hours

    # Enhanced visualization
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    plt.title('Original Rooftop')
    plt.imshow(original_grayscale, cmap='gray')  # Use original grayscale values
    plt.axis('off')
    
    plt.subplot(142)
    plt.title('Panel Placement Layout')
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(143)
    plt.title(f'Solar Power Density\n({total_power_kw:.1f} kW potential)')
    colorful_heatmap = create_colorful_visualization(heat_map)
    plt.imshow(cv2.cvtColor(colorful_heatmap, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(144)
    plt.title(f'Coverage Analysis\n({total_panels} panels)')
    coverage_vis = cv2.addWeighted(
        cv2.cvtColor(panel_mask, cv2.COLOR_GRAY2BGR),
        0.7,
        original_img,
        0.3,
        0
    )
    plt.imshow(cv2.cvtColor(coverage_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Save enhanced visualizations
    cv2.imwrite('panel_layout.png', result_img)
    cv2.imwrite('power_density.png', colorful_heatmap)
    cv2.imwrite('coverage_analysis.png', coverage_vis)

    results = {
        'num_panels': total_panels,
        'total_power_kw': total_power_kw,
        'annual_energy_kwh': annual_energy_kwh,
        'panels_info': panels_info
    }

    # Perform performance evaluation
    evaluation_metrics = evaluate_model(results, panel_mask, binary_mask)
    
    return results, evaluation_metrics

if __name__ == "__main__":
    import glob
    import os
    from datetime import datetime

    print("Solar Panel Placement System - Batch Processing")
    print("=" * 50)

    # Create a results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Find all input images
    input_images = sorted(glob.glob('[0-9]*.jpg'))  # This will match 1.jpg, 2.jpg, etc.
    
    if not input_images:
        print("❌ Error: No input images found (looking for numbered jpg files like 1.jpg, 2.jpg, etc.)")
        exit(1)

    print(f"📁 Found {len(input_images)} images to process")
    print("-" * 50)

    # Store results for summary
    all_results = []
    all_metrics = []

    # Process each image
    for idx, img_path in enumerate(input_images, 1):
        print(f"\n🔄 Processing image {idx}/{len(input_images)}: {img_path}")
        
        # Create output directory for this image
        output_dir = os.path.join(base_output_dir, f"image_{idx}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate mask filename (assuming same name but .png extension)
        mask_name = os.path.splitext(img_path)[0] + '.png'
        
        if not os.path.exists(mask_name):
            print(f"⚠️ Warning: Mask file {mask_name} not found, skipping...")
            continue

        try:
            # Change working directory to output directory
            original_cwd = os.getcwd()
            os.chdir(output_dir)
            
            # Process image and get results
            results, metrics = place_solar_panels(os.path.join(original_cwd, mask_name),
                                               os.path.join(original_cwd, img_path))
            
            all_results.append(results)
            all_metrics.append(metrics)
            
            print(f"✅ Successfully processed {img_path}")
            print(f"   - Panels placed: {results['num_panels']}")
            print(f"   - Total power: {results['total_power_kw']:.2f} kW")
            print(f"   - Overall improvement vs baseline: {metrics['summary']['total_improvement']:.2f}%")
            
            os.chdir(original_cwd)
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {str(e)}")
            os.chdir(original_cwd)
            continue

    # Generate overall summary
    if all_results:
        print("\n=== Overall Performance Summary ===")
        avg_improvement = np.mean([m['summary']['total_improvement'] for m in all_metrics])
        total_power = sum(r['total_power_kw'] for r in all_results)
        total_panels = sum(r['num_panels'] for r in all_results)
        
        print(f"Total number of panels placed: {total_panels}")
        print(f"Total power capacity: {total_power:.2f} kW")
        print(f"Average improvement over baseline: {avg_improvement:.2f}%")
        
        # Save overall summary
        summary = {
            'total_panels': total_panels,
            'total_power_kw': total_power,
            'average_improvement': avg_improvement,
            'processed_images': len(all_results),
            'timestamp': timestamp
        }
        
        with open(os.path.join(base_output_dir, 'overall_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"\nResults saved in: {base_output_dir}")
    else:
        print("\n❌ No images were successfully processed") 