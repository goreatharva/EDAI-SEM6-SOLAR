from __future__ import print_function
import numpy as np
from skimage.filters import gabor
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import cv2
import os
import sys
import glob

def gabor_fit_func(img, frequency, theta):
    """Apply Gabor filter and fit Gaussian mixture model"""
    # Normalize input image
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Apply Gabor filter
    filter_img = gabor_filter(img, frequency, theta)
    
    # Enhance filter response
    filter_img = np.abs(filter_img)  # Get magnitude
    filter_img = cv2.normalize(filter_img, None, 0, 1, cv2.NORM_MINMAX)
    
    # Convert for histogram analysis
    hist1, hist2 = convert(filter_img)
    
    # Check if we have enough distinct values for clustering
    unique_values = np.unique(hist2)
    if len(unique_values) < 2:
        print("Warning: Not enough distinct values for clustering")
        # Try to enhance contrast
        filter_img = cv2.equalizeHist((filter_img * 255).astype(np.uint8)) / 255.0
        hist1, hist2 = convert(filter_img)
        unique_values = np.unique(hist2)
        
        if len(unique_values) < 2:
            mean1, mean2 = np.min(hist2), np.max(hist2)
            std1, std2 = 1, 1
        else:
            mean1, mean2, std1, std2 = gaussian_curve(hist2)
    else:
        mean1, mean2, std1, std2 = gaussian_curve(hist2)
    
    cost = gabor_cost_func(mean1, mean2, std1, std2)
    return cost, filter_img

def gabor_cost_func(m1, m2, v1, v2):
    """Calculate cost function for Gabor filter results"""
    # Avoid division by zero and ensure numerical stability
    eps = 1e-10
    denominator = v1 + v2 + eps
    J = abs(m2 - m1) / denominator
    return float(J)

def gabor_filter(gray, frequency, theta):
    """Apply Gabor filter with given parameters"""
    # Adjust parameters for better filter response
    mask = 15  # Increased mask size
    sigma = mask / 3  # Adjusted sigma
    
    # Apply multiple filters with slightly different parameters
    filt_real1, _ = gabor(gray, frequency=frequency, theta=theta * np.pi,
                         sigma_x=sigma, sigma_y=sigma, n_stds=mask)
    filt_real2, _ = gabor(gray, frequency=frequency * 1.1, theta=theta * np.pi,
                         sigma_x=sigma, sigma_y=sigma, n_stds=mask)
    filt_real3, _ = gabor(gray, frequency=frequency * 0.9, theta=theta * np.pi,
                         sigma_x=sigma, sigma_y=sigma, n_stds=mask)
    
    # Combine filter responses
    filt_real = (filt_real1 + filt_real2 + filt_real3) / 3
    return filt_real

def convert(filt_image):
    """Convert filter output for histogram analysis"""
    hist = filt_image.flatten()
    # Add small amount of noise to prevent identical values
    hist = hist + np.random.normal(0, 1e-6, hist.shape)
    hist2 = np.vstack(hist)
    return hist, hist2

def gaussian_curve(hist2):
    """Fit Gaussian mixture model to histogram"""
    try:
        # Initialize GMM with better parameters
        nmodes = 2
        GMModel = GaussianMixture(
            n_components=nmodes,
            covariance_type='full',
            random_state=42,
            reg_covar=1e-6,  # Add regularization
            n_init=5  # Multiple initialization attempts
        )
        
        # Fit model
        GMModel.fit(hist2)
        
        # Extract parameters
        mu1, mu2 = GMModel.means_.flatten()
        v1, v2 = np.diagonal(GMModel.covariances_).flatten()
        v11, v22 = np.sqrt(v1), np.sqrt(v2)
        
        # Ensure proper ordering (mu1 should be smaller than mu2)
        if mu1 > mu2:
            return mu2, mu1, v22, v11
        return mu1, mu2, v11, v22
        
    except Exception as e:
        print(f"Warning: Gaussian fitting failed ({str(e)}), using fallback")
        # Fallback to simple statistics
        values = hist2.flatten()
        mean = np.mean(values)
        std = np.std(values)
        return mean - std, mean + std, std, std

def process_image(image_path, frequency=0.6, theta=0.8):
    """Process an image with Gabor filter and visualize results"""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        print("Please place a valid image file (e.g., example3.jpg) in the same directory.")
        sys.exit(1)

    # Read and convert image to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        sys.exit(1)

    # Apply Gabor filter
    try:
        # Preprocess image
        img = cv2.GaussianBlur(img, (3, 3), 0)  # Remove noise
        
        cost, filtered = gabor_fit_func(img, frequency, theta)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
        # Filtered image
        plt.subplot(132)
        plt.title(f'Gabor Filter (f={frequency:.2f}, θ={theta:.2f}π)')
        plt.imshow(filtered, cmap='gray')
        plt.axis('off')
        
        # Histogram
        plt.subplot(133)
        plt.title(f'Response Histogram (Cost: {cost:.4f})')
        hist_values = filtered.flatten()
        plt.hist(hist_values, bins=50, density=True, alpha=0.7)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save results
        cv2.imwrite('gabor_filtered.png', (filtered * 255).astype(np.uint8))
        
        return cost, filtered
    except Exception as e:
        print(f"Error in Gabor processing: {str(e)}")
        return None, None

def main():
    # Look for both .jpg and .png files
    image_files = glob.glob('*.jpg') + glob.glob('*.png')
    
    if not image_files:
        print("No image files found in current directory.")
        print("Please place .jpg or .png files in the directory.")
        sys.exit(1)
        
    print("Available images:", image_files)
    image_path = input("Enter image name to process (or press Enter for first image): ").strip() or image_files[0]
    
    try:
        # Get user input with default values
        frequency = float(input('Enter frequency (default=0.6): ').strip() or 0.6)
        theta = float(input('Enter theta in π units (default=0.8): ').strip() or 0.8)
        
        # Validate input parameters
        if not (0.1 <= frequency <= 2.0):
            print("Warning: Frequency should be between 0.1 and 2.0, using default value 0.6")
            frequency = 0.6
        if not (0.0 <= theta <= 2.0):
            print("Warning: Theta should be between 0 and 2π, using default value 0.8")
            theta = 0.8
        
        # Process image
        cost, filtered = process_image(image_path, frequency, theta)
        if filtered is not None:
            print(f"\nProcessing complete:")
            print(f"- Cost function value: {cost:.4f}")
            print("- Results saved as 'gabor_filtered.png'")
            
            # Provide feedback on filter response
            if cost < 0.1:
                print("\nNote: Low cost value indicates weak texture separation.")
                print("Try adjusting frequency/theta or using a different image.")
        
    except ValueError as e:
        print("Error: Please enter valid numeric values for frequency and theta")
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()
