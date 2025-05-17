import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_gabor_filters():
    """Create a bank of Gabor filters with different parameters"""
    filters = []
    ksize = 31
    # Optimized parameter ranges based on empirical testing
    sigma_list = [2.0, 3.0, 4.0]
    theta_list = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]
    lambda_list = [np.pi/2, np.pi/3, np.pi/4]
    gamma_list = [0.3, 0.5, 0.7]
    psi_list = [0, np.pi/2]  # Phase offset
    
    for sigma in sigma_list:
        for theta in theta_list:
            for lambda_ in lambda_list:
                for gamma in gamma_list:
                    for psi in psi_list:
                        try:
                            kernel = cv2.getGaborKernel(
                                (ksize, ksize), 
                                sigma, 
                                theta, 
                                lambda_,
                                gamma,
                                psi,
                                ktype=cv2.CV_32F
                            )
                            # Avoid division by zero
                            kernel_sum = np.sum(np.abs(kernel))
                            if kernel_sum > 1e-10:  # Numerical stability threshold
                                kernel /= kernel_sum
                                filters.append({
                                    'kernel': kernel,
                                    'params': {
                                        'theta': theta,
                                        'lambda': lambda_,
                                        'sigma': sigma,
                                        'gamma': gamma,
                                        'psi': psi
                                    }
                                })
                        except Exception as e:
                            print(f"Warning: Failed to create filter with params: σ={sigma}, θ={theta}, λ={lambda_}, γ={gamma}, ψ={psi}")
                            continue
    
    if not filters:
        raise ValueError("Failed to create any valid Gabor filters")
    
    return filters

def apply_gabor_filter(img, kernel):
    """Apply Gabor filter with enhanced error handling"""
    try:
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        filtered = np.abs(filtered)  # Get magnitude
        
        # Avoid division by zero in normalization
        if filtered.max() - filtered.min() > 1e-10:
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        else:
            # If the response is too uniform, return zeros
            filtered = np.zeros_like(filtered)
            
        return filtered.astype(np.uint8)
    except Exception as e:
        print(f"Warning: Filter application failed: {str(e)}")
        return np.zeros_like(img, dtype=np.uint8)

def segment_texture(responses):
    """Perform texture segmentation using K-means clustering"""
    try:
        # Reshape responses for clustering
        h, w = responses[0].shape
        features = np.column_stack([resp.ravel() for resp in responses])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine optimal number of clusters (2-5)
        best_score = float('-inf')
        best_n_clusters = 2
        best_labels = None
        
        for n_clusters in range(2, 6):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            score = kmeans.score(features_scaled)
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
        
        # Reshape labels back to image dimensions
        segmentation = best_labels.reshape(h, w)
        
        return segmentation, best_n_clusters
    except Exception as e:
        print(f"Warning: Segmentation failed: {str(e)}")
        return None, None

def process_image(image_path):
    """Process image with enhanced Gabor filter bank and clustering"""
    # Read and validate image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Check minimum image dimensions
    if img.shape[0] < 32 or img.shape[1] < 32:
        raise ValueError("Image dimensions too small, minimum 32x32 required")
    
    # Normalize and enhance image
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Create and apply Gabor filter bank
    filters = create_gabor_filters()
    responses = []
    filter_params = []
    
    for filter_data in filters:
        kernel = filter_data['kernel']
        params = filter_data['params']
        
        # Apply filter
        response = apply_gabor_filter(img, kernel)
        
        # Only keep meaningful responses
        if np.std(response) > 1.0:  # Threshold for meaningful response
            responses.append(response)
            filter_params.append(params)
    
    if not responses:
        raise ValueError("No meaningful filter responses generated")
    
    # Get best response based on energy
    response_energies = [np.sum(resp ** 2) for resp in responses]
    best_idx = np.argmax(response_energies)
    best_response = responses[best_idx]
    best_params = filter_params[best_idx]
    
    # Perform texture segmentation
    segmentation, n_clusters = segment_texture(responses)
    
    # Calculate texture features safely
    hist = cv2.calcHist([best_response], [0], None, [256], [0, 256])
    hist = hist.ravel()
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    
    # Calculate texture features with error handling
    features = {
        'mean': float(np.mean(best_response)),
        'std': float(np.std(best_response)),
        'energy': float(np.sum(best_response ** 2) / (best_response.size)),
        'entropy': float(-np.sum(hist * np.log2(hist + 1e-7)))
    }
    
    # Visualization
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(231)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    # Best filter response
    plt.subplot(232)
    plt.title(f'Best Gabor Response\nθ={best_params["theta"]/np.pi:.2f}π\nλ={best_params["lambda"]:.2f}')
    plt.imshow(best_response, cmap='gray')
    plt.axis('off')
    
    # Segmentation result
    if segmentation is not None:
        plt.subplot(233)
        plt.title(f'Texture Segmentation\n{n_clusters} clusters')
        plt.imshow(segmentation, cmap='nipy_spectral')
        plt.axis('off')
    
    # Show top 2 additional responses
    sorted_indices = np.argsort(response_energies)[-3:-1]
    for idx, resp_idx in enumerate(sorted_indices):
        plt.subplot(234 + idx)
        plt.title(f'Filter Response {idx+2}\nθ={filter_params[resp_idx]["theta"]/np.pi:.2f}π\nλ={filter_params[resp_idx]["lambda"]:.2f}')
        plt.imshow(responses[resp_idx], cmap='gray')
        plt.axis('off')
    
    # Histogram with features
    plt.subplot(236)
    plt.title('Response Histogram\n' + '\n'.join([f'{k}: {v:.2f}' for k, v in features.items()]))
    plt.bar(range(256), hist)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    cv2.imwrite('gabor_response.png', best_response)
    if segmentation is not None:
        cv2.imwrite('texture_segmentation.png', (segmentation * 255 / (n_clusters - 1)).astype(np.uint8))
    
    return best_response, hist, features, segmentation if segmentation is not None else None

if __name__ == "__main__":
    # Look for images in current directory
    import glob
    import sys
    
    try:
        images = glob.glob('*.jpg') + glob.glob('*.png')
        
        if not images:
            print("No image files found in current directory")
            sys.exit(1)
            
        print("Available images:", images)
        image_path = input("Enter image name to process (or press Enter for first image): ").strip() or images[0]
        
        response, hist, features, segmentation = process_image(image_path)
        
        print("\nTexture Features:")
        for name, value in features.items():
            print(f"{name}: {value:.2f}")
            
        if segmentation is not None:
            print(f"\nTexture segmentation completed with {len(np.unique(segmentation))} regions")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 