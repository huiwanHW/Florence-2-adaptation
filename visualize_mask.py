import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the mask image
mask_path = 'docker_output/Segmentation/Three/FUGC_890/part3/labels/0006.png'

# Read the image in grayscale
img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    print('Original mask statistics:')
    print('Image shape:', img.shape)
    print('Min value:', np.min(img))
    print('Max value:', np.max(img))
    print('Unique values:', np.unique(img))
    print()
    
    # Create a scaled version for visualization (0-255 range)
    scaled_mask = (img / np.max(img) * 255).astype(np.uint8)
    
    print('Scaled mask statistics:')
    print('Min value:', np.min(scaled_mask))
    print('Max value:', np.max(scaled_mask))
    print('Unique values:', np.unique(scaled_mask))
    print()
    
    # Create a color-coded version
    color_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    color_mask[img == 1] = [0, 255, 0]  # Class 1: Green
    color_mask[img == 2] = [0, 0, 255]  # Class 2: Red
    
    # Save the visualizations
    cv2.imwrite('mask_original.png', img)
    cv2.imwrite('mask_scaled.png', scaled_mask)
    cv2.imwrite('mask_color.png', color_mask)
    
    print('Visualizations saved:')
    print('1. mask_original.png - Original mask (appears black)')
    print('2. mask_scaled.png - Scaled to 0-255 (visible)')
    print('3. mask_color.png - Color-coded (green=class 1, red=class 2)')
else:
    print('Failed to read image')
