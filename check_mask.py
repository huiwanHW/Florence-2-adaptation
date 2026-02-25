import cv2
import numpy as np

# Path to the mask image
mask_path = 'train/Segmentation/Three/FUGC_890/part1/labeled_data/labels/0015.png'

# Read the image in grayscale
img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    print('Image shape:', img.shape)
    print('Min value:', np.min(img))
    print('Max value:', np.max(img))
    print('Unique values:', np.unique(img))
    print('First 10x10 pixels:')
    print(img[:10, :10])
else:
    print('Failed to read image')
