import sys
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

sys.path.append(r'D:\OneDrive\Codes\vsc_py\Tools')
from read_tools import *

image_path = r"D:\School\pic\s"

if os.path.isfile(image_path):
    image_paths = [image_path]
elif os.path.isdir(image_path):
    image_paths = [x[1] for x in find_files(image_path, file_extension='tif')]

# Dictionary to store ratios for each group
group_ratios = defaultdict(list)

for e in image_paths:
    # Load image in grayscale
    image = cv2.imread(e, cv2.IMREAD_GRAYSCALE)
    
    # Ensure image is loaded
    if image is None:
        print('Could not open or find the image:', e)
        continue

    # Apply pseudocolor mapping
    pseudo_color_image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    
    # Save the pseudocolor image
    output_path = rf"D:\School\pic\s\{os.path.splitext(os.path.basename(e))[0]}_2.jpg"
    cv2.imwrite(output_path, pseudo_color_image)
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(pseudo_color_image, cv2.COLOR_BGR2HSV)

    # Define HSV range for yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Define HSV range for red
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    # Create masks for yellow and red regions
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Calculate areas of yellow and red regions
    yellow_area = np.sum(yellow_mask > 0)
    red_area = np.sum(red_mask > 0)

    # Compute the ratio of yellow area to red area
    if red_area > 0:
        ratio = yellow_area / red_area
    else:
        ratio = float('inf')  # Handle case when there is no red area
    print(os.path.basename(e),ratio)
    # Get the group name from the filename
    group_name = os.path.basename(e).split('_')[0]

    # Store the ratio in the dictionary under the corresponding group
    group_ratios[group_name].append(ratio)

# Calculate and print average ratio for each group
for group_name, ratios in group_ratios.items():
    average_ratio = np.mean(ratios)
    print(f'Group: {group_name}, Average Yellow to Red Area Ratio: {average_ratio}')
