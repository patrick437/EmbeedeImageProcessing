from PIL import Image
import os
from glob import glob
from matplotlib import pyplot as plt
import cv2
import numpy as np

image_dir = 'c:/Users/Patrick/Documents/EmbeddedImageProcessing/Mini-Project/Mini_project_images/easy/*'
image_files = [file for file in glob(image_dir, recursive=True) if file.endswith(('.jpg', '.png'))]

def drawLine(img, x, y, color=[0, 255, 0], thickness=80):
    if len(x) == 0: 
        return
    
    lineParameters = np.polyfit(x, y, 1) 
    
    m = lineParameters[0]
    b = lineParameters[1]
    
    maxY = img.shape[0]
    maxX = img.shape[1]
    y1 = maxY
    x1 = int((y1 - b)/m)
    y2 = int((maxY/2)) + 60  # note: hardcoded, sets the length of the line to half the image height + 60 pixels
    x2 = int((y2 - b)/m)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Helper function - split the detected lines into left and right lines
def draw_lines(img, lines, color=[0, 255, 0], thickness=2):  # Reduced thickness to see individual lines
    if lines is None:
        return img
        
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, min_line_len, max_line_gap) 
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    print(f"Number of lines detected: {0 if lines is None else len(lines)}")
    
    # Draw all detected lines
    if lines is not None:
         # Filter lines based on angle
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
             # Check if line is vertical (90 ± 5 degrees)
            is_vertical = abs(angle - 90) < 5
            
            # Check if line is horizontal (0 or 180 ± 5 degrees)
            is_horizontal = angle < 5 or angle > 175
            
            if is_vertical or is_horizontal:
                filtered_lines.append(line)

        line_img = draw_lines(line_img, filtered_lines)
    
    return line_img

def preprocess_image(img):
    # Convert to HSV for better white line detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for white colors
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply the mask
    masked = cv2.bitwise_and(img, img, mask=white_mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    return blurred
def standardize_image(image, target_height=600, target_width=900):
    # Get current dimensions
    height, width = image.shape[:2]
    
    # Calculate scaling factors for both dimensions
    height_scale = target_height / height
    width_scale = target_width / width
    
    # Use the smaller scaling factor to maintain aspect ratio
    # This ensures the image fits within the target dimensions
    scale = min(height_scale, width_scale)
    
    # Calculate new dimensions
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Resize image
    standardized = cv2.resize(image, (new_width, new_height))
    
    # Create a black canvas of target size
    final_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate positioning to center the image
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

     # Place the resized image in the center of the canvas
    final_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = standardized
    
    return final_image

def process_parking_image(img):
    fig, axes = plt.subplots(3, 3, figsize=(12, 24))
    
    # 1. Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([170, 60, 255])
    
    # Create mask for white colors
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Convert the image to binary (0 and 255)
    _, binary = cv2.threshold(white_mask, 50, 255, cv2.THRESH_BINARY)
    
    axes[0][0].imshow(binary, cmap='gray')
    axes[0][0].set_title('Binary Image')
    axes[0][0].axis('off')
    
    # Get car blobs using opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    car_blobs = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=5)
    
    axes[0][1].imshow(car_blobs, cmap='gray')
    axes[0][1].set_title('Car Blobs')
    axes[0][1].axis('off')
    
    #Dilate to show the cars more clearly before subtraction
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    dilated = cv2.dilate(car_blobs, kernel_vertical, iterations=5)

    # Subtract car blobs from binary image
    lines_only = cv2.subtract(binary, dilated)
    
    axes[0][2].imshow(lines_only, cmap='gray')
    axes[0][2].set_title('Lines After Subtraction')
    axes[0][2].axis('off')
    
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (7,3))
    parking_area = cv2.dilate(lines_only, kernel_vertical, iterations=15)
    
    axes[1][0].imshow(dilated, cmap='gray')
    axes[1][0].set_title('Vertical Dilation')
    axes[1][0].axis('off')
    

    
    axes[1][1].imshow(cv2.cvtColor(parking_area, cv2.COLOR_BGR2RGB))
    axes[1][1].set_title('Final Line Detection')
    axes[1][1].axis('off')
    
    # Hide unused subplots
    axes[1][2].axis('off')
    axes[2][0].axis('off')
    axes[2][1].axis('off')
    axes[2][2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return parking_area



