import cv2
import pytesseract
import numpy as np

# Set the path to tesseract.exe (change if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = "D:/GHCLDATA/soil_moisture_analysis/data/2km_Jan21-Dec24_Sentinel-2_L2A-855842435060540-timelapse_030.jpg"

# Load the image
image = cv2.imread(image_path)  # Replace with your image file

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get image dimensions
height, width = gray.shape

# Crop the top-right corner (adjust values if needed)
x1, y1 = int(width * 0.75), 0   # 75% width from the left
x2, y2 = width, int(height * 0.2)  # Top 20% of the image
cropped = gray[y1:y2, x1:x2]

# Apply thresholding for better OCR
_, thresh = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY_INV)

# Extract text using Tesseract OCR
extracted_text = pytesseract.image_to_string(thresh, config="--psm 6")

# Print extracted text
print("Extracted Date:", extracted_text.strip())
