import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps
import io
import numpy as np
import cv2
import pytesseract

# Load the trained model
class_names = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 
               'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=20, bias=True)
model.load_state_dict(torch.load('cnn_model1.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Define a function to predict the class
def predict(image, model, transform):
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Optional: Specify the path to the Tesseract executable if it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'path_to_your_tesseract_executable'

# Function to preprocess the image for better OCR results
def preprocess_image(image):
    gray_image = image.convert('L')  # Convert to grayscale
    np_image = np.array(gray_image)
    _, binary_image = cv2.threshold(np_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return Image.fromarray(binary_image)

# Function to use Tesseract OCR for recognizing Javanese characters
def ocr_javanese(image):
    # Tesseract configuration for Javanese script recognition
    custom_config = r'--oem 3 --psm 6 -l jav'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

# Streamlit app
st.title("Aksara Jawa Detection")

# Camera input
image_data = st.camera_input("Take a picture")

if image_data is not None:
    # Load the image
    image = Image.open(io.BytesIO(image_data.getvalue()))
    
    # Display the image
    st.image(image, caption='Captured Image', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Recognize Javanese characters using Tesseract OCR
    recognized_text = ocr_javanese(preprocessed_image)
    
    # Display the recognized text
    st.write(f"Recognized Text: {recognized_text}")
