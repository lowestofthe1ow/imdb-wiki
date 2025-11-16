import os
import cv2
import torch
import numpy as np

from tqdm import tqdm
from facenet_pytorch import MTCNN, extract_face
from PIL import Image

image_file_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")

mtcnn = MTCNN(
    keep_all=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def crop_to_face(input_path, margin_percentage, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.open(input_path).convert("RGB")
    img_w, img_h = img.size

    # Detect faces and get confidence values for each
    faces, confidence = mtcnn.detect(img)

    # If detector can't find a face, give up and just copy the input image
    if faces is None:
        print("No face detected, copying input image...")
        img.save(output_path)
        return

    x_min, y_min, x_max, y_max = faces[np.argmax(confidence)].tolist()

    face_width = x_max - x_min
    face_height = y_max - y_min

    # Calculate 40% margins
    pixel_padding_x = face_width * margin_percentage
    pixel_padding_y = face_height * margin_percentage

    # Get cropped boundaries with added margin
    cropped_x_min = max(0, int(x_min - pixel_padding_x))
    cropped_y_min = max(0, int(y_min - pixel_padding_y))
    cropped_x_max = min(img_w, int(x_max + pixel_padding_x))
    cropped_y_max = min(img_h, int(y_max + pixel_padding_y))

    cropped_img = img.crop((cropped_x_min, cropped_y_min, cropped_x_max, cropped_y_max))

    cropped_img.save(output_path)
