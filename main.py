import cv2
import pandas as pd

model = 'dex_imdb_wiki.caffemodel'
proto = 'age.prototxt'

# The paper did not give specific mean values for the network
# I found these from looking up "ImageNet mean values" and they seem to work
mean = (0.485, 0.456, 0.406)

# Paper used a MATLAB script for extracting faces from an image
# I'm too lazy to set that up right now so manual cropping it is
# Input layer has to be 224x224
input_size = (224, 224)

age_net = cv2.dnn.readNetFromCaffe(proto, model)

def predict_age(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f'Error loading file at {image_path}')
        exit()

    blob = cv2.dnn.blobFromImage(
        image,
        # If we normalize to 255 it starts saying I'm 40 years old so it stays 1.0
        scalefactor = 1.0, # 1.0 / 255.0,
        size = input_size,
        mean = mean,
        swapRB = False, # I'm pretty sure we dont need to go from RGB to BGR
        crop = False
    )

    age_net.setInput(blob)

    # Do a single forward pass through the network
    # Result is a softmax vector showing probability distribution for ages 1 to 101
    predictions = age_net.forward()

    age_range = range(1, 102) # [1, 2, ..., 101]

    # Get weighted sum of ages to arrive at a single age estimate
    estimated_age = (predictions * age_range).sum()

    return estimated_age

images = pd.DataFrame({'files': ['test1.jpg', 'test2.jpg']})
images['predicted_age'] = images['files'].apply(predict_age)

print(images)
