"""
Data from ChaLearn LAP 2015 should be in a single directory:

[--input-path directory]
├── raw
│   ├── test/
│   ├── train/
│   └── validation/
├── test.csv
├── train.csv
└── validation.csv

https://drive.google.com/drive/folders/1fe73NhI7OqmjnB-G6pr3xK8S-_XXgD6j?usp=sharing
"""

import argparse
import os

from tqdm import tqdm
from src.datasets.preprocess.crop_to_face import image_file_extensions, crop_to_face

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", type=str)
parser.add_argument("--output-path", type=str)
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

for subfolder in ["test", "train", "validation"]:
    for filename in tqdm(os.listdir(os.path.join(args.input_path, subfolder))):
        if filename.lower().endswith(image_file_extensions):
            crop_to_face(
                os.path.join(args.input_path, subfolder, filename),
                0.4,
                os.path.join(args.output_path, subfolder, filename),
            )
