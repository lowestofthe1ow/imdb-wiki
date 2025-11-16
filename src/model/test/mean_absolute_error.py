"""
Test inference against APPA REAL testing set
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from torchvision import models
from torchvision.models import VGG16_Weights
from tqdm import tqdm

from src.datasets.constants import VAL_TEST_TRANSFORMS
from src.modules.finetune_dex import FinetuneDEX

tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument("--testing-set", default="APPAREAL", type=str)
parser.add_argument("--model-path", type=str)
args = parser.parse_args()


vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg16.classifier[6] = torch.nn.Linear(4096, 101)


# Get testing set from either APPA-REAL or ChaLearn LAP 2015
test_images = None
match args.testing_set:
    case "APPAREAL":
        test_images = pd.read_csv("data/appa-real-release/gt_avg_test.csv")
        test_images["file_name"] = (
            "data/appa-real-release/test/"
            + test_images["file_name"]
            + "_face.jpg"  # ..._face.jpg is the cropped face image
        )
    case "CHALEARN":
        test_images = pd.read_csv("data/chalearn_2015/test.csv")
        test_images["file_name"] = "data/chalearn_2015/test/" + test_images["file_name"]


checkpoint = torch.load(args.model_path)
model = FinetuneDEX(model=vgg16, criterion=None).to("cuda")
model.load_state_dict(checkpoint["state_dict"], strict=False)

model.eval()

transform = VAL_TEST_TRANSFORMS


def test(model: FinetuneDEX, image_path: str):
    with torch.no_grad():  # No need to calculate gradients during inference
        image = Image.open(image_path).convert("RGB")
        try:
            output = model(torch.unsqueeze(transform(image), 0).to("cuda")).to("cpu")
            prob = torch.nn.functional.softmax(output, dim=1)[0]
            age_range = torch.arange(101)  # [1, 2, ..., 101]
            estimated_age = (prob * age_range).sum()  # weighted sum of ages
            return estimated_age.item()
        except Exception as e:
            print(f"Error opening `{image_path}`:\n{e}")


test_images["estimate"] = test_images["file_name"].progress_apply(
    lambda x: test(model, x)
)

# MAE
test_images["error_apparent"] = abs(
    test_images["estimate"] - test_images["apparent_age_avg"]
)
print(f"MAE (apparent) = {test_images["error_apparent"].mean()}")

if args.testing_set == "APPAREAL":
    test_images["error_real"] = abs(test_images["estimate"] - test_images["real_age"])
    print(f"MAE (real) = {test_images["error_real"].mean()}")

# print(test_images.head(5))
