"""
Test inference against APPA REAL testing set
"""
import pandas as pd
import torch
from torchvision import models
from PIL import Image
from src.modules.finetune_mvl import FinetuneMVL
from src.modules.finetune_dex import FinetuneDEX
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
import argparse
from src.datasets.constants import TRANSFORM_TO_SIZE, IMAGENET_MEAN, IMAGENET_STD
from tqdm import tqdm

tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str)
args = parser.parse_args()

vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)

# "When training for classification, the output layer is adapted to 101 output neurons
# corresponding to natural numbers from 0 to 100..."
vgg16.classifier[6] = torch.nn.Linear(4096, 101)

# .ckpt file
CHECKPOINT = args.model_path

transform = transforms.Compose(
    [
        transforms.Resize(TRANSFORM_TO_SIZE),
        # TODO: Seems to perform better without the center crop?
        # transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Used only in DEX
layer_params = [
    {
        # Flatten an iterator that goes over all the classifier layers in VGG16
        "params": [p for i in range(6) for p in vgg16.classifier[i].parameters()],
        "lr": 0.0001,
    },
    {"params": list(vgg16.classifier[6].parameters()), "lr": 0.001},
]


model = FinetuneMVL.load_from_checkpoint(
    CHECKPOINT, model=vgg16, batch_size=128  # , layer_params=layer_params
)

model.eval()


def test(model: FinetuneDEX | FinetuneMVL, image_path: str):
    with torch.no_grad():  # No need to calculate gradients during inference
        image = Image.open(image_path).convert("RGB")
        try:
            output = model(torch.unsqueeze(transform(image), 0).to("cuda")).to("cpu")
            prob = torch.nn.functional.softmax(output, dim=1)[0]
            age_range = torch.arange(1, 102)  # [1, 2, ..., 101]
            estimated_age = (prob * age_range).sum() # weighted sum of ages
        except Exception as e:
            print(f"Error opening `{image_path}`:\n{e}")
    return estimated_age.item()


# Ready the test set from APPA-REAL
test_images = pd.read_csv("data/appa-real-release/gt_avg_test.csv")
test_images["file_name"] = "data/appa-real-release/test/" + test_images["file_name"]
test_images["estimate"] = test_images["file_name"].progress_apply(lambda x: test(model, x))

# MAE
test_images["error_apparent"] = abs(
    test_images["estimate"] - test_images["apparent_age_avg"]
)
test_images["error_real"] = abs(
    test_images["estimate"] - test_images["real_age"]
)

print(test_images.head(5))

print(f"MAE (apparent) = {test_images["error_apparent"].mean()}")
print(f"MAE (real) = {test_images["error_real"].mean()}")
