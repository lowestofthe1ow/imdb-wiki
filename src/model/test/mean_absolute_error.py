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
        transforms.Resize((256, 256)),
        # TODO: Seems to perform better without the center crop?
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


def test(model, image_path):
    with torch.no_grad():  # No need to calculate gradients during inference
        image = Image.open(image_path).convert("RGB")
        output = model(torch.unsqueeze(transform(image), 0).to("cuda")).to("cpu")
        prob = torch.nn.functional.softmax(output, dim=1)[0]
        age_range = torch.arange(1, 102)  # [1, 2, ..., 101]

        # Get weighted sum of ages to arrive at a single age estimate
        estimated_age = (prob * age_range).sum()
    return estimated_age.item()


# Ready the test set from APPA-REAL
test_images = pd.read_csv("data/appa-real-release/gt_avg_test.csv")
test_images["file_name"] = "data/appa-real-release/test/" + test_images["file_name"]
test_images["estimate"] = test_images["file_name"].apply(lambda x: test(model, x))

# MAE
test_images["error_apparent"] = abs(
    test_images["estimate"] - test_images["apparent_age_avg"]
)
test_images["error_real"] = abs(test_images["estimate"] - test_images["real_age"])

print(test_images.head(5))

print(f"MAE (apparent) = {test_images["error_apparent"].mean()}")
print(f"MAE (real) = {test_images["error_real"].mean()}")
