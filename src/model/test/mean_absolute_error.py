"""
Test inference against APPA-REAL or ChaLearn LAP 2015 testing set
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
import umap.umap_ as umap

from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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

transform = VAL_TEST_TRANSFORMS  # Image pre-transforms


def test(model: FinetuneDEX, image_path: str):
    """Performs forward passes through the model and saves embeddings"""
    with torch.no_grad():  # No need to calculate gradients during inference
        image = Image.open(image_path).convert("RGB")
        try:
            output, embeddings = model.forward(
                torch.unsqueeze(transform(image), 0).to("cuda"),
                capture_embeddings=True,  # Captures embeddings at the last FC layer
            )

            # Move data frrom CUDA device back to CPU (required)
            output, embeddings = output.to("cpu"), embeddings.to("cpu")

            # DEX softmax expected value
            prob = torch.nn.functional.softmax(output, dim=1)[0]
            age_range = torch.arange(101)  # [1, 2, ..., 101]
            estimated_age = (prob * age_range).sum()

            return estimated_age.item(), torch.flatten(embeddings)
        except Exception as e:
            print(f"Error opening `{image_path}`:\n{e}")


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

# Load the model
vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg16.classifier[6] = torch.nn.Linear(4096, 101)
checkpoint = torch.load(args.model_path)
model = FinetuneDEX(model=vgg16, criterion=None).to("cuda")
model.load_state_dict(checkpoint["state_dict"], strict=False)

model.eval()  # Switches model to evaluation mode (needed for inference)


# Get model outputs on testing set
# Series of tuples containing age estimates and embeddings
output = test_images["file_name"].progress_apply(lambda x: test(model, x))

# Split into the estimate and embeddings columns
test_images[["estimate", "embeddings"]] = pd.DataFrame(
    output.tolist(), index=test_images.index
)


# Use UMAP for dimensionality reduction
# TODO: Look into t-SNE
print("Saving image embeddings from model...")
embeddings_array = np.array(test_images["embeddings"].tolist())
mapper = umap.UMAP(n_components=2, random_state=339)
reduced_embeddings = mapper.fit_transform(embeddings_array)


# Plot reduced embeddings on a scatterplot
print("Saving to scatterplot...")
plt.figure(figsize=(8, 6))
plt.title("Embeddings")
sns.scatterplot(
    x=reduced_embeddings[:, 0],
    y=reduced_embeddings[:, 1],
    hue=test_images["apparent_age_avg"],
    alpha=0.8,
    palette="viridis",
)
plt.savefig(os.path.basename(args.model_path) + "_embeddings.png")


# Print MAE results (apparent age)
test_images["error_apparent"] = abs(
    test_images["estimate"] - test_images["apparent_age_avg"]
)
print(f"MAE (apparent) = {test_images["error_apparent"].mean()}")

# Print MAE results (real age, only for APPA-REAL)
if args.testing_set == "APPAREAL":
    test_images["error_real"] = abs(test_images["estimate"] - test_images["real_age"])
    print(f"MAE (real) = {test_images["error_real"].mean()}")
