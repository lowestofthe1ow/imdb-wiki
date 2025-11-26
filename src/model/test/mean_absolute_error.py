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
from torchvision import models
from torchvision.models import VGG16_Weights
from tqdm import tqdm

from src.model.test.extract_from_caffe import get_state_dict
from src.model.test.test_on_set import test_on_set
from src.modules.finetune_dex import FinetuneDEX

tqdm.pandas(leave=False)

parser = argparse.ArgumentParser()
parser.add_argument("--testing-set", default="APPAREAL", type=str)
parser.add_argument("--model-path", type=str)
parser.add_argument("--output-plot-path", default="embeddings", type=str)
parser.add_argument("--output-pkl-path", default="pkl", type=str)
parser.add_argument(
    "--model-captions-path", default="checkpoints/checkpoint_captions.csv", type=str
)
args = parser.parse_args()

OUTPUT_FIGURE_SIZE = (8, 6)  # Figure size in inches


def get_caption(model_path):
    """Converts model filename to a more descriptive caption.
    By default, uses CSV file in /checkpoints
    """
    captions = pd.read_csv(args.model_captions_path, skipinitialspace=True)
    value = captions.loc[captions["model_path"] == model_path, "caption"]
    return value.item()


"""
For the APPA-REAL test set, this should produce a DataFrame with:

 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   file_name         1978 non-null   object
 1   num_ratings       1978 non-null   int64
 2   apparent_age_avg  1978 non-null   float64
 3   apparent_age_std  1978 non-null   float64
 4   real_age          1978 non-null   int64
 5   race              1978 non-null   object
 6   gender            1978 non-null   object
 7   estimate          1978 non-null   float64
 8   embeddings        1978 non-null   object
"""
results = test_on_set(args.testing_set, args.model_path)

model_caption = get_caption(args.model_path)
results["model"] = model_caption
results["testing_set"] = args.testing_set

# We write the testing results to a .pkl file for easy storage
# Filename example: Original Caffe model on IMDB-WIKI + ChaLearn LAP_test on APPAREAL_results.pkl
out_filename = model_caption + f"_test on {args.testing_set}_results.pkl"
out_fullpath = os.path.join(args.output_pkl_path, out_filename)
os.makedirs(args.output_pkl_path, exist_ok=True)
results.to_pickle(out_fullpath)

print(f"Wrote to .pkl file at {out_fullpath}")

# MAE results (apparent age)
mae_apparent = abs(results["estimate"] - results["apparent_age_avg"]).mean()
print(f"MAE (apparent) = {mae_apparent}")

# MAE results (real age, only for APPA-REAL)
if args.testing_set == "APPAREAL":
    mae_real = abs(results["estimate"] - results["real_age"]).mean()
    print(f"MAE (real) = {mae_real}")

epsilon = (
    1 - np.exp(-1 * mae_apparent**2 / (2 * (results["apparent_age_std"] ** 2))).mean()
)
print(f"Epsilon = {epsilon}")

# Use UMAP for dimensionality reduction
# TODO: Look into t-SNE maybe? But I think UMAP is ok
print("Saving image embeddings from model...")
embeddings_array = np.array(results["embeddings"].tolist())
mapper = umap.UMAP(n_components=2)  # 2D scatterplot
reduced_embeddings = mapper.fit_transform(embeddings_array)

# Plot reduced embeddings on a scatterplot
print("Saving to scatterplot...")
plt.figure(figsize=OUTPUT_FIGURE_SIZE)
plt.title(f"UMAP embeddings\n{model_caption}, testing on {args.testing_set}")
sns.scatterplot(
    x=reduced_embeddings[:, 0],
    y=reduced_embeddings[:, 1],
    hue=results["apparent_age_avg"],
    alpha=0.8,
    palette="viridis",
)
out_filename = os.path.basename(args.model_path) + f"_{args.testing_set}_embeddings.png"

os.makedirs(args.output_plot_path, exist_ok=True)
plt.savefig(os.path.join(args.output_plot_path, out_filename))
