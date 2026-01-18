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

from pathlib import Path
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
parser.add_argument("--output-plot-path", default="results", type=str)
parser.add_argument("--output-pkl-path", default="results", type=str)
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
results_dir = Path("results") / args.testing_set / os.path.basename(args.model_path)
results = test_on_set(args.testing_set, args.model_path, results_dir)

model_caption = get_caption(args.model_path)
results["model"] = model_caption
results["testing_set"] = args.testing_set

# We write the testing results to a .pkl file for easy storage
# Filename example: Original Caffe model on IMDB-WIKI + ChaLearn LAP_test on APPAREAL_results.pkl
model_basename = os.path.basename(args.model_path)
out_folder = os.path.join(args.output_plot_path, args.testing_set, model_basename)
os.makedirs(out_folder, exist_ok=True)

out_filename = f"{model_caption}_test on {args.testing_set}_results.pkl"
out_fullpath = os.path.join(out_folder, out_filename)
results.to_pickle(out_fullpath)

# Support old path for backward compatibility with other scripts (e.g. cosine_sim.py)
# TODO: Remove this in the future
os.makedirs("pkl", exist_ok=True)
results.to_pickle(os.path.join("pkl", out_filename))

print(f"Wrote to .pkl file at {out_fullpath}")

def compute_mae(results: pd.DataFrame, age_column: str) -> float:
    """Computes Mean Absolute Error (MAE) based on testing set"""
    return abs(results["estimate"] - results[age_column]).mean()

def compute_epsilon(mae: float, std_column: pd.Series) -> float:
    """Computes epsilon value based on MAE and standard deviation column"""
    return (1 - np.exp(-1 * mae**2 / (2 * (std_column ** 2))).mean())

match args.testing_set:
    case "CHALEARN":
        mae_apparent = compute_mae(results, "apparent_age_avg")
        epsilon = compute_epsilon(mae_apparent, results["apparent_age_std"])
        print(f"MAE (apparent) = {mae_apparent}")
        print(f"Epsilon = {epsilon}")
        umap_hue = results["apparent_age_avg"]
    case "APPAREAL":
        mae_apparent = compute_mae(results, "apparent_age_avg")
        epsilon = compute_epsilon(mae_apparent, results["apparent_age_std"])
        mae_real = compute_mae(results, "real_age")
        umap_hue = results["real_age"]
        print(f"MAE (apparent) = {mae_apparent}")
        print(f"MAE (real) = {mae_real}")
        print(f"Epsilon = {epsilon}")
    case "FILIPINO_FACES":
        mae_real = compute_mae(results, "age")
        umap_hue = results["age"]
        print(f"MAE (real) = {mae_real}")


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
    hue=umap_hue,
    alpha=0.8,
    palette="viridis",
)

out_filename = f"{model_basename}_{args.testing_set}_embeddings.png"
plt.savefig(os.path.join(out_folder, out_filename))
