import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics.pairwise import cosine_similarity

from src.model.test.load_pkl import load_pkl

parser = argparse.ArgumentParser()
parser.add_argument("--pkl-path", default="pkl", type=str)
parser.add_argument("--output-path", default="cosine_sim", type=str)
# Note only APPAREAL has race/gender data
# (I haven't added a fallback for CHALEARN)
parser.add_argument("--testing-set", default="APPAREAL", type=str)
args = parser.parse_args()


def mean_embeddings(embeddings):
    """Gets the mean of a Series of PyTorch Tensors, each representing image embeddings"""
    return torch.mean(torch.stack(embeddings.tolist(), dim=0), dim=0).numpy()


results = load_pkl(args.pkl_path, args.testing_set)

# Needed when joining with ground-truth values
results["estimate_rounded"] = results["estimate"].round().astype("Int64")
results["apparent_age_avg_rounded"] = (
    results["apparent_age_avg"].round().astype("Int64")
)

ground_truth_embeddings = (
    results.groupby(["model", "race", "gender", "apparent_age_avg_rounded"])
    .agg({"embeddings": mean_embeddings})
    .reset_index()
)

estimated_embeddings = (
    results.groupby(["model", "race", "gender", "estimate_rounded"])
    .agg({"embeddings": mean_embeddings})
    .reset_index()
)[
    ["estimate_rounded", "embeddings"]
]  # Discard all other columns to avoid duplicates when merging

cosine_sim = pd.merge(
    ground_truth_embeddings,
    estimated_embeddings,
    left_on="apparent_age_avg_rounded",
    right_on="estimate_rounded",
    how="inner",
    suffixes=("_real", "_estimate"),
)

print("Merged df for ground-truth and estimated image embeddings")
print(cosine_sim.info())

cosine_sim["cosine_sim"] = cosine_sim.apply(
    lambda x: cosine_similarity(
        # Reshape since sklearn's cosine_similarity requires 2D
        x["embeddings_real"].reshape(1, -1),
        x["embeddings_estimate"].reshape(1, -1),
    )[0][0],
    axis=1,
)

os.makedirs(args.output_path, exist_ok=True)

print("Cosine similarity df ready:")
print(cosine_sim.info())
print(f"Writing to file ")
cosine_sim.to_pickle(os.path.join(args.output_path, "cosine_sim.pkl"))
