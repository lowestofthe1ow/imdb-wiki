# csv_imdbwiki.py
# Creates a CSV file mapping the IMDB-WIKI images to their labels

# Parameters:
# --imdb-path   Path to directory containing IMDB dataset folder
# --wiki-path   Path to directory containing WIKI dataset folder

import argparse
import pandas as pd
import numpy as np
import os
import sys

from scipy.io import loadmat
from datetime import datetime
from dateutil.relativedelta import relativedelta
from imblearn.under_sampling import RandomUnderSampler

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imdb-path", type=str)
parser.add_argument("--wiki-path", type=str)
args = parser.parse_args()


def extract_metadata(root, path, var_name):
    """Extracts IMDB-WIKI data located at root/path

    Args:
        root (str): Path to directory containing the imdb_crop/ folder
        path (str): Path to the .mat file to read metadata from
        var_name (str): 'imdb' or 'wiki'
    """

    metadata = loadmat(path, mat_dtype=False)

    # Extract all needed variables
    dob = metadata[var_name]["dob"][0][0][0]
    photo_taken = metadata[var_name]["photo_taken"][0][0][0]
    full_path = np.array([path[0] for path in metadata[var_name]["full_path"][0][0][0]])

    # Create data frame
    df = pd.DataFrame(
        np.array([dob, photo_taken, full_path]).transpose(),
        columns=["dob", "photo_taken", "full_path"],
    )

    # Set file paths
    df["full_path"] = df["full_path"].apply(
        lambda x: os.path.join(root, f"{var_name}_crop", x)
    )

    # Convert birthday to numeric and subtract Unix epoch datenum
    df["dob"] = pd.to_numeric(df["dob"], errors="coerce")
    df = df[df["dob"] > 100000]  # Remove images with data corrupted for some reason
    df["dob"] = pd.to_datetime(df["dob"] - 719529, unit="D", errors="coerce")

    # Photo assumed to be taken on July 1 of the year (as per paper)
    df["photo_taken"] = pd.to_datetime(
        {"year": df["photo_taken"], "month": 7, "day": 1}
    )

    # Drop NA values
    df.dropna()

    # Some images on the WIKI dataset are broken for whatever reason
    df = df[df["photo_taken"].dt.month <= 12]
    df = df[df["photo_taken"].dt.month >= 1]
    df = df[df["dob"].dt.month <= 12]
    df = df[df["dob"].dt.month >= 1]

    # Get age and limit to [1, 100]
    df["age"] = df.apply(
        lambda x: relativedelta(x["photo_taken"], x["dob"]).years, axis=1
    )
    df = df[df["age"] > 0]
    df = df[df["age"] < 101]

    return df


# Extract metadata from both IMDB and WIKI datasets then concatenate
metadata_imdb = extract_metadata(
    args.imdb_path, os.path.join(args.imdb_path, "imdb_crop/imdb.mat"), "imdb"
)
metadata_wiki = extract_metadata(
    args.wiki_path, os.path.join(args.wiki_path, "wiki_crop/wiki.mat"), "wiki"
)
metadata = pd.concat([metadata_imdb, metadata_wiki])


# Downsample the dataset to balance age classes
# DEX conference paper simply discarded from the most populous age groups
print(
    f"Total number of images: {len(metadata)} ({len(metadata_imdb)} IMDB; {len(metadata_wiki)} WIKI)"
)
print("Downsampling...")

TOTAL_SAMPLES = 260282  # DEX paper used this many samples for training
sizes = metadata.groupby("age").size()

# Obtain target sizes for each class by decrementing from most populous age groups
while sizes.sum() > TOTAL_SAMPLES:
    sizes[sizes.idxmax()] -= 1

print(f"Downsampled size: {sizes.sum()}")

# Run a RandomUnderSampler to randomly delete samples based on target sizes
X = metadata.drop("age", axis=1)
y = metadata["age"]

rus = RandomUnderSampler(sampling_strategy=sizes.to_dict(), random_state=339)
X_resampled, y_resampled = rus.fit_resample(X, y)

X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled_df = pd.Series(y_resampled, name="age")

metadata_downsized = pd.concat([X_resampled_df, y_resampled_df], axis=1).sort_index()


# Write to CSV file
os.makedirs("data", exist_ok=True)
metadata_downsized.to_csv("data/imdbwiki.csv", index=False)
