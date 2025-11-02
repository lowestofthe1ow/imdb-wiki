import pandas as pd
import numpy as np
import os
import sys

from scipy.io import loadmat
from datetime import datetime
from dateutil.relativedelta import relativedelta
from imblearn.under_sampling import RandomUnderSampler


def extract_metadata(path, var_name, path_prefix):
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
        lambda x: os.path.join(path_prefix, f"{var_name}_crop", x)
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


metadata_imdb = extract_metadata(
    os.path.join(sys.argv[1], "imdb_crop/imdb.mat"), "imdb", sys.argv[1]
)
metadata_wiki = extract_metadata(
    os.path.join(sys.argv[2], "wiki_crop/wiki.mat"), "wiki", sys.argv[2]
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

metadata = pd.concat([metadata_imdb, metadata_wiki])

print(
    f"Total number of images: {len(metadata)} ({len(metadata_imdb)} IMDB; {len(metadata_wiki)} WIKI)"
)
print("Downsampling...")

# I'm aware this is questionable but well... calculates the target sizes for each age class
TOTAL_SAMPLES = 260282  # DEX paper used this many samples for training
sizes = metadata.groupby("age").size()

# Discard from largest class every time
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
