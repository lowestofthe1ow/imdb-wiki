import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch

from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from src.model.test.load_pkl import load_pkl

parser = argparse.ArgumentParser()
parser.add_argument("--pkl-path", default="pkl", type=str)
parser.add_argument("--output-path", default="cosine_sim", type=str)
# Note only APPAREAL has race/gender data
# (I haven't added a fallback for CHALEARN)
parser.add_argument("--testing-set", default="APPAREAL", type=str)
args = parser.parse_args()


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# Reads all .pkl files from the pkl/ directory and stacks them into one big df
raw_results = load_pkl(args.pkl_path, args.testing_set)

# All of the data is gonna be for the same testing set anyway, might as well
raw_results = raw_results.drop("testing_set", axis=1)

print("----------")


# (1) Race and gender statistics
race_gender_stats = raw_results  # Work with a clean copy of the df

print("Statistics by race and gender (across all models):")

race_gender_stats["apparent_real_diff"] = (
    # We want to see differences between apparent and real age per race/gender
    race_gender_stats["apparent_age_avg"]
    - race_gender_stats["real_age"]
)

race_gender_stats = (
    race_gender_stats.groupby(["race", "gender"])
    .mean(numeric_only=True)
    .sort_values(by=["apparent_real_diff"], ascending=False),
)

print(race_gender_stats)
print("----------")


def anova_race_gender(group):
    unique_races = group["race"].unique()
    unique_genders = group["gender"].unique()

    subgroups = []

    # Get all race-gender subgroups in the group (in this case, per model)
    for race in unique_races:
        for gender in unique_genders:
            subgroups.append(
                group[(group["race"] == race) & (group["gender"] == gender)][
                    "mae_apparent"
                ]
            )

    F, p = stats.f_oneway(*subgroups)
    return pd.Series({"F-statistic": F, "p-value": p})


# (2) Performance metrics
summarized_results = raw_results  # Work with a clean copy of the df

summarized_results["mae_real"] = abs(
    summarized_results["estimate"] - summarized_results["real_age"]
)

summarized_results["mae_apparent"] = abs(
    summarized_results["estimate"] - summarized_results["apparent_age_avg"]
)

# See paper for epsilon-error calculation
summarized_results["epsilon"] = 1 - np.exp(
    -1
    * summarized_results["mae_apparent"] ** 2
    / (2 * (summarized_results["apparent_age_std"] ** 2))
)

# We only care about these... for now
summarized_results = summarized_results[
    ["model", "race", "gender", "mae_apparent", "mae_real", "epsilon"]
]

# ANOVA lets us measure variances between each group
anova_results = summarized_results.groupby("model").apply(anova_race_gender)
summarized_results = summarized_results.merge(anova_results, on="model", how="left")

summarized_results_grouped = summarized_results.groupby(
    ["model", "race", "gender"]
).mean(numeric_only=True)

# Sort groups by apparent age MAE
summarized_results_sorted = summarized_results_grouped.reset_index().sort_values(
    by=["model", "mae_apparent"], ascending=False
)

print(summarized_results_sorted)
