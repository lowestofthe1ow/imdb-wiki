import argparse
import pandas as pd
import os

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--fairface-path", type=str)
args = parser.parse_args()

raw_train = pd.read_csv(os.path.join(args.fairface_path, "fairface_label_train.csv"))
raw_val = pd.read_csv(os.path.join(args.fairface_path, "fairface_label_val.csv"))


def mean_from_string(str):
    """Calculates the mean of an range string, e.g. the string "1-10" will return 5.5"""
    if "-" in str:
        first, second = str.split("-")
        return (int(first) + int(second)) / 2
    elif str == "more than 70":  # Why is it even formatted like this lmao
        return 70
    else:
        return int(str)


def clean(df):
    """Cleans up either the train or validation sets for FairFace"""
    return pd.DataFrame(
        {
            # Append FairFace directory to full_path
            "full_path": df["file"].apply(
                lambda x: os.path.join(args.fairface_path, x)
            ),
            # Convert age ranges into singular ages
            "age": df["age"].apply(lambda x: mean_from_string(x)),
            "race": df["race"],
            "gender": df["gender"],
        }
    )


train = clean(raw_train)
val = clean(raw_val)

os.makedirs("data", exist_ok=True)
train.to_csv("data/fairface_train.csv", index=False)
val.to_csv("data/fairface_val.csv", index=False)
