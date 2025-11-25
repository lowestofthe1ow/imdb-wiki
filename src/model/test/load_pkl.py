import os
import pandas as pd


def load_pkl(pkl_path, testing_set):
    result_dfs = []

    # Get all model results from .pkl files directory
    for filename in os.listdir(pkl_path):
        # Only read files ending in APPAREAL_results.pkl (or whatever testing set is)
        if filename.endswith(testing_set + "_results.pkl"):
            file_path = os.path.join(pkl_path, filename)
            df = pd.read_pickle(file_path)
            result_dfs.append(df)

    # Load all the .pkl files into one big DataFrame
    results = pd.concat(result_dfs, ignore_index=True)

    print("Model results loaded.")
    results.describe()

    return results
