import os
import glob
import pandas as pd

def load_data(folder="."):
    """
    Finds, loads, and concatenates all CSV files inside the specified folder.

    Args:
        folder (str): The folder path containing CSV files. Defaults to '.'.

    Returns:
        pd.DataFrame: Concatenated DataFrame.

    Raises:
        FileNotFoundError: If no CSV files are found in the specified folder.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in directory '{folder}'.")

    print(f"  Found {len(files)} file(s):")
    for f in files:
        print(f"    • {os.path.basename(f)}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df
