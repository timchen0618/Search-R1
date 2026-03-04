"""
Simple script to load Qampari parquet data.
"""

import pandas as pd
from pathlib import Path


def load_qampari(data_dir: str = "data/qampari"):
    """Load train_base.parquet and test_base.parquet from the given directory."""
    data_path = Path(data_dir)
    train_df = pd.read_parquet(data_path / "train_base.parquet")
    test_df = pd.read_parquet(data_path / "test_base.parquet")
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_qampari(data_dir='data/musique')
    print(f"Train: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")
    print("\nTrain columns:", train_df.columns.tolist())
    print("\nTrain head:\n", train_df.head())
