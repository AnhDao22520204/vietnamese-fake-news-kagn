from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_prepared_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
    )
