import glob
import os

import pandas as pd


def _read_csv_folder(path: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em: {path}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_slice_train_test(slice_idx: int, split_path: str = ""):
    train_path = os.path.join(split_path, f"slice_{slice_idx}", "train")
    test_path = os.path.join(split_path, f"slice_{slice_idx}", "test")

    train_df = _read_csv_folder(train_path)
    test_df = _read_csv_folder(test_path)

    train_df["SessionId"] = train_df["SessionId"].astype(str)
    test_df["SessionId"] = test_df["SessionId"].astype(str)

    train_df["ItemId"] = train_df["ItemId"].astype(str)
    test_df["ItemId"] = test_df["ItemId"].astype(str)

    train_df["Time"] = train_df["Time"].astype(int)
    test_df["Time"] = test_df["Time"].astype(int)

    return train_df, test_df
