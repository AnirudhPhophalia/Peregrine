import getpass

import h5py
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, login

from config import (
    INPUT_RECEIVER_MODE,
    MAX_ROWS_PER_FILE,
    N_FILES_TO_LOAD,
    REPO_ID,
)


def check_hf_login():
    try:
        user = HfApi().whoami()
        print(f"Already logged into Hugging Face as: {user['name']}")
    except Exception:
        print("\nHugging Face login required.")
        token = getpass.getpass("Paste your HuggingFace token: ").strip()

        if not token:
            raise RuntimeError("No Hugging Face token provided.")

        login(token=token)
        user = HfApi().whoami()
        print(f"Successfully logged into Hugging Face as: {user['name']}")


def get_data_files(mode=INPUT_RECEIVER_MODE):
    api = HfApi()

    files = api.list_repo_files(
        REPO_ID,
        repo_type="dataset",
    )

    key = f"train_{mode}"

    data_files = sorted(
        f
        for f in files
        if key in f and "stats" not in f.lower()
    )

    if not data_files:
        raise RuntimeError(
            f"No files matching '{key}' were found in {REPO_ID}."
        )

    print(f"\nFound {len(data_files)} {key} data files.")

    for f in data_files[:10]:
        print(" ", f)

    return data_files


def smart_load(path, max_rows):
    with open(path, "rb") as f:
        header = f.read(8)

    # HDF5
    if header[:4] == b"\x89HDF":
        with h5py.File(path, "r") as hf:
            if "data" in hf:
                ds = hf["data"]
            else:
                keys = list(hf.keys())
                ds = hf[keys[0]]

            available_rows = len(ds)
            n = min(max_rows, available_rows)

            print(
                f"Loading {n:,} rows out of "
                f"{available_rows:,} available rows"
            )

            return pd.DataFrame(ds[:n])

    # Parquet
    if header[:4] == b"PAR1":
        return pd.read_parquet(path).head(max_rows)

    # CSV
    return pd.read_csv(
        path,
        nrows=max_rows,
    )


def load_dataset(mode=INPUT_RECEIVER_MODE):
    check_hf_login()

    data_files = get_data_files(mode)
    selected_files = data_files[:N_FILES_TO_LOAD]

    print(f"\nFiles requested: {N_FILES_TO_LOAD}")
    print(f"Files selected: {len(selected_files)}")
    print(f"Maximum rows per file: {MAX_ROWS_PER_FILE:,}")
    print(f"Input receiver data: {mode.upper()}")

    dfs = []

    for i, f in enumerate(selected_files, start=1):
        print("\n" + "=" * 55)
        print(f"FILE [{i}/{len(selected_files)}]")
        print(f"Downloading: {f}")
        print("=" * 55)

        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f,
            repo_type="dataset",
        )

        df_part = smart_load(
            local_path,
            MAX_ROWS_PER_FILE,
        )

        print(f"Loaded rows: {len(df_part):,}")
        dfs.append(df_part)

    df = pd.concat(
        dfs,
        ignore_index=True,
    )

    print("\n" + "=" * 55)
    print("DATASET LOADING SUMMARY")
    print("=" * 55)
    print(f"Files loaded: {len(dfs)}")
    print(f"Total rows loaded: {len(df):,}")
    print(f"Final shape: {df.shape}")
    print("=" * 55)

    return df
