import pandas as pd
import h5py
import getpass

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    login
)

from config import (
    REPO_ID,
    N_FILES_TO_LOAD,
    MAX_ROWS_PER_FILE
)


def check_hf_login():

    try:
        user = HfApi().whoami()

        print(
            f"Already logged into Hugging Face as: "
            f"{user['name']}"
        )

    except Exception:

        print("\nHugging Face login required.")

        token = getpass.getpass(
            "Paste your HuggingFace token: "
        ).strip()

        if not token:
            raise RuntimeError(
                "No Hugging Face token provided."
            )

        try:
            login(token=token)

            user = HfApi().whoami()

            print(
                f"Successfully logged in as: "
                f"{user['name']}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Hugging Face login failed: {e}"
            )


def get_data_files():

    api = HfApi()

    files = api.list_repo_files(
        REPO_ID,
        repo_type="dataset"
    )

    data_files = sorted([
        f for f in files
        if "train_scan" in f
        and "stats" not in f.lower()
    ])

    print(
        f"\nFound {len(data_files)} train_scan data files."
    )

    print("\nFirst available files:")

    for f in data_files[:10]:
        print(" ", f)

    return data_files


def smart_load(path, max_rows):

    with open(path, "rb") as f:
        header = f.read(8)

    # HDF5
    if header[:4] == b"\x89HDF":

        with h5py.File(path, "r") as hf:

            keys = list(hf.keys())

            print("HDF5 keys:", keys)

            # Explicitly use data if available
            if "data" in hf:
                ds = hf["data"]
            else:
                ds = hf[keys[0]]

            available_rows = len(ds)

            n = min(
                max_rows,
                available_rows
            )

            print(
                f"Loading {n:,} rows "
                f"out of {available_rows:,} available rows"
            )

            data = ds[:n]

            return pd.DataFrame(data)

    # PARQUET
    elif header[:4] == b"PAR1":

        df = pd.read_parquet(path)

        return df.head(max_rows)

    # CSV
    else:

        return pd.read_csv(
            path,
            nrows=max_rows
        )


def load_dataset():
    check_hf_login()
    data_files = get_data_files()
    selected_files = data_files[
        :N_FILES_TO_LOAD
    ]
    if not selected_files:
        raise RuntimeError(
            "No dataset files were selected."
        )
    print(
        f"\nFiles requested: {N_FILES_TO_LOAD}"
    )
    print(
        f"Files selected: {len(selected_files)}"
    )
    print(
        f"Maximum rows per file: "
        f"{MAX_ROWS_PER_FILE:,}"
    )
    dfs = []
    total_rows = 0

    for i, f in enumerate(
        selected_files,
        start=1
    ):

        print("\n" + "=" * 55)

        print(
            f"FILE [{i}/{len(selected_files)}]"
        )

        print(
            f"Downloading: {f}"
        )

        print("=" * 55)


        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f,
            repo_type="dataset"
        )


        print("Loading data...")


        df_part = smart_load(
            local_path,
            MAX_ROWS_PER_FILE
        )


        rows_loaded = len(df_part)

        total_rows += rows_loaded


        print(
            f"Loaded rows from this file: "
            f"{rows_loaded:,}"
        )

        print(
            f"Cumulative rows: "
            f"{total_rows:,}"
        )


        dfs.append(df_part)


    df = pd.concat(
        dfs,
        ignore_index=True
    )
    print("\n" + "=" * 55)
    print("FINAL DATASET LOADING SUMMARY")
    print("=" * 55)

    print(
        f"Files requested: "
        f"{N_FILES_TO_LOAD}"
    )

    print(
        f"Files successfully loaded: "
        f"{len(dfs)}"
    )

    print(
        f"Total rows loaded: "
        f"{len(df):,}"
    )

    print(
        f"Final dataset shape: "
        f"{df.shape}"
    )

    print("=" * 55)


    return df