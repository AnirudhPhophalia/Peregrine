def inspect_dataset(df):
    print("\n========== DATASET INFORMATION ==========\n")
    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nNumeric summary:")
    print(df.describe(include="all"))

    print("\nColumn positions:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
