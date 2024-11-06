import pandas as pd


def categorical_to_one_hot(
    df: pd.DataFrame, column: str, drop_column: bool = True
) -> pd.DataFrame:
    dummy_df = pd.get_dummies(df[column], prefix=column, dtype=int)
    df = pd.concat([df, dummy_df], axis=1)

    if drop_column:
        df = df.drop(column, axis=1)

    return df


def standardize_column(
    df: pd.DataFrame, column: str, rename: bool = True
) -> pd.DataFrame:
    df = df.copy(deep=True)

    mean = df[column].mean()
    std = df[column].std()

    new_col = f"{column}_standardized" if rename else column
    df[new_col] = (df[column] - mean) / std
    return df
