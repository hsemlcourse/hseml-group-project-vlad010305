
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

TARGET_COL = "Machine failure"

ID_COLS = [
    "UDI",
    "Product ID",
]

LEAKAGE_COLS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target.

    ID columns and leakage columns are removed from features.
    """
    drop_cols = [target_col] + ID_COLS + LEAKAGE_COLS
    existing_drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=existing_drop_cols)
    y = df[target_col]

    return X, y


def make_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
):
    """Create stratified train/validation/test split.

    Final split proportions by default:
    - train: 70%
    - validation: 15%
    - test: 15%
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_relative_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test