import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-inspired features for predictive maintenance task."""
    df = df.copy()

    df["temperature_diff"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["power_proxy"] = (
        df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    )

    df["torque_per_rpm"] = (
        df["Torque [Nm]"] / df["Rotational speed [rpm]"]
    )

    df["wear_torque_interaction"] = (
        df["Tool wear [min]"] * df["Torque [Nm]"]
    )

    return df