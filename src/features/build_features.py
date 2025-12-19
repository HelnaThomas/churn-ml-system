import pandas as pd
import yaml
from pathlib import Path


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def validate_columns(df, config):
    expected = (
        config["features"]["numerical"]
        + config["features"]["categorical"]
        + [config["data"]["target"]]
    )

    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")


def clean_total_charges(df):
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df["TotalCharges"].fillna(0, inplace=True)
    return df


def encode_target(df, target_col):
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    return df


def main():
    config = load_config()

    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]

    df = pd.read_csv(raw_path)

    validate_columns(df, config)
    df = clean_total_charges(df)
    df = encode_target(df, config["data"]["target"])

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"Processed features saved to {processed_path}")


if __name__ == "__main__":
    main()
