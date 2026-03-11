import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    data_path = config["data"]["processed_path"]
    target = config["data"]["target"]

    df = pd.read_csv(data_path)

    X = df.drop(columns=[target])
    y = df[target]

    categorical = config["features"]["categorical"]
    numerical = config["features"]["numerical"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numerical),
        ]
    )

    model = LogisticRegression(max_iter=5000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc}")

    joblib.dump(pipeline, "models/churn_model.joblib")
    print("Model saved to models/churn_model.joblib")


if __name__ == "__main__":
    main()