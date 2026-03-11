import pandas as pd
import yaml
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split


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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    model = joblib.load("models/churn_model.joblib")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()