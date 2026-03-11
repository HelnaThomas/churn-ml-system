import pandas as pd
import joblib


def main():

    model = joblib.load("models/churn_model.joblib")

    input_data = pd.read_csv("data/processed/features.csv")

    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    output = input_data.copy()
    output["churn_prediction"] = predictions
    output["churn_probability"] = probabilities

    output.to_csv("data/processed/churn_predictions.csv", index=False)

    print("Predictions saved to data/processed/churn_predictions.csv")


if __name__ == "__main__":
    main()