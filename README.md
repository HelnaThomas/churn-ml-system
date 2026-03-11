
# Customer Churn Prediction ML System 

## 1. Problem Statement

The goal of this project is to predict **customer churn** for a telecom company.
Customer churn refers to customers leaving a service or cancelling their subscription.
Predicting churn helps companies identify customers likely to leave so they can take preventive actions such as targeted offers or customer support interventions.

---

## 2. Dataset

The project uses the **Telco Customer Churn dataset**, which contains customer information such as:

* Customer tenure
* Contract type
* Internet service type
* Monthly charges
* Payment method
* Whether the customer churned or not

The **target variable** is a binary label indicating whether the customer left the service.

---

## 3. System Architecture

Instead of building everything inside a single notebook, the project was structured as a **modular machine learning pipeline**.

Pipeline flow:

Raw Data
→ Feature Engineering
→ Model Training
→ Model Evaluation
→ Batch Inference

Each stage is implemented as a separate script inside the `src` directory to mimic real-world ML systems.

---

## 4. Feature Engineering

The feature engineering module processes the raw dataset and prepares it for machine learning.

Main tasks performed:

* Data cleaning
* Handling categorical variables
* Preparing input features
* Saving processed data

Output file:
data/processed/features.csv

This processed dataset is then used for model training.

---

## 5. Model Training

The training module loads the processed dataset and trains a classification model.

Steps involved:

1. Load processed features
2. Split dataset into training and test sets
3. Train a Logistic Regression model
4. Save the trained model

Saved model artifact:
models/churn_model.joblib

Saving the model allows it to be reused later for predictions.

---

## 6. Model Evaluation

The evaluation module measures how well the trained model performs on unseen data.

Metrics used include:

* Accuracy
* Precision
* Recall
* ROC-AUC
* Confusion Matrix

Example performance:
Accuracy ≈ 0.82
ROC-AUC ≈ 0.86

These metrics help understand the model’s ability to distinguish between churn and non-churn customers.

---

## 7. Batch Inference

The inference module loads the trained model and generates predictions on a dataset.

Steps:

1. Load the saved model
2. Apply the model to input features
3. Generate predictions
4. Save predictions

Output file:
data/processed/churn_predictions.csv

This simulates how models generate predictions in production pipelines.

---

## 8. Project Structure

The project follows a structured repository layout similar to production ML systems.

churn-ml-system
│
├── configs
│   └── config.yaml
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── src
│   ├── features
│   ├── training
│   ├── evaluation
│   └── inference
│
├── run_pipeline.py
└── README.md

This modular design makes the system easier to maintain and scale.

---

## 9. Engineering Practices Used

The project follows several good engineering practices:

* Version control using Git
* Repository hosting on GitHub
* Modular pipeline design
* Configuration-driven pipeline
* Ignoring large files using `.gitignore`
* Maintaining folder structure with `.gitkeep`
* Saving trained models as artifacts

These practices align with real-world machine learning engineering workflows.

---

## 10. Possible Improvements

If this system were extended further, the following improvements could be added:

* Experiment tracking
* Model serving API
* Containerization for deployment
* CI/CD pipelines
* Cloud deployment

These additions would transform the project into a fully production-ready ML system.

---

## 11. Summary

"I built a modular machine learning pipeline to predict telecom customer churn. The system processes raw customer data, performs feature engineering, trains a logistic regression model, evaluates performance using metrics such as accuracy and ROC-AUC, and generates predictions. The project is structured like a production ML repository with separate modules for feature engineering, training, evaluation, and inference, and the entire workflow is version controlled using Git."
