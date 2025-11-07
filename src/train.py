# src/train.py
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json

# CONFIGURATION
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "titanic-experiment")
MODEL_NAME = "titanic-rf"
RANDOM_SEED = 42

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Simple preprocessing for Titanic dataset (example)
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    X = df[features]
    y = df['Survived']
    return X, y

def save_baseline_stats(X, y, out_path="data/baseline_stats.json"):
    stats = {
        "n_rows": int(len(X)),
        "features": {col: {"mean": float(X[col].mean()), "std": float(X[col].std())} for col in X.columns}
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

def main():
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    with mlflow.start_run() as run:
        n_estimators = 100
        max_depth = 6
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        # Log params/metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        # Log model
        mlflow.sklearn.log_model(clf, artifact_path="model", registered_model_name=MODEL_NAME)

        # Save baseline for drift checks (use the train set stats)
        save_baseline_stats(X_train, y_train, out_path="data/baseline_stats.json")
        mlflow.log_artifact("data/baseline_stats.json", artifact_path="baseline")

        print(f"Run_id: {run.info.run_id}, acc: {acc:.4f}, auc: {auc:.4f}")

if __name__ == "__main__":
    main()
