"""
------------------------------------------------------------
CardioVision
Preprocessing Module

Tasks:
1. Load Dataset
2. Explore Dataset
3. Handle Missing Values
4. Split Features and Target
5. Train-Test Split
6. Feature Scaling
7. Save Scaler
------------------------------------------------------------
"""

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data():

    # ==========================================================
    # Project Paths
    # ==========================================================

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "framingham.csv")

    MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==========================================================
    # Load Dataset
    # ==========================================================

    df = pd.read_csv(DATASET_PATH)

    print("=" * 60)
    print("Original Dataset Shape :", df.shape)
    print("=" * 60)

    # ==========================================================
    # Dataset Information
    # ==========================================================

    print("\nFirst Five Rows")
    print(df.head())

    print("\nDataset Information")
    print(df.info())

    print("\nStatistical Summary")
    print(df.describe())

    print("\nMissing Values Before Filling:")
    print(df.isnull().sum())

    # ==========================================================
    # Fill Missing Values
    # ==========================================================

    numeric_columns = [
        "education",
        "cigsPerDay",
        "BPMeds",
        "totChol",
        "BMI",
        "heartRate",
        "glucose"
    ]

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    print("\nMissing Values After Filling:")
    print(df.isnull().sum())

    # ==========================================================
    # Target Distribution
    # ==========================================================

    print("\nTarget Variable Distribution")
    print(df["TenYearCHD"].value_counts())

    # ==========================================================
    # Feature Matrix and Target Vector
    # ==========================================================

    X = df.drop("TenYearCHD", axis=1)
    y = df["TenYearCHD"]

    # ==========================================================
    # Train-Test Split
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ==========================================================
    # Feature Scaling
    # ==========================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==========================================================
    # Save Scaler
    # ==========================================================

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    print("\nScaler saved successfully!")

    # ==========================================================
    # Dataset Shapes
    # ==========================================================

    print("\nTraining Data Shape :", X_train.shape)
    print("Testing Data Shape  :", X_test.shape)

    print("\nPreprocessing Completed Successfully!")

    # ==========================================================
    # Return Processed Data
    # ==========================================================

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        df
    )


# ==============================================================
# Main Function
# ==============================================================

if __name__ == "__main__":

    X_train, X_test, y_train, y_test, scaler, df = preprocess_data()