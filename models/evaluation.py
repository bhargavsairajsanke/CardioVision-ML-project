import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess_data

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(IMAGE_DIR, exist_ok=True)

# ==========================================================
# Load Data
# ==========================================================

X_train, X_test, y_train, y_test, scaler, df = preprocess_data()

# ==========================================================
# Load Results
# ==========================================================

results = pd.read_csv(
    os.path.join(
        MODEL_DIR,
        "model_results.csv"
    )
)

# ==========================================================
# Load Best Model
# ==========================================================

with open(
    os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    ),
    "rb"
) as file:

    model = pickle.load(file)

# ==========================================================
# Prediction
# ==========================================================

y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):

    y_score = model.predict_proba(X_test)[:,1]

else:

    y_score = model.decision_function(X_test)

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, _ = roc_curve(y_test, y_score)

auc = roc_auc_score(y_test, y_score)

plt.figure(figsize=(8,6))

plt.plot(fpr,tpr,label=f"AUC = {auc:.3f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "roc_curve.png"
    )
)

plt.close()

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "confusion_matrix.png"
    )
)

plt.close()

# ==========================================================
# Accuracy Comparison
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.ylabel("Accuracy")

plt.title("Accuracy Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "accuracy_comparison.png"
    )
)

plt.close()

# ==========================================================
# Precision Comparison
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results["Model"],
    results["Precision"]
)

plt.ylabel("Precision")

plt.title("Precision Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "precision_comparison.png"
    )
)

plt.close()

# ==========================================================
# Recall Comparison
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results["Model"],
    results["Recall"]
)

plt.ylabel("Recall")

plt.title("Recall Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "recall_comparison.png"
    )
)

plt.close()

# ==========================================================
# F1 Comparison
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results["Model"],
    results["F1 Score"]
)

plt.ylabel("F1 Score")

plt.title("F1 Score Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "f1_comparison.png"
    )
)

plt.close()

# ==========================================================
# Feature Importance
# ==========================================================

if hasattr(model, "feature_importances_"):

    feature_importance = pd.DataFrame({

        "Feature":df.drop(
            "TenYearCHD",
            axis=1
        ).columns,

        "Importance":model.feature_importances_

    })

    feature_importance = feature_importance.sort_values(

        by="Importance",

        ascending=False

    )

    plt.figure(figsize=(10,6))

    plt.bar(

        feature_importance["Feature"],

        feature_importance["Importance"]

    )

    plt.xticks(rotation=90)

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            IMAGE_DIR,

            "feature_importance.png"

        )

    )

    plt.close()

print("="*60)
print("Evaluation Completed Successfully")
print("Graphs saved inside static/images/")
print("="*60)