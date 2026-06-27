import os
import pickle
import pandas as pd

from preprocessing import preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# Load Data
# ==========================================================

X_train, X_test, y_train, y_test, scaler, df = preprocess_data()

# ==========================================================
# Models
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

}

# ==========================================================
# Results
# ==========================================================

results = []

best_model = None
best_model_name = ""

best_accuracy = 0

# ==========================================================
# Training Loop
# ==========================================================

for name, model in models.items():

    print("\n" + "="*60)
    print(f"Training {name}")
    print("="*60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_test)[:,1]

    else:

        y_prob = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")

    results.append({

        "Model":name,
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1 Score":f1,
        "ROC AUC":auc

    })

    # Save every model

    filename = name.lower().replace(" ","_") + ".pkl"

    with open(os.path.join(MODEL_DIR, filename),"wb") as file:

        pickle.dump(model,file)

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# ==========================================================
# Save Best Model
# ==========================================================

with open(

    os.path.join(MODEL_DIR,"best_model.pkl"),

    "wb"

) as file:

    pickle.dump(best_model,file)

# ==========================================================
# Comparison Table
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(

    by="Accuracy",

    ascending=False

)

print("\n")

print("="*80)

print("MODEL COMPARISON")

print("="*80)

print(results_df)

# ==========================================================
# Save Results
# ==========================================================

results_df.to_csv(

    os.path.join(
        MODEL_DIR,
        "model_results.csv"
    ),

    index=False

)

# ==========================================================
# Best Model
# ==========================================================

print("\n")

print("="*80)

print("BEST MODEL")

print("="*80)

print(f"Model    : {best_model_name}")

print(f"Accuracy : {best_accuracy:.4f}")

print("="*80)

print("\nAll models saved successfully.")

print("Results saved successfully.")