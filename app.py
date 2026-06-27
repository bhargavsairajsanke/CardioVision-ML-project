import os
import pickle
import numpy as np

from flask import Flask, render_template, request

# ==========================================================
# Flask App
# ==========================================================

app = Flask(__name__)

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# ==========================================================
# Load Model and Scaler
# ==========================================================

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as file:
    model = pickle.load(file)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as file:
    scaler = pickle.load(file)

# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# Prediction
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        male = int(request.form["male"])
        age = float(request.form["age"])
        education = float(request.form["education"])
        currentSmoker = int(request.form["currentSmoker"])
        cigsPerDay = float(request.form["cigsPerDay"])
        BPMeds = int(request.form["BPMeds"])
        prevalentStroke = int(request.form["prevalentStroke"])
        prevalentHyp = int(request.form["prevalentHyp"])
        diabetes = int(request.form["diabetes"])
        totChol = float(request.form["totChol"])
        sysBP = float(request.form["sysBP"])
        diaBP = float(request.form["diaBP"])
        BMI = float(request.form["BMI"])
        heartRate = float(request.form["heartRate"])
        glucose = float(request.form["glucose"])

        data = np.array([[
            male,
            age,
            education,
            currentSmoker,
            cigsPerDay,
            BPMeds,
            prevalentStroke,
            prevalentHyp,
            diabetes,
            totChol,
            sysBP,
            diaBP,
            BMI,
            heartRate,
            glucose
        ]])

        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data)[0][1] * 100
        else:
            probability = 0

        if prediction == 1:
            result = "High Risk"
            color = "red"
        else:
            result = "Low Risk"
            color = "green"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability, 2),
            color=color
        )

    except Exception as e:

        return f"Error: {e}"


# ==========================================================
# Dashboard
# ==========================================================

@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)