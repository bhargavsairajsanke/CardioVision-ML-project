# CardioVision: Coronary Heart Disease Risk Analysis & Patient Profiling

## Overview

CardioVision is a machine learning-based web application that predicts the 10-year risk of Coronary Heart Disease (CHD) using clinical patient data from the Framingham Heart Study dataset. The project compares multiple supervised learning algorithms, performs patient clustering using unsupervised learning, and provides an interactive dashboard for model evaluation and visualization through a Flask web application.

---

## Features

- Predicts 10-year Coronary Heart Disease (CHD) risk.
- Compares five supervised machine learning models.
- Performs patient clustering using KMeans and DBSCAN.
- Interactive Flask web application for prediction.
- Dashboard with model evaluation and clustering visualizations.
- Automatic preprocessing and feature scaling.
- Saves trained models for deployment using Pickle.

---

## Machine Learning Models

### Supervised Learning
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting

### Unsupervised Learning
- KMeans Clustering
- DBSCAN Clustering

---

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Visualizations include:

- ROC Curve
- Confusion Matrix
- Feature Importance
- Model Comparison Charts
- KMeans Cluster Visualization
- DBSCAN Cluster Visualization

---

## 🛠 Technologies Used

### Programming Languages
- Python
- HTML
- CSS

### Python Libraries
- Flask
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pickle

---

## Project Structure

```
CardioVision/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── framingham.csv
│
├── models/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── supervised_models.py
│   ├── evaluation.py
│   └── clustering.py
│
├── saved_models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── kmeans.pkl
│   ├── dbscan.pkl
│   └── model_results.csv
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── images/
│
└── templates/
    ├── index.html
    ├── result.html
    └── dashboard.html
```

---

## ⚙ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<username>/CardioVision.git
```

### 2. Navigate to the Project Directory

```bash
cd CardioVision
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1

Run preprocessing

```bash
python models/preprocessing.py
```

### Step 2

Train supervised models

```bash
python models/supervised_models.py
```

### Step 3

Generate evaluation graphs

```bash
python models/evaluation.py
```

### Step 4

Perform clustering

```bash
python models/clustering.py
```

### Step 5

Start the Flask application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Dashboard

The dashboard displays:

- Heart disease prediction
- ROC Curve
- Confusion Matrix
- Accuracy Comparison
- Precision Comparison
- Recall Comparison
- F1-Score Comparison
- Feature Importance
- KMeans Clustering
- DBSCAN Clustering

---

## Dataset

Dataset Used:

**Framingham Heart Study Dataset**

Target Variable:

- **TenYearCHD**
  - 0 → No CHD Risk
  - 1 → CHD Risk

---

## Future Enhancements

- Hyperparameter tuning using GridSearchCV
- Improved model accuracy
- User authentication
- Deployment on Render or Railway
- Interactive charts
- PDF report generation
- Docker support

---

##  Team Contibutions

Team Member 1
Name : Devubarki Pushya mithra

Specific Contibutions:

Machine Learning Development & Backend Integration:

Collaboratively developed the data preprocessing pipeline, including missing value handling, feature scaling, and train-test splitting.
Collaboratively implemented and trained supervised learning models (Logistic Regression, SVM, Decision Tree, Random Forest, and Gradient Boosting) for heart disease risk prediction.
Integrated trained machine learning models into the Flask backend to enable real-time prediction.
Participated in end-to-end testing, debugging, and project integration to ensure reliable application performance.



---

