import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "framingham.csv")

IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

df = pd.read_csv(DATASET_PATH)

# Fill missing values
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

print("Dataset Loaded Successfully")

# =====================================================
# 1. Target Distribution
# =====================================================

plt.figure(figsize=(6,5))

df["TenYearCHD"].value_counts().plot(kind="bar")

plt.title("Target Distribution")
plt.xlabel("Ten Year CHD")
plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR,"target_distribution.png"))

plt.close()

# =====================================================
# 2. Age Distribution
# =====================================================

plt.figure(figsize=(7,5))

plt.hist(df["age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR,"age_distribution.png"))

plt.close()

# =====================================================
# 3. Gender Distribution
# =====================================================

plt.figure(figsize=(6,5))

df["male"].value_counts().plot(kind="bar")

plt.title("Gender Distribution")
plt.xlabel("0 = Female, 1 = Male")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR,"gender_distribution.png"))

plt.close()

# =====================================================
# 4. Smoking Distribution
# =====================================================

plt.figure(figsize=(6,5))

df["currentSmoker"].value_counts().plot(kind="bar")

plt.title("Smoking Distribution")
plt.xlabel("0 = Non-Smoker, 1 = Smoker")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR,"smoking_distribution.png"))

plt.close()

# =====================================================
# 5. BMI Distribution
# =====================================================

plt.figure(figsize=(7,5))

plt.hist(df["BMI"], bins=20)

plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR,"bmi_distribution.png"))

plt.close()

# =====================================================
# 6. Cholesterol Distribution
# =====================================================

plt.figure(figsize=(7,5))

plt.hist(df["totChol"], bins=20)

plt.title("Cholesterol Distribution")
plt.xlabel("Total Cholesterol")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR,"cholesterol_distribution.png"))

plt.close()

# =====================================================
# 7. Heart Rate Distribution
# =====================================================

plt.figure(figsize=(7,5))

plt.hist(df["heartRate"], bins=20)

plt.title("Heart Rate Distribution")
plt.xlabel("Heart Rate")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR,"heart_rate_distribution.png"))

plt.close()

# =====================================================
# 8. Correlation Heatmap
# =====================================================

plt.figure(figsize=(12,10))

corr = df.corr()

plt.imshow(corr)

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR,"correlation_heatmap.png"))

plt.close()

print("\nAll graphs generated successfully!")
print("Saved inside static/images/")