# -------------------------------
# Part 1: Imports & Data Load
# -------------------------------
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


script_dir = Path(__file__).resolve().parent
possible_paths = [
    script_dir / "titanic_passengers.csv",
    script_dir.parent / "titanic_passengers.csv",
    script_dir.parent / "Data" / "titanic_passengers.csv"
]

data_path = None
for path in possible_paths:
    if path.exists():
        data_path = path
        break

if data_path is None:
    print("ERROR: Could not find 'titanic_passengers.csv'. Looked in:")
    for p in possible_paths:
        print(" -", p)
    sys.exit(1)

# Load the CSV
df = pd.read_csv(data_path)

print(f"Loaded data from: {data_path}")
print(df.head())
print("Data shape:", df.shape)


# -------------------------------
# Part 2: Data Inspection
# -------------------------------

print("\nColumns:")
print(df.columns.tolist())

print("\nData Info:")
df.info()

print("\nNumerical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isna().sum())

# -------------------------------
# Part 3: Data Cleaning
# -------------------------------

# Work on a copy to preserve raw data
df_clean = df.copy()

# Drop columns that are mostly missing or not useful for analysis
columns_to_drop = ["Cabin", "Ticket", "Name"]
df_clean.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# Fill missing Age values with median (robust to outliers)
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

# Fill missing Embarked values with most frequent category
df_clean["Embarked"] = df_clean["Embarked"].fillna(
    df_clean["Embarked"].mode()[0]
)

# Verify cleaning worked
print("\nMissing Values After Cleaning:")
print(df_clean.isna().sum())

# -------------------------------
# Part 4: Exploratory Data Analysis
# -------------------------------

# Survival count
plt.figure()
df_clean["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()


# Survival rate by sex
plt.figure()
df_clean.groupby("Sex")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()


# Survival rate by passenger class
plt.figure()
df_clean.groupby("Pclass")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.tight_layout()
plt.show()


# Age distribution by survival
plt.figure()
df_clean[df_clean["Survived"] == 1]["Age"].plot(
    kind="hist", bins=30, alpha=0.7, label="Survived"
)
df_clean[df_clean["Survived"] == 0]["Age"].plot(
    kind="hist", bins=30, alpha=0.7, label="Did Not Survive"
)
plt.legend()
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -------------------------------
# Part 5: Feature Engineering & Model Preparation
# -------------------------------

# Create new features
df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)

# Define target and features
X = df_clean.drop(columns="Survived")
y = df_clean["Survived"]

# Identify feature types
numeric_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone"
]

categorical_features = [
    "Sex",
    "Embarked",
    "Pclass"
]

# Final verification
print("\nFeature Engineering Complete")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# -------------------------------
# Part 6: Modeling & Evaluation
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing pipelines
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Build model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
