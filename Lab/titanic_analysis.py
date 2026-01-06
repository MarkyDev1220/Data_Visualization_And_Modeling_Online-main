# -------------------------------
# Part 1: Imports & Data Load
# -------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = Path("titanic_passengers.csv")
df = pd.read_csv(data_path)

print(df.head())
print(df.shape)


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
