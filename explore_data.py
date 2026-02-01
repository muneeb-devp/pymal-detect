"""
Exploratory Data Analysis for Brazilian Malware Dataset
"""
import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('brazilian-malware-dataset-master/brazilian-malware.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Number of instances: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== First few rows ===")
print(df.head())

print("\n=== Data types ===")
print(df.dtypes)

print("\n=== Target variable (Label) distribution ===")
print(df['Label'].value_counts())
print(f"\nClass balance:")
print(df['Label'].value_counts(normalize=True))

print("\n=== Missing values ===")
missing = df.isnull().sum()
print(f"Total missing values: {missing.sum()}")
if missing.sum() > 0:
    print("\nColumns with missing values:")
    print(missing[missing > 0])

print("\n=== Basic statistics ===")
print(df.describe())

# Check for any non-numeric columns that need encoding
print("\n=== Non-numeric columns ===")
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(non_numeric)
for col in non_numeric:
    if col != 'Label':
        print(f"\n{col} unique values: {df[col].nunique()}")
        print(df[col].value_counts().head(10))
