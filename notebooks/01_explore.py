import pandas as pd

df = pd.read_csv('../data/raw/Fraud_Data.csv')
print(df.shape)
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True))

# Step 1: Look at the first few rows
print(df.head())

# Step 2: Check column names, data types, and how many non-null values exist
print(df.info())

# Step 3: Check for missing values specifically
print(df.isnull().sum())