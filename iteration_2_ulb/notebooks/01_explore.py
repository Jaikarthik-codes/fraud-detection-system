import pandas as pd

df = pd.read_csv('../data/raw/creditcard.csv')

print(df.shape)
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True))
print(df.isnull().sum())
print(df.info())