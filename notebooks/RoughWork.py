import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/processed/Fraud_Data_cleaned.csv')
X = df.drop(columns=['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Check which features the model found most useful
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print(feature_importance.sort_values(ascending=False))