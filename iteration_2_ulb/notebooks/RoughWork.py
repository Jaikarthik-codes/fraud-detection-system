import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('../data/raw/creditcard.csv')
scaler = StandardScaler()
scaler.fit(df[['Amount', 'Time']])
joblib.dump(scaler, '../models/scaler.pkl')
print("Scaler saved")