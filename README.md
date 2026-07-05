# FraudLens

Built from scratch to understand every decision deeply enough 
to defend it in a technical interview.

## Iteration 1 — Fraud_Data.csv (Documented Dead End)

E-commerce fraud dataset with interpretable features.
Engineered: time_to_purchase, purchase_hour, high_value.

Finding: time_to_purchase dominated at 68% feature importance
even after full engineering. Recall ceiling at 0.53 confirmed.
Switched datasets based on evidence, not assumption.

## Iteration 2 — ULB Credit Card Fraud Dataset

284,807 transactions, V1-V28 PCA features + Amount + Time.
Fraud rate: 0.17% — handled with SMOTE after train/test split.

Results:
Model              Precision  Recall   F1
Random Forest        0.82      0.82    0.82  ← winner
Gradient Boosting    0.11      0.90    0.19
AdaBoost             0.05      0.91    0.10

Random Forest won on F1. GB and AdaBoost had higher recall but 
catastrophic precision due to sequential tree architecture 
amplifying SMOTE synthetic patterns.

## Flask App

Simple web demo — select a preset transaction, get fraud 
prediction with probability score.

## Stack
Python, pandas, NumPy, scikit-learn, imbalanced-learn, Flask

## Run
```bash
cd iteration_2_ulb/backend
python app.py
```
Open http://127.0.0.1:5000