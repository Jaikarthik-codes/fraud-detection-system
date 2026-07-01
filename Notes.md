## Step 1 - Data Exploration

Dataset: Fraud_Data.csv
Shape: 151,112 rows x 11 columns

Class distribution:
- Not fraud (0): 136,961 rows = 90.6%
- Fraud (1): 14,151 rows = 9.4%

Key finding: Dataset is imbalanced - 90.6% not fraud vs 9.4% fraud.

Why accuracy is misleading here:
A model that predicts "not fraud" for every single transaction
would score 90.6% accuracy without learning anything useful.
It would catch zero fraud cases but still look impressive on paper.
So we use precision, recall and F1 instead.

Metrics we will use and why:
- Recall: of all real fraud cases, how many did the model catch?
  (most important - missed fraud costs real money)
- Precision: of everything flagged as fraud, how many actually were?
  (false alarms have a cost too - blocking real customers is bad)
- F1: balance between recall and precision, useful for comparing models

This is a supervised learning problem:
The 'class' column already tells us which transactions are fraud.
The model learns by comparing its predictions against these known answers.
Without a label column like this, supervised learning is not possible.

## Step 2 - Preprocessing

Decision: Used get_dummies() for source, browser, sex instead of 
numbering them 1/2/3.
Why: These are categories with no mathematical relationship between them.
Numbering them would make the model think "Safari > Chrome" which is meaningless.
get_dummies() creates separate True/False columns for each category instead.

Feature engineered: time_to_purchase (seconds between signup and purchase)
Why: Fraudsters tend to sign up and buy almost instantly. This gap is
a real behavioral signal, not something that existed in the raw data.

Dropped: user_id, device_id, ip_address, signup_time, purchase_time
Why: IDs carry no learnable signal. Raw timestamps replaced by 
time_to_purchase. ip_address would need extra work to be useful.

## Step 3 - Model Training

Split data: 80% training (120,889 rows) / 20% testing (30,223 rows)
- test_size=0.2 → 20% for testing
- random_state=42 → same split every run, results are reproducible
- stratify=y → preserves 9.4% fraud ratio in both train and test sets

X = all columns except 'class' (the features/clues)
y = 'class' column only (the correct answers)
Why separate? If 'class' stayed in X, model would cheat by 
looking directly at the answer during training.

Results (all three models):
- Recall on fraud: 0.53 (catching only 53% of real fraud cases)
- Precision on fraud: 0.97-1.00 (when it flags fraud, it's usually right)
- Accuracy: 0.95-0.96 (misleading due to imbalance)

Feature importance (Random Forest):
- time_to_purchase: 77.5% of the work
- purchase_value: 12.4%
- age: 9.2%
- browser/source/sex columns: basically nothing (<0.1% each)

Why is recall stuck at 0.53?
The model is essentially a one-feature model. It catches obvious 
fraud (very short signup-to-purchase gap) but misses cases where 
fraudsters wait longer. Would need richer features in a real system.

False Positive = model said fraud, was actually legitimate (annoyed customer)
False Negative = model said not fraud, was actually fraud (money lost)
False negatives are more dangerous in fraud detection.