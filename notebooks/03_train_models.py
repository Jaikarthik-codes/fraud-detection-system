import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

## Step 1: Load cleaned data
df = pd.read_csv("../data/processed/Fraud_Data_Cleaned.csv")

# Step 2: Separate features (X) from label (y)
x = df.drop(columns=["class"])
y = df["class"]

# Step 3: Split into training and test sets
X_train, X_test, Y_train,Y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Step 4: Train Random Forest
rf_model = RandomForestClassifier(n_estimators = 100, random_state=42)
rf_model.fit(X_train, Y_train)
rf_peds = rf_model.predict(X_test)
print("=== Random Forest ===")
print(confusion_matrix(Y_test, rf_peds))
print(classification_report(Y_test, rf_peds))

# Step 5: Train Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, Y_train)
gb_preds = gb_model.predict(X_test)
print("=== Gradient Boosting ===")
print(confusion_matrix(Y_test, gb_preds))
print(classification_report(Y_test, gb_preds))

# Step 6: Train AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, Y_train)
ada_preds = ada_model.predict(X_test)
print("=== AdaBoost ===")
print(confusion_matrix(Y_test, ada_preds))
print(classification_report(Y_test, ada_preds))

