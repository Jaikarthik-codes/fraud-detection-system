import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('../data/processed/creditcard_cleaned.csv')
x = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, Y_train_sm = smote.fit_resample(X_train, Y_train)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train_sm, Y_train_sm)
gb_model.fit(X_train_sm, Y_train_sm)
ada_model.fit(X_train_sm, Y_train_sm)

# Predictions
rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)
ada_preds = ada_model.predict(X_test)

# Probabilities for AUC
rf_proba = rf_model.predict_proba(X_test)[:, 1]
gb_proba = gb_model.predict_proba(X_test)[:, 1]
ada_proba = ada_model.predict_proba(X_test)[:, 1]

print("=== Random Forest ===")
print(confusion_matrix(Y_test, rf_preds))
print(classification_report(Y_test, rf_preds))
print(f"AUC-ROC: {roc_auc_score(Y_test, rf_proba):.4f}")

print("=== Gradient Boosting ===")
print(confusion_matrix(Y_test, gb_preds))
print(classification_report(Y_test, gb_preds))
print(f"AUC-ROC: {roc_auc_score(Y_test, gb_proba):.4f}")

print("=== AdaBoost ===")
print(confusion_matrix(Y_test, ada_preds))
print(classification_report(Y_test, ada_preds))
print(f"AUC-ROC: {roc_auc_score(Y_test, ada_proba):.4f}")

print("\n=== Summary ===")
results = {
    'Model': ['Random Forest', 'Gradient Boosting', 'AdaBoost'],
    'Precision': [
        classification_report(Y_test, rf_preds, output_dict=True)['1']['precision'],
        classification_report(Y_test, gb_preds, output_dict=True)['1']['precision'],
        classification_report(Y_test, ada_preds, output_dict=True)['1']['precision']
    ],
    'Recall': [
        classification_report(Y_test, rf_preds, output_dict=True)['1']['recall'],
        classification_report(Y_test, gb_preds, output_dict=True)['1']['recall'],
        classification_report(Y_test, ada_preds, output_dict=True)['1']['recall']
    ],
    'F1': [
        classification_report(Y_test, rf_preds, output_dict=True)['1']['f1-score'],
        classification_report(Y_test, gb_preds, output_dict=True)['1']['f1-score'],
        classification_report(Y_test, ada_preds, output_dict=True)['1']['f1-score']
    ],
    'AUC-ROC': [
        roc_auc_score(Y_test, rf_proba),
        roc_auc_score(Y_test, gb_proba),
        roc_auc_score(Y_test, ada_proba)
    ]
}

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))