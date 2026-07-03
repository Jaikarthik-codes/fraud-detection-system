## Step 1 - Data Exploration (ULB Dataset)

Dataset: creditcard.csv
Shape: 284,807 rows x 31 columns
Columns: Time, V1-V28 (PCA transformed), Amount, Class
No missing values in any column.

Class distribution:
- Not fraud (0): 284,315 = 99.83%
- Fraud (1): 492 = 0.17%

Far more imbalanced than Iteration 1 (9.4% fraud vs 0.17% fraud).
Dumb model accuracy would be 99.83% — even more misleading than before.

## Step 2 - Preprocessing (ULB Dataset)

Columns scaled: Amount and Time
Why: V1-V28 are already scaled from PCA transformation.
Amount and Time are raw values — different scales would bias the model.
Used StandardScaler: transforms values to mean=0, standard deviation=1.

Split: 80% training (227,845 rows) / 20% testing (56,962 rows)
stratify=Y preserves 0.17% fraud ratio in both sets.

SMOTE applied to training data only:
Before: Fraud = 394, Not fraud = 227,451
After: Fraud = 227,451, Not fraud = 227,451 (perfectly balanced)

Why SMOTE after split and not before:
Test data must reflect reality (0.17% fraud).
If we SMOTEd before splitting, test data would contain fake synthetic 
fraud cases — evaluation numbers would not reflect real performance.

Why SMOTE instead of just copying fraud cases:
Copying the same 394 cases adds no new information.
SMOTE creates realistic synthetic cases between existing fraud cases,
giving the model genuinely new patterns to learn from.