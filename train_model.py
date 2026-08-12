"""
train_model.py
----------------
This script prepares the final Loan Approval Prediction model for deployment.
It repeats the same cleaning / encoding / feature selection / scaling steps
from the Week 4 notebook, trains the final model (Logistic Regression - the
best model from our comparison), and saves everything the Streamlit app
needs to make predictions later.

Run this once:  python train_model.py
"""

import os
import urllib.request
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------------
# Step 1: Make sure the output folder exists
# ------------------------------------------------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ------------------------------------------------------------------
# Step 2: Download the dataset (same direct link as before)
# ------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/sachin365123/CSV-files-for-Data-Science-and-Machine-Learning/main/Loan%20Approval%20Prediction.csv"
DATA_PATH = "data/loan_approval_data.csv"

if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Dataset downloaded to", DATA_PATH)
else:
    print("Dataset already exists at", DATA_PATH)

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------------
# Step 3: Clean the data (same as Week 4)
# ------------------------------------------------------------------
data = df.copy()
data = data.drop('Loan_ID', axis=1)

categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

numeric_cols = ['LoanAmount', 'Loan_Amount_Term']
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# ------------------------------------------------------------------
# Step 4: Encode categorical (text) columns into numbers
# ------------------------------------------------------------------
cols_to_encode = ['Gender', 'Married', 'Dependents', 'Education',
                   'Self_Employed', 'Property_Area', 'Loan_Status']

encoders = {}
for col in cols_to_encode:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# ------------------------------------------------------------------
# Step 5: Split into features (X) and target (y)
# ------------------------------------------------------------------
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# ------------------------------------------------------------------
# Step 6: Feature selection - keep the top 8 features
# ------------------------------------------------------------------
selector = SelectKBest(score_func=f_classif, k=8)
selector.fit(X, y)
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask].tolist()
print("Selected features:", selected_features)

X = X[selected_features]

# ------------------------------------------------------------------
# Step 7: Scale the features
# ------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ------------------------------------------------------------------
# Step 8: Train/test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# Step 9: Train all 3 models and collect metrics (for the app's
# "Model Performance" section, so we can show the full comparison)
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    metrics[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "CV Accuracy": cv_scores.mean()
    }
    print(name, "->", metrics[name])

# ------------------------------------------------------------------
# Step 10: Pick the final model - Logistic Regression (best F1 Score
# from our Week 4 comparison) - and save everything needed for the app
# ------------------------------------------------------------------
FINAL_MODEL_NAME = "Logistic Regression"
final_model = models[FINAL_MODEL_NAME]

joblib.dump(final_model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(selected_features, "model/feature_names.pkl")
joblib.dump(metrics, "model/metrics.pkl")
joblib.dump(FINAL_MODEL_NAME, "model/final_model_name.pkl")

print("\nAll files saved inside the 'model' folder:")
print(" - model.pkl (final trained model)")
print(" - scaler.pkl (StandardScaler)")
print(" - encoders.pkl (LabelEncoders for text columns)")
print(" - feature_names.pkl (the 8 selected feature names, in order)")
print(" - metrics.pkl (accuracy/precision/recall/F1/CV for all 3 models)")
print(" - final_model_name.pkl (name of the chosen model)")
