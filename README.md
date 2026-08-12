# 🏦 Loan Approval Prediction — Machine Learning Project

A complete end-to-end Machine Learning project that predicts whether a loan application will be **Approved** or **Rejected**, based on applicant details such as income, education, credit history, and property area.

## 📁 Project Files

| File | Description |
|---|---|
| `loan_approval_prediction.ipynb` | Main Jupyter Notebook — full project with code, explanations, and charts |
| `loan_approval_prediction_source.py` | Plain Python script version of the same project |
| `loan_approval_data.csv` | Dataset used in this project (also available via direct link below) |
| `model_comparison_report.md` / `.pdf` | Report comparing all trained models |
| `README.md` | This file |

## 📊 Dataset

**Direct download link:**
https://raw.githubusercontent.com/sachin365123/CSV-files-for-Data-Science-and-Machine-Learning/main/Loan%20Approval%20Prediction.csv

This is the well-known **Loan Prediction Dataset**, originally from Analytics Vidhya's "Loan Prediction" practice hackathon. It has 614 loan applications with 13 columns, including the target column `Loan_Status` (Y = Approved, N = Rejected).

**Columns:** `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status`

## 🛠️ Steps Covered

1. **Data Collection** — downloaded directly from the link above
2. **Data Understanding** — shape, data types, missing values
3. **Exploratory Data Analysis (EDA)** — small charts for target distribution, credit history, income, and correlations
4. **Data Cleaning** — filled missing values (mode for categorical, median for numeric)
5. **Feature Engineering**
   - Encoding (Label Encoding for text columns)
   - Feature Selection (`SelectKBest`, top 8 features)
   - Scaling (`StandardScaler`)
6. **Train/Test Split** — 80/20 split
7. **Model Training** — Logistic Regression, Random Forest, Support Vector Machine (SVM)
8. **Cross Validation** — 5-fold cross validation on the training set
9. **Model Evaluation** — Accuracy, Precision, Recall, F1 Score, Confusion Matrix
10. **Model Comparison** — side-by-side table and bar chart, best model auto-selected by F1 Score

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.8618 | 0.8400 | 0.9882 | **0.9081** |
| Support Vector Machine | 0.8537 | 0.8317 | 0.9882 | 0.9032 |
| Random Forest | 0.7886 | 0.8391 | 0.8588 | 0.8488 |

**Best Model: Logistic Regression** — highest F1 Score and recall, simple and easy to explain.

## ▶️ How to Run

```bash
# Install the required libraries
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the script version
python loan_approval_prediction_source.py

# OR open the notebook
jupyter notebook loan_approval_prediction.ipynb
```

Or open `loan_approval_prediction.ipynb` directly in [Google Colab](https://colab.research.google.com/) — no installation needed.

## 📌 Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- jupyter (to view/run the notebook)

## 👤 Author

Musfirah Ather — AI student, The Islamia University of Bahawalpur
https://loan-approval-prediction-yarjpgc7nzuu4xcue3wzoe.streamlit.app/
