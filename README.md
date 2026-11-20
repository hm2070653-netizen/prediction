# Customer Churn Prediction

An end-to-end machine learning project based on the **Customer Churn Prediction – A–Z Project Guide**.

## Goal
Predict the probability that a telecom customer will cancel their subscription (churn), then provide a web app for predictions.

## Project structure
```text
churn-prediction/
├── data/                 # Put the Telco Customer Churn CSV here
├── notebooks/            # EDA and modeling notebook
│   └── customer_churn.ipynb
├── src/                  # Reusable preprocessing and training code
│   └── train.py
├── models/               # Saved trained pipeline will be stored here
├── app.py                # Streamlit prediction app
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset
Download the Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV file (usually `WA_Fn-UseC_-Telco-Customer-Churn.csv`) in `data/`.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Workflow
1. Explore the dataset in `notebooks/customer_churn.ipynb`.
2. Clean `TotalCharges`, drop `customerID`, and encode `Churn` as 1/0.
3. Train and compare Logistic Regression, Random Forest, and XGBoost.
4. Evaluate using recall, F1-score, ROC-AUC, confusion matrix, ROC and precision-recall curves.
5. Save the full preprocessing + model pipeline into `models/churn_pipeline.joblib`.
6. Run the web app:
   ```powershell
   streamlit run app.py
   ```

## Notes
- The app expects a trained pipeline in `models/churn_pipeline.joblib`.
- The notebook and training script are starter templates. Run them after placing the dataset in `data/`.
- Model performance depends on the dataset and training choices; no accuracy is claimed before training.
