
import pandas as pd

# Load original dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Fill missing TotalCharges with the median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove customer ID because it is not a useful prediction feature
df = df.drop(columns=["customerID"])

# Convert target labels into numbers
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Save cleaned dataset
df.to_csv("data/cleaned_churn.csv", index=False)

print("\nData cleaning completed!")
print("Cleaned dataset shape:", df.shape)

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nMissing values remaining:", df.isnull().sum().sum())

print("\nFirst 5 cleaned records:")
print(df.head())
