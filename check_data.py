
import pandas as pd

# Load the customer churn dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

# Display dataset information
print("Dataset loaded successfully!")
print("Total rows and columns:", df.shape)

print("\nFirst 5 customers:")
print(df.head())

print("\nAll column names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())
