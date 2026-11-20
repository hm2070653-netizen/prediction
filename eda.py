
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_churn.csv")

sns.set_theme(style="whitegrid")

# Graph 1: Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Stayed, 1 = Left)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/churn_distribution.png")
plt.show()

# Graph 2: Churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/churn_by_contract.png")
plt.show()

# Graph 3: Churn by tenure
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn (0 = Stayed, 1 = Left)")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.savefig("data/churn_by_tenure.png")
plt.show()

# Basic statistics
print("Total customers:", len(df))
print("\nChurn percentage:")
print((df["Churn"].mean() * 100).round(2), "%")

print("\nAverage tenure by churn status:")
print(df.groupby("Churn")["tenure"].mean().round(2))
