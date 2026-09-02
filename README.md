# ==========================================
# 1. IMPORTS & SETUP
# ==========================================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visual formatting style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Image folder automatically setup karne ke liye
os.makedirs("screenshots", exist_ok=True)


# ==========================================
# 2. DATASET INFORMATION & LOADING
# ==========================================
# (Demo ke liye synthetic dataset create kar rahe hain - apna CSV yahan replace karein)
# df = pd.read_csv('data/your_dataset.csv')

np.random.seed(42)
dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
categories = np.random.choice(
    ["Electronics", "Clothing", "Home & Kitchen"], size=100
)
sales = np.random.randint(100, 1000, size=100)
profit = sales * np.random.uniform(0.1, 0.4, size=100)

df = pd.DataFrame(
    {
        "Date": dates,
        "Category": categories,
        "Sales": sales,
        "Profit": profit,
    }
)

print("--- DATASET INFORMATION ---")
print(df.info())
print("\n--- FIRST 5 ROWS ---")
print(df.head())


# ==========================================
# 3. ANALYSIS PERFORMED (DATA CLEANING & EDA)
# ==========================================
# Missing values check
print("\nMissing Values Count:")
print(df.isnull().sum())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Category-wise Aggregation
category_summary = (
    df.groupby("Category")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values(by="Sales", ascending=False)
)


# ==========================================
# 4. GRAPHS & SCREENSHOT EXPORT
# ==========================================

# Graph 1: Sales Trend Over Time
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Sales"], color="#2b5c8f", linewidth=2, label="Sales")
plt.title("Daily Sales Trend Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.tight_layout()
# Screenshot Auto-Save
plt.savefig("screenshots/trend_graph.png", dpi=300)
plt.show()

# Graph 2: Category-wise Sales & Profit (Bivariate Analysis)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=category_summary,
    x="Category",
    y="Sales",
    hue="Category",
    palette="Blues_d",
    legend=False,
)
plt.title("Total Sales by Product Category", fontsize=14, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Total Sales ($)")
plt.tight_layout()
# Screenshot Auto-Save
plt.savefig("screenshots/category_sales.png", dpi=300)
plt.show()

# Graph 3: Correlation Heatmap
plt.figure(figsize=(6, 4))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
# Screenshot Auto-Save
plt.savefig("screenshots/heatmap.png", dpi=300)
plt.show()


# ==========================================
# 5. KEY FINDINGS (PRINT OUTPUT)
# ==========================================
top_cat = category_summary.iloc[0]["Category"]
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()

print("\n================ KEY FINDINGS ================")
print(f"1. Overall Total Revenue: ${total_sales:,.2f}")
print(f"2. Overall Total Profit:  ${total_profit:,.2f}")
print(f"3. Top Performing Category: '{top_cat}' with highest total sales.")
print(
    f"4. Average Profit Margin:  {(total_profit / total_sales) * 100:.2f}%"
)
print("==============================================")