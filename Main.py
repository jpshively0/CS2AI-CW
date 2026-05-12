
# ============================================
# CS2AI Fuel Price Forecasting Coursework
# FINAL VERSION
# ============================================

# Install required libraries first:
# pip install pandas matplotlib seaborn scikit-learn

# ============================================
# IMPORT LIBRARIES
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("fuel_prices.csv")

# ============================================
# KEEP ONLY FIRST 5 COLUMNS
# ============================================

df = df.iloc[:, :5]

# ============================================
# RENAME COLUMNS
# ============================================

df.columns = [
    "Year",
    "Month",
    "PetrolPrice",
    "DieselPrice",
    "CrudeOilIndex"
]

# ============================================
# REMOVE REPEATED HEADER ROWS
# ============================================

df = df[df["Year"] != "Year"]

# ============================================
# CLEAN CRUDE OIL COLUMN
# Removes values like "98.10r"
# ============================================

df["CrudeOilIndex"] = (
    df["CrudeOilIndex"]
    .astype(str)
    .str.replace("r", "", regex=False)
)

# ============================================
# CONVERT TO NUMERIC
# ============================================

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["PetrolPrice"] = pd.to_numeric(
    df["PetrolPrice"],
    errors="coerce"
)

df["DieselPrice"] = pd.to_numeric(
    df["DieselPrice"],
    errors="coerce"
)

df["CrudeOilIndex"] = pd.to_numeric(
    df["CrudeOilIndex"],
    errors="coerce"
)

# ============================================
# DROP MISSING VALUES
# ============================================

df = df.dropna()

# ============================================
# CONVERT YEAR TO INTEGER
# ============================================

df["Year"] = df["Year"].astype(int)

# ============================================
# CREATE DATE COLUMN
# ============================================

df["Date"] = pd.to_datetime(
    df["Year"].astype(str) + "-" + df["Month"].astype(str),
    format="%Y-%B"
)

# ============================================
# FEATURE ENGINEERING
# AUTOREGRESSIVE FEATURE
# ============================================

df["DieselPrevMonth"] = df["DieselPrice"].shift(1)

# Remove missing row from shift()
df = df.dropna()

# ============================================
# DISPLAY CLEANED DATASET
# ============================================

print("\n================================")
print("CLEANED DATASET")
print("================================")

print(df.head())

print("\n================================")
print("DATA TYPES")
print("================================")

print(df.dtypes)

# ============================================
# BASIC STATISTICS
# ============================================

print("\n================================")
print("SUMMARY STATISTICS")
print("================================")

print(df.describe())

# ============================================
# EXPLORATORY DATA ANALYSIS
# ============================================

# --------------------------------------------
# DIESEL PRICE OVER TIME
# --------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["DieselPrice"]
)

plt.title("UK Diesel Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Diesel Price (Pence per litre)")

plt.tight_layout()

plt.savefig("diesel_prices.png")

plt.close()

# --------------------------------------------
# PETROL PRICE OVER TIME
# --------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["PetrolPrice"]
)

plt.title("UK Petrol Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Petrol Price (Pence per litre)")

plt.tight_layout()

plt.savefig("petrol_prices.png")

plt.close()

# --------------------------------------------
# CRUDE OIL INDEX OVER TIME
# --------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["CrudeOilIndex"]
)

plt.title("Crude Oil Index Over Time")
plt.xlabel("Date")
plt.ylabel("Crude Oil Index")

plt.tight_layout()

plt.savefig("crude_oil_index.png")

plt.close()

# --------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    df[[
        "PetrolPrice",
        "DieselPrice",
        "CrudeOilIndex",
        "DieselPrevMonth"
    ]].corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.close()

# ============================================
# TRAIN / TEST SPLIT
# TIME SERIES SPLIT
# ============================================

train = df[df["Year"] < 2023]
test = df[df["Year"] >= 2023]

# ============================================
# FEATURES AND TARGET
# ============================================

X_train = train[[
    "PetrolPrice",
    "CrudeOilIndex",
    "DieselPrevMonth"
]]

y_train = train["DieselPrice"]

X_test = test[[
    "PetrolPrice",
    "CrudeOilIndex",
    "DieselPrevMonth"
]]

y_test = test["DieselPrice"]

# ============================================
# FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ============================================
# MODEL 1 - LINEAR REGRESSION
# ============================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_predictions = linear_model.predict(
    X_test_scaled
)

# ============================================
# MODEL 2 - RANDOM FOREST REGRESSOR
# ============================================

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = random_forest_model.predict(
    X_test
)

# ============================================
# EVALUATION FUNCTION
# ============================================

def evaluate_model(y_true, predictions, model_name):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = mean_squared_error(
        y_true,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_true,
        predictions
    )

    print("\n================================")
    print(model_name)
    print("================================")

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

# ============================================
# EVALUATE MODELS
# ============================================

evaluate_model(
    y_test,
    linear_predictions,
    "Linear Regression"
)

evaluate_model(
    y_test,
    random_forest_predictions,
    "Random Forest Regressor"
)

# ============================================
# ACTUAL VS PREDICTED GRAPH
# ============================================

plt.figure(figsize=(14,7))

plt.plot(
    test["Date"],
    y_test,
    label="Actual Diesel Prices"
)

plt.plot(
    test["Date"],
    linear_predictions,
    label="Linear Regression Predictions"
)

plt.plot(
    test["Date"],
    random_forest_predictions,
    label="Random Forest Predictions"
)

plt.title("Actual vs Predicted Diesel Prices")

plt.xlabel("Date")
plt.ylabel("Diesel Price")

plt.legend()

plt.tight_layout()

plt.savefig("predicted_vs_actual.png")

plt.close()

# ============================================
# RESIDUAL PLOT
# ============================================

residuals = y_test - linear_predictions

plt.figure(figsize=(10,6))

plt.scatter(
    linear_predictions,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residual Plot - Linear Regression")

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.tight_layout()

plt.savefig("residual_plot.png")

plt.close()

# ============================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance":
    random_forest_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

print(importance)

# ============================================
# FEATURE IMPORTANCE GRAPH
# ============================================

plt.figure(figsize=(8,6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Random Forest Feature Importance")

plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.close()

# ============================================
# SAVE CLEANED DATASET
# ============================================

df.to_csv(
    "cleaned_fuel_prices.csv",
    index=False
)

# ============================================
# FINAL OUTPUT MESSAGE
# ============================================

print("\n================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("================================")

print("\nSaved Files:")

print("- cleaned_fuel_prices.csv")
print("- diesel_prices.png")
print("- petrol_prices.png")
print("- crude_oil_index.png")
print("- correlation_heatmap.png")
print("- predicted_vs_actual.png")
print("- residual_plot.png")
print("- feature_importance.png")

# ============================================
# END OF PROJECT
# ============================================
