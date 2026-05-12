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

# Load dataset

df = pd.read_csv("fuel_prices.csv")

# Keep only first 5 columns

df = df.iloc[:, :5]

# Rename columns

df.columns = [
    "Year",
    "Month",
    "PetrolPrice",
    "DieselPrice",
    "CrudeOilIndex"
]

# Remove repeated header rows

df = df[df["Year"] != "Year"]

# Clean crude oil column

df["CrudeOilIndex"] = (
    df["CrudeOilIndex"]
    .astype(str)
    .str.replace("r", "", regex=False)
)

# Convert columns to numeric

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

# Drop missing values

df = df.dropna()

# Convert year to integer

df["Year"] = df["Year"].astype(int)

# Create date column

df["Date"] = pd.to_datetime(
    df["Year"].astype(str) + "-" + df["Month"].astype(str),
    format="%Y-%B"
)

# Create autoregressive feature

df["DieselPrevMonth"] = df["DieselPrice"].shift(1)

# Remove missing row from shift()

df = df.dropna()

# Display cleaned dataset

print("\nCLEANED DATASET")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)

# Display summary statistics

print("\nSUMMARY STATISTICS")
print(df.describe())

# Diesel price graph

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

# Petrol price graph

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

# Crude oil index graph

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

# Correlation heatmap

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

# Train test split

train = df[df["Year"] < 2023]
test = df[df["Year"] >= 2023]

# Features and target

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

# Feature scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Linear Regression model

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_predictions = linear_model.predict(
    X_test_scaled
)

# Random Forest model

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

# Evaluation function

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

    print(f"\n{model_name}")

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

# Evaluate models

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

# Actual vs predicted graph

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

# Residual plot

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

# Random Forest feature importance

importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance":
    random_forest_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)

# Feature importance graph

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

# Save cleaned dataset

df.to_csv(
    "cleaned_fuel_prices.csv",
    index=False
)

# Final output message

print("\nPROJECT COMPLETED SUCCESSFULLY")

print("\nSaved Files:")

print("- cleaned_fuel_prices.csv")
print("- diesel_prices.png")
print("- petrol_prices.png")
print("- crude_oil_index.png")
print("- correlation_heatmap.png")
print("- predicted_vs_actual.png")
print("- residual_plot.png")
print("- feature_importance.png")
