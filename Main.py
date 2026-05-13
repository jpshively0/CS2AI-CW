
# Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# SNR

def SNR(df, column_name):

    signal = df[column_name].mean()

    noise = df[column_name].std()

    ratio = signal / noise

    return ratio

# Imputation

def Imp(df, column_name):

    return df[column_name].fillna(
        method='ffill',
        inplace=True
    )

# Alternative imputation

def ImpAlt(df, column_name):

    df[column_name] = (
        df[column_name]
        .interpolate()
    )

    return df[column_name]

# Normalisation

def MidMaxNormal(df, column_names):

    scaler = MinMaxScaler()

    df_normal = df.copy()

    df_normal[column_names] = (
        scaler.fit_transform(
            df_normal[column_names]
        )
    )

    return df_normal

# Table transformation

def Transform(df, column_name, target):

    df[column_name] = (
        df[column_name]
        .astype(int)
    )

    column_avg = (
        df.groupby(column_name)[[target]]
        .mean()
    )

    return column_avg

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

# Imputation

ImpAlt(df, "CrudeOilIndex")

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

# Create rolling mean feature

df["DieselRollingMean3"] = (
    df["DieselPrice"]
    .rolling(window=3)
    .mean()
)

# Remove missing rows created by lagging

df = df.dropna()

# Signal-to-noise ratio

snr_value = SNR(
    df,
    "DieselPrice"
)

print("\nSIGNAL TO NOISE RATIO")
print(snr_value)

# Min-Max normalization

df = MidMaxNormal(
    df,
    [
        "PetrolPrice",
        "DieselPrice",
        "CrudeOilIndex"
    ]
)

# Table transformation

year_average = Transform(
    df,
    "Year",
    "DieselPrice"
)

print("\nYEARLY DIESEL PRICE AVERAGE")
print(year_average.head())

# Display cleaned dataset

print("\nCLEANED DATASET")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)

# Display summary statistics

print("\nSUMMARY STATISTICS")
print(df.describe())

# Mean median and mode

mean_price = df["DieselPrice"].mean()

median_price = df["DieselPrice"].median()

mode_price = df["DieselPrice"].mode()[0]

print("\nMEAN DIESEL PRICE")
print(mean_price)

print("\nMEDIAN DIESEL PRICE")
print(median_price)

print("\nMODE DIESEL PRICE")
print(mode_price)

# Diesel graph with mean median mode

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["DieselPrice"],
    label="Diesel Price"
)

plt.axhline(
    mean_price,
    linestyle="--",
    label="Mean"
)

plt.axhline(
    median_price,
    linestyle=":",
    label="Median"
)

plt.axhline(
    mode_price,
    linestyle="-.",
    label="Mode"
)

plt.title("Diesel Prices with Mean Median Mode")

plt.xlabel("Date")

plt.ylabel("Diesel Price")

plt.legend()

plt.tight_layout()

plt.savefig("diesel_mean_median_mode.png")

plt.close()

# Shewhart control chart

upper_limit = (
    mean_price +
    (3 * df["DieselPrice"].std())
)

lower_limit = (
    mean_price -
    (3 * df["DieselPrice"].std())
)

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["DieselPrice"],
    label="Diesel Price"
)

plt.axhline(
    upper_limit,
    linestyle="--",
    label="Upper Limit"
)

plt.axhline(
    lower_limit,
    linestyle="--",
    label="Lower Limit"
)

plt.axhline(
    mean_price,
    linestyle="-.",
    label="Mean"
)

plt.title("Shewhart Control Chart")

plt.xlabel("Date")

plt.ylabel("Diesel Price")

plt.legend()

plt.tight_layout()

plt.savefig("shewhart_chart.png")

plt.close()

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
        "DieselPrevMonth",
        "DieselRollingMean3"
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

# Original dataset features

X_train_original = train[[
    "PetrolPrice",
    "CrudeOilIndex"
]]

X_test_original = test[[
    "PetrolPrice",
    "CrudeOilIndex"
]]

# Modified dataset features

X_train_modified = train[[
    "PetrolPrice",
    "CrudeOilIndex",
    "DieselPrevMonth",
    "DieselRollingMean3"
]]

X_test_modified = test[[
    "PetrolPrice",
    "CrudeOilIndex",
    "DieselPrevMonth",
    "DieselRollingMean3"
]]

# Target variable

y_train = train["DieselPrice"]
y_test = test["DieselPrice"]

# Feature scaling

scaler = StandardScaler()

X_train_original_scaled = scaler.fit_transform(
    X_train_original
)

X_test_original_scaled = scaler.transform(
    X_test_original
)

X_train_modified_scaled = scaler.fit_transform(
    X_train_modified
)

X_test_modified_scaled = scaler.transform(
    X_test_modified
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

# Linear Regression - Original Data

linear_original = LinearRegression()

linear_original.fit(
    X_train_original_scaled,
    y_train
)

linear_original_predictions = linear_original.predict(
    X_test_original_scaled
)

evaluate_model(
    y_test,
    linear_original_predictions,
    "Linear Regression - Original Data"
)

# Linear Regression - Modified Data

linear_modified = LinearRegression()

linear_modified.fit(
    X_train_modified_scaled,
    y_train
)

linear_modified_predictions = linear_modified.predict(
    X_test_modified_scaled
)

evaluate_model(
    y_test,
    linear_modified_predictions,
    "Linear Regression - Modified Data"
)

# Decision Tree - Original Data

decision_tree_original = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

decision_tree_original.fit(
    X_train_original,
    y_train
)

decision_tree_original_predictions = (
    decision_tree_original.predict(
        X_test_original
    )
)

evaluate_model(
    y_test,
    decision_tree_original_predictions,
    "Decision Tree - Original Data"
)

# Decision Tree - Modified Data

decision_tree_modified = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

decision_tree_modified.fit(
    X_train_modified,
    y_train
)

decision_tree_modified_predictions = (
    decision_tree_modified.predict(
        X_test_modified
    )
)

evaluate_model(
    y_test,
    decision_tree_modified_predictions,
    "Decision Tree - Modified Data"
)

# Cross validation

cross_scores = cross_val_score(
    linear_modified,
    X_train_modified_scaled,
    y_train,
    cv=5
)

print("\nCROSS VALIDATION SCORES")
print(cross_scores)

print("\nAVERAGE CROSS VALIDATION SCORE")
print(cross_scores.mean())

# Actual vs predicted graph

plt.figure(figsize=(14,7))

plt.plot(
    test["Date"],
    y_test,
    label="Actual Diesel Prices"
)

plt.plot(
    test["Date"],
    linear_modified_predictions,
    label="Linear Regression Predictions"
)

plt.plot(
    test["Date"],
    decision_tree_modified_predictions,
    label="Decision Tree Predictions"
)

plt.title("Actual vs Predicted Diesel Prices")

plt.xlabel("Date")
plt.ylabel("Diesel Price")

plt.legend()

plt.tight_layout()

plt.savefig("predicted_vs_actual.png")

plt.close()

# Residual plot

residuals = y_test - linear_modified_predictions

plt.figure(figsize=(10,6))

plt.scatter(
    linear_modified_predictions,
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

# Decision Tree feature importance

importance = pd.DataFrame({

    "Feature": X_train_modified.columns,

    "Importance":
    decision_tree_modified.feature_importances_

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

plt.title("Decision Tree Feature Importance")

plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.close()

# Results table

results = pd.DataFrame({

    "Model": [

        "Linear Regression Original",
        "Linear Regression Modified",
        "Decision Tree Original",
        "Decision Tree Modified"

    ],

    "MAE": [

        mean_absolute_error(
            y_test,
            linear_original_predictions
        ),

        mean_absolute_error(
            y_test,
            linear_modified_predictions
        ),

        mean_absolute_error(
            y_test,
            decision_tree_original_predictions
        ),

        mean_absolute_error(
            y_test,
            decision_tree_modified_predictions
        )

    ],

    "RMSE": [

        mean_squared_error(
            y_test,
            linear_original_predictions
        ) ** 0.5,

        mean_squared_error(
            y_test,
            linear_modified_predictions
        ) ** 0.5,

        mean_squared_error(
            y_test,
            decision_tree_original_predictions
        ) ** 0.5,

        mean_squared_error(
            y_test,
            decision_tree_modified_predictions
        ) ** 0.5

    ],

    "R2": [

        r2_score(
            y_test,
            linear_original_predictions
        ),

        r2_score(
            y_test,
            linear_modified_predictions
        ),

        r2_score(
            y_test,
            decision_tree_original_predictions
        ),

        r2_score(
            y_test,
            decision_tree_modified_predictions
        )

    ]

})

print("\nMODEL RESULTS TABLE")

print(results)

results.to_csv(
    "model_results.csv",
    index=False
)

print("\nMODEL RESULTS SAVED")
print("model_results.csv")

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
print("- diesel_mean_median_mode.png")
print("- shewhart_chart.png")
print("- model_results.csv")
# Main menu system

def main():

    while True:

        print("\n================================")
        print("CS2AI FUEL PRICE FORECASTING")
        print("================================")

        print("1 - Display Dataset")
        print("2 - Summary Statistics")
        print("3 - Signal-to-Noise Ratio")
        print("4 - Imputation")
        print("5 - Normalisation")
        print("6 - Table Transformation")
        print("7 - Evaluate Linear Regression")
        print("8 - Evaluate Decision Tree")
        print("9 - Show Diesel Price Graph")
        print("10 - Show Correlation Heatmap")
        print("0 - Exit")

        choice = input("\nSelect an option: ")

        # Display dataset

        if choice == "1":

            print("\nDATASET")
            print(df.head())

        # Summary statistics

        elif choice == "2":

            print("\nSUMMARY STATISTICS")
            print(df.describe())

        # Signal to noise ratio

        elif choice == "3":

            snr_value = SNR(
                df,
                "DieselPrice"
            )

            print("\nSIGNAL TO NOISE RATIO")
            print(snr_value)

        # Imputation

        elif choice == "4":

            ImpAlt(
                df,
                "CrudeOilIndex"
            )

            print("\nIMPUTATION COMPLETE")

            print(
                df["CrudeOilIndex"]
                .isnull()
                .sum()
            )

        # Normalisation

        elif choice == "5":

            df_normal = MidMaxNormal(
                df,
                [
                    "PetrolPrice",
                    "DieselPrice",
                    "CrudeOilIndex"
                ]
            )

            print("\nNORMALISED DATA")

            print(
                df_normal[[
                    "PetrolPrice",
                    "DieselPrice",
                    "CrudeOilIndex"
                ]].head()
            )

        # Table transformation

        elif choice == "6":

            year_average = Transform(
                df,
                "Year",
                "DieselPrice"
            )

            print("\nYEARLY DIESEL PRICE AVERAGE")

            print(year_average.head())

        # Linear Regression results

        elif choice == "7":

            evaluate_model(
                y_test,
                linear_modified_predictions,
                "Linear Regression - Modified Data"
            )

        # Decision Tree results

        elif choice == "8":

            evaluate_model(
                y_test,
                decision_tree_modified_predictions,
                "Decision Tree - Modified Data"
            )

        # Diesel price graph

        elif choice == "9":

            plt.figure(figsize=(12,6))

            plt.plot(
                df["Date"],
                df["DieselPrice"]
            )

            plt.title(
                "UK Diesel Prices Over Time"
            )

            plt.xlabel("Date")

            plt.ylabel(
                "Diesel Price (Pence per litre)"
            )

            plt.show()

        # Correlation heatmap

        elif choice == "10":

            plt.figure(figsize=(8,6))

            sns.heatmap(
                df[[
                    "PetrolPrice",
                    "DieselPrice",
                    "CrudeOilIndex",
                    "DieselPrevMonth",
                    "DieselRollingMean3"
                ]].corr(),
                annot=True
            )

            plt.title("Correlation Heatmap")

            plt.show()

        # Exit program

        elif choice == "0":

            print("\nPROGRAM CLOSED")
            break

        # Invalid input

        else:

            print("\nINVALID OPTION")

# Run program

main()

