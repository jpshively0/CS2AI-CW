
# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    cross_val_score,
    TimeSeriesSplit
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from statsmodels.tsa.arima.model import ARIMA


# functions

def SNR(df, column_name):

    signal = df[column_name].mean()

    noise = df[column_name].std()

    if noise == 0 or pd.isna(noise):
        return 0

    return signal / noise


def evaluate_model(y_true, predictions, model_name):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    print(f"\n{model_name}")

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")


# load dataset

df = pd.read_csv("4.1.2.csv")


# remove unnamed columns

df = df.loc[
    :,
    ~df.columns.str.contains("^Unnamed")
]


# keep first 5 columns

df = df.iloc[:, :5]


# rename columns

df.columns = [

    "PetrolPrice",
    "SuperUnleaded",
    "PremiumUnleaded",
    "CrudeOilIndex",
    "Year"

]


# convert to numeric

numeric_columns = [

    "PetrolPrice",
    "SuperUnleaded",
    "PremiumUnleaded",
    "CrudeOilIndex",
    "Year"

]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# remove invalid years

df = df[
    (df["Year"] >= 1970) &
    (df["Year"] <= 2025)
]


# sort by year

df = df.sort_values(
    by="Year"
)


# reset index

df = df.reset_index(
    drop=True
)


# fix corrupted premium values

df.loc[
    df["PremiumUnleaded"] > 1000,
    "PremiumUnleaded"
] = (
    df.loc[
        df["PremiumUnleaded"] > 1000,
        "PremiumUnleaded"
    ] / 100
)


# replace false zeros

columns_to_clean = [

    "PetrolPrice",
    "PremiumUnleaded",
    "CrudeOilIndex"

]

for column in columns_to_clean:

    df[column] = (
        df[column]
        .replace(0, np.nan)
    )


# interpolate missing values

for column in columns_to_clean:

    df[column] = (
        df[column]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )


# remove null values

df = df.dropna()


# reset index

df = df.reset_index(
    drop=True
)


# create time index

df["TimeIndex"] = df["Year"]


# feature engineering

df["PremiumPrevYear"] = (
    df["PremiumUnleaded"]
    .shift(1)
)

df["PremiumRollingMean3"] = (
    df["PremiumUnleaded"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

df["PremiumChange"] = (
    df["PremiumUnleaded"]
    .pct_change()
)

df["CrudeOilChange"] = (
    df["CrudeOilIndex"]
    .pct_change()
)


# remove feature nulls

df = df.dropna()

df = df.reset_index(
    drop=True
)


# dataset information

print("\nDATASET SIZE")

print(df.shape)

print("\nSIGNAL TO NOISE RATIO")

print(
    SNR(
        df,
        "PremiumUnleaded"
    )
)


# summary statistics

print("\nSUMMARY STATISTICS")

print(
    df.describe()
)


# calculate mean median mode

mean_price = (
    df["PremiumUnleaded"]
    .mean()
)

median_price = (
    df["PremiumUnleaded"]
    .median()
)

mode_price = (
    df["PremiumUnleaded"]
    .mode()[0]
)


# print central tendency

print("\nMEAN")
print(mean_price)

print("\nMEDIAN")
print(median_price)

print("\nMODE")
print(mode_price)


# create year ticks

year_ticks = range(
    int(df["Year"].min()),
    int(df["Year"].max()) + 1,
    2
)


# premium price graph

plt.figure(figsize=(12, 6))

plt.plot(
    df["TimeIndex"],
    df["PremiumUnleaded"],
    linewidth=2
)

plt.title(
    "Premium Unleaded Prices Over Time"
)

plt.xlabel("Year")

plt.ylabel(
    "Premium Unleaded Price"
)

plt.xticks(
    year_ticks,
    rotation=45
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "premium_prices.png"
)

plt.close()


# petrol price graph

plt.figure(figsize=(12, 6))

plt.plot(
    df["TimeIndex"],
    df["PetrolPrice"],
    linewidth=2
)

plt.title(
    "Petrol Prices Over Time"
)

plt.xlabel("Year")

plt.ylabel(
    "Petrol Price"
)

plt.xticks(
    year_ticks,
    rotation=45
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "petrol_prices.png"
)

plt.close()


# crude oil graph

plt.figure(figsize=(12, 6))

plt.plot(
    df["TimeIndex"],
    df["CrudeOilIndex"],
    linewidth=2
)

plt.title(
    "Crude Oil Index Over Time"
)

plt.xlabel("Year")

plt.ylabel(
    "Crude Oil Index"
)

plt.xticks(
    year_ticks,
    rotation=45
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "crude_oil.png"
)

plt.close()


# mean median mode graph

plt.figure(figsize=(14, 7))

plt.plot(
    df["TimeIndex"],
    df["PremiumUnleaded"],
    label="Premium Unleaded",
    linewidth=2
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

plt.title(
    "Premium Unleaded Prices with Mean Median Mode"
)

plt.xlabel("Year")

plt.ylabel(
    "Premium Unleaded Price"
)

plt.xticks(
    year_ticks,
    rotation=45
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "mean_median_mode.png"
)

plt.close()


# shewhart control chart

mean_value = (
    df["PremiumUnleaded"]
    .mean()
)

std_value = (
    df["PremiumUnleaded"]
    .std()
)

upper_limit = (
    mean_value +
    (3 * std_value)
)

lower_limit = (
    mean_value -
    (3 * std_value)
)

plt.figure(figsize=(14, 7))

plt.plot(
    df["TimeIndex"],
    df["PremiumUnleaded"],
    label="Premium Unleaded",
    linewidth=2
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
    mean_value,
    linestyle="-.",
    label="Mean"
)

plt.title(
    "Shewhart Control Chart"
)

plt.xlabel("Year")

plt.ylabel(
    "Premium Unleaded Price"
)

plt.xticks(
    year_ticks,
    rotation=45
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "shewhart.png"
)

plt.close()


# correlation heatmap

plt.figure(figsize=(10, 8))

correlation_columns = [

    "PetrolPrice",
    "PremiumUnleaded",
    "CrudeOilIndex",
    "PremiumPrevYear",
    "PremiumRollingMean3",
    "PremiumChange",
    "CrudeOilChange"

]

sns.heatmap(

    df[
        correlation_columns
    ].corr(),

    annot=True,
    cmap="rocket",
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png"
)

plt.close()


# train test split

train = df[
    df["Year"] < 2018
]

test = df[
    df["Year"] >= 2018
]


# select features

X_train = train[[

    "PetrolPrice",
    "CrudeOilIndex",
    "PremiumPrevYear",
    "PremiumRollingMean3",
    "CrudeOilChange"

]]

X_test = test[[

    "PetrolPrice",
    "CrudeOilIndex",
    "PremiumPrevYear",
    "PremiumRollingMean3",
    "CrudeOilChange"

]]


# select target variable

y_train = train[
    "PremiumUnleaded"
]

y_test = test[
    "PremiumUnleaded"
]


# feature scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# linear regression model

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_predictions = (
    linear_model.predict(
        X_test_scaled
    )
)

evaluate_model(
    y_test,
    linear_predictions,
    "Linear Regression"
)


# decision tree model

tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_predictions = (
    tree_model.predict(
        X_test
    )
)

evaluate_model(
    y_test,
    tree_predictions,
    "Decision Tree"
)


# random forest model

forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

forest_model.fit(
    X_train,
    y_train
)

forest_predictions = (
    forest_model.predict(
        X_test
    )
)

evaluate_model(
    y_test,
    forest_predictions,
    "Random Forest"
)


# arima model

arima_model = ARIMA(
    train["PremiumUnleaded"],
    order=(3, 1, 2)
)

arima_fit = arima_model.fit()

arima_predictions = (
    arima_fit.forecast(
        steps=len(test)
    )
)

evaluate_model(
    y_test,
    arima_predictions,
    "ARIMA"
)


# time series cross validation

time_split = TimeSeriesSplit(
    n_splits=5
)

cross_scores = cross_val_score(

    linear_model,

    X_train_scaled,

    y_train,

    cv=time_split
)

print("\nCROSS VALIDATION")

print(cross_scores)

print("\nAVERAGE CV SCORE")

print(
    cross_scores.mean()
)


# actual vs predicted graph

plt.figure(figsize=(14, 7))

plt.plot(
    test["TimeIndex"],
    y_test,
    marker="o",
    linewidth=2,
    label="Actual"
)

plt.plot(
    test["TimeIndex"],
    linear_predictions,
    marker="o",
    label="Linear Regression"
)

plt.plot(
    test["TimeIndex"],
    tree_predictions,
    marker="o",
    label="Decision Tree"
)

plt.plot(
    test["TimeIndex"],
    forest_predictions,
    marker="o",
    label="Random Forest"
)

plt.plot(
    test["TimeIndex"],
    arima_predictions,
    marker="o",
    label="ARIMA"
)

plt.title(
    "Actual vs Predicted"
)

plt.xlabel("Year")

plt.ylabel(
    "Premium Unleaded Price"
)

plt.xticks(
    year_ticks,
    rotation=45
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "predicted_vs_actual.png"
)

plt.close()


# residual plot

residuals = (
    y_test -
    linear_predictions
)

plt.figure(figsize=(10, 6))

plt.scatter(
    linear_predictions,
    residuals
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title(
    "Residual Plot - Linear Regression"
)

plt.xlabel(
    "Predicted Values"
)

plt.ylabel(
    "Residuals"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "residual_plot.png"
)

plt.close()


# feature importance graph

importance = (
    forest_model
    .feature_importances_
)

feature_names = (
    X_train.columns
)

plt.figure(figsize=(8, 6))

plt.bar(
    feature_names,
    importance
)

plt.title(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png"
)

plt.close()


# create results table

results = pd.DataFrame({

    "Model": [

        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "ARIMA"

    ],

    "MAE": [

        mean_absolute_error(
            y_test,
            linear_predictions
        ),

        mean_absolute_error(
            y_test,
            tree_predictions
        ),

        mean_absolute_error(
            y_test,
            forest_predictions
        ),

        mean_absolute_error(
            y_test,
            arima_predictions
        )

    ],

    "RMSE": [

        np.sqrt(
            mean_squared_error(
                y_test,
                linear_predictions
            )
        ),

        np.sqrt(
            mean_squared_error(
                y_test,
                tree_predictions
            )
        ),

        np.sqrt(
            mean_squared_error(
                y_test,
                forest_predictions
            )
        ),

        np.sqrt(
            mean_squared_error(
                y_test,
                arima_predictions
            )
        )

    ],

    "R2": [

        r2_score(
            y_test,
            linear_predictions
        ),

        r2_score(
            y_test,
            tree_predictions
        ),

        r2_score(
            y_test,
            forest_predictions
        ),

        r2_score(
            y_test,
            arima_predictions
        )

    ]

})


# print results table

print("\nMODEL RESULTS TABLE")

print(results)


# save results table

results.to_csv(
    "model_results.csv",
    index=False
)


# save cleaned dataset

df.to_csv(
    "cleaned_fuel_prices.csv",
    index=False
)


# final output

print("\nPROJECT COMPLETED SUCCESSFULLY")

print("\nSaved Files:")

print("- premium_prices.png")
print("- petrol_prices.png")
print("- crude_oil.png")
print("- mean_median_mode.png")
print("- shewhart.png")
print("- correlation_heatmap.png")
print("- predicted_vs_actual.png")
print("- residual_plot.png")
print("- feature_importance.png")
print("- model_results.csv")
print("- cleaned_fuel_prices.csv")
