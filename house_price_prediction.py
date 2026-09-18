# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("Loading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. EXPLORE DATA
# --------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: MedHouseVal")


# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures normalized successfully!")


# --------------------------------------------------
# 6. TRAIN LINEAR REGRESSION MODEL
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train_scaled, y_train)

print("Linear Regression model trained successfully!")


# --------------------------------------------------
# 7. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "-" * 40)
print("MODEL PERFORMANCE")
print("-" * 40)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# --------------------------------------------------
# 9. ACTUAL VS PREDICTED PRICES
# --------------------------------------------------

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nFirst 10 predictions:")
print(results.head(10))


# --------------------------------------------------
# 10. VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test,
    y=y_pred,
    alpha=0.5
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


# --------------------------------------------------
# 11. TEST YOUR OWN HOUSE
# --------------------------------------------------

print("\n" + "-" * 40)
print("TEST YOUR OWN HOUSE")
print("-" * 40)

print("\nEnter the following house details:")

MedInc = float(input("Median income: "))
HouseAge = float(input("House age: "))
AveRooms = float(input("Average rooms: "))
AveBedrms = float(input("Average bedrooms: "))
Population = float(input("Population: "))
AveOccup = float(input("Average occupancy: "))
Latitude = float(input("Latitude: "))
Longitude = float(input("Longitude: "))

new_house = pd.DataFrame({
    "MedInc": [MedInc],
    "HouseAge": [HouseAge],
    "AveRooms": [AveRooms],
    "AveBedrms": [AveBedrms],
    "Population": [Population],
    "AveOccup": [AveOccup],
    "Latitude": [Latitude],
    "Longitude": [Longitude]
})

new_house_scaled = scaler.transform(new_house)

predicted_price = model.predict(new_house_scaled)[0]

print(f"\nPredicted House Price: ${predicted_price * 100000:,.2f}")