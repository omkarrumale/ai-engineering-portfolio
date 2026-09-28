import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score , mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

# ---------- 1. Load the cleaned data ----------
df = pd.read_csv("housing_cleaned.csv")
# print(df.head(10))
# print(df.isnull().sum())
# print(df.describe())
# print(df.duplicated().sum())
# print(df.info())

# Fill missing total_bedrooms with the median value
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# ---------- 2. Create new features ----------
df['room_per_household'] = (df['total_rooms'] / df['households'])
df['bedroom_per_room'] = (df['total_bedrooms'] / df['total_rooms'])
df['population_per_household'] = (df['population'] / df['households'])

# ---------- 3. Log transform ----------
# These columns have a few very big values (outliers).
# log1p squeezes the big values so the data is more even.
skewed_cols = [
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'room_per_household',
    'population_per_household'
]
 
for col in skewed_cols:
    df[col] = np.log1p(df[col])

# ---------- 4. Split into features and target, then train and test ----------
x = df.drop('median_house_value', axis = 1)
y = df['median_house_value']

# 80% data for training, 20% for testing
X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

# ---------- 5. One-hot encode ocean_proximity ----------
# Converts text categories into 0/1 columns so the model can use them
# drop = "first" removes one column to avoid duplicate information
# handle_unknown = 'ignore' avoids error if test data has a new category
encoder = OneHotEncoder(
    sparse_output=False,
    drop = "first",
    handle_unknown='ignore'
)

# fit_transform on train (learns categories), only transform on test
X_train_encoded = encoder.fit_transform(X_train[['ocean_proximity']])
X_test_ecoder = encoder.transform(X_test[['ocean_proximity']])

# ---------- 6. Join numerical columns + encoded columns ----------
numerical_cols = [
    'longitude',
    'latitude', 
    'housing_median_age', 
    'total_rooms',
    'total_bedrooms', 
    'population', 
    'households', 
    'median_income',
    'room_per_household',
    'population_per_household',
    'bedroom_per_room'

    
]

x_train_numerical = X_train[numerical_cols]
x_test_numerical = X_test[numerical_cols]

# np.hstack joins the columns side by side
x_train_final = np.hstack([
    x_train_numerical,
    X_train_encoded
])

x_test_final = np.hstack([
    x_test_numerical,
    X_test_ecoder
])

# ---------- 7. Scaling ----------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_final)  
x_test_scaled = scaler.transform(x_test_final)        

# ---------- 8. Polynomial features ----------
# degree = 2 adds squares and column combinations so the model can learn curves
# fit on train only, transform on test (no data leakage)
poly = PolynomialFeatures(degree=2 , include_bias = False)
x_train_poly = poly.fit_transform(x_train_final)
x_test_poly = poly.transform(x_test_final)

# ---------- 9. Train the model ----------
model = LinearRegression()
model.fit(x_train_poly, y_train)

# ---------- 10. Predict on test data and check the score ----------
y_pred = model.predict(x_test_poly)

# ------ Testing -------
r2 = r2_score(y_test , y_pred)
print("R2_SCORE:" , r2)

mae = mean_absolute_error(y_test , y_pred)
print("Mean_absolute_error:" , mae)

rmse = root_mean_squared_error(y_test , y_pred)
print("root_mean_squared_error:" , rmse)