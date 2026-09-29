# California House Price Prediction using Linear Regression

**R² Score: 0.711 | MAE: 37,977 | RMSE: 53,506**

I used the California Housing dataset to predict `median_house_value` with Linear
Regression, and learned the full regression workflow along the way.

## Results

| Model | Test R² | MAE | RMSE |
|---|---:|---:|---:|
| Basic Linear Regression | 0.658 | 42,731 | 58,223 |
| + Polynomial Features (degree 2) | **0.711** | **37,977** | **53,506** |

- The final model explains about 71.1% of the variation in house prices.
- The average error ($37.9k) is about 19% of the mean house value ($193k).


## Dataset

[California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices)]


- **Target:** `median_house_value`

- **Features:** `longitude`, 
`latitude`, 
`housing_median_age`, 
`total_rooms`,
`total_bedrooms`, 
`population`, 
`households`, 
`median_income`, 
`ocean_proximity`

- **Engineered features:** 
`room_per_household`, 
`bedroom_per_room`,
`population_per_household`

## Workflow

```text
Load data → Feature engineering → Log transform → Train/test split (80/20)
→ Impute + scale + one-hot encode (fitted on train only)
→ Polynomial features (degree 2) → Linear Regression → Evaluate
```

Preprocessing is done inside a scikit-learn `Pipeline` so nothing from the test
set leaks into training.

## How to Run

```bash
pip install -r requirements.txt
python classical_ml/linear_regression.py
```

## Tech Stack

Python, pandas, NumPy, scikit-learn