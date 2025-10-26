import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load and preprocess training data ---
logging.info("Loading training dataset...")
train = pd.read_csv('train.csv')

# Convert dates
train['Order_Placed_Date'] = pd.to_datetime(train['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
train['Delivery_Date'] = pd.to_datetime(train['Delivery_Date'], format='%m/%d/%y', errors='coerce')

# Fix swapped dates
mask = train['Delivery_Date'] < train['Order_Placed_Date']
train.loc[mask, ['Order_Placed_Date', 'Delivery_Date']] = train.loc[mask, ['Delivery_Date', 'Order_Placed_Date']].values
logging.info(f"Swapped dates for {mask.sum()} records where Delivery_Date < Order_Placed_Date")

# Create derived feature
train['Shipping_Duration'] = (train['Delivery_Date'] - train['Order_Placed_Date']).dt.days

# Define columns
num_cols = [
    'Supplier_Reliability','Equipment_Height','Equipment_Width','Equipment_Weight',
    'Equipment_Value','Base_Transport_Fee','Shipping_Duration'
]

cat_cols = [
    'Supplier_Name','Equipment_Type','CrossBorder_Shipping','Urgent_Shipping',
    'Installation_Service','Transport_Method','Fragile_Equipment',
    'Hospital_Info','Rural_Hospital'
]

target = 'Transport_Cost'

# Ensure numeric columns are numeric
for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')

# Drop missing target or duration
train = train.dropna(subset=[target, 'Shipping_Duration'])

# Split data
X = train[num_cols + cat_cols + ['Hospital_Id']]
y = train[target]

# --- Preprocessing pipeline ---
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

numeric_transformer = Pipeline(steps=[('imputer', num_imputer), ('scaler', scaler)])
categorical_transformer = Pipeline(steps=[('imputer', cat_imputer), ('onehot', onehot)])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# --- Train/validation split ---
X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train_full.drop(columns=['Hospital_Id'])
X_val = X_val_full.drop(columns=['Hospital_Id'])

# --- Ridge Regression ---
ridge = Ridge(max_iter=10000, random_state=42)

param_grid = {
    'regressor__alpha': [0.1, 1, 10, 100, 500]
}

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', ridge)])

logging.info("Tuning Ridge hyperparameters...")
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
best_rmse = -search.best_score_

logging.info(f"Best Ridge RMSE (CV): {best_rmse:.2f}")
logging.info(f"Best parameters: {search.best_params_}")

# --- Validation performance ---
y_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
logging.info(f"Validation RMSE: {rmse:.2f}")
logging.info(f"Validation R² Score: {r2:.3f}")

# --- Test prediction ---
logging.info("Loading and preprocessing test data...")
test = pd.read_csv('test.csv')
test['Order_Placed_Date'] = pd.to_datetime(test['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
test['Delivery_Date'] = pd.to_datetime(test['Delivery_Date'], format='%m/%d/%y', errors='coerce')

# Fix swapped dates
mask_test = test['Delivery_Date'] < test['Order_Placed_Date']
test.loc[mask_test, ['Order_Placed_Date', 'Delivery_Date']] = test.loc[mask_test, ['Delivery_Date', 'Order_Placed_Date']].values
logging.info(f"Swapped dates for {mask_test.sum()} records where Delivery_Date < Order_Placed_Date")

test['Shipping_Duration'] = (test['Delivery_Date'] - test['Order_Placed_Date']).dt.days

for col in num_cols:
    test[col] = pd.to_numeric(test[col], errors='coerce')

X_test = test[num_cols + cat_cols + ['Hospital_Id']]
X_test_features = X_test.drop(columns=['Hospital_Id'])

logging.info("Predicting on test data using Ridge Regression...")
test_preds = best_model.predict(X_test_features)

# --- Save predictions ---
output_df = pd.DataFrame({
    'Hospital_Id': X_test['Hospital_Id'],
    'Transport_Cost_Predicted': test_preds
})
output_df.to_csv('test_transport_predictions_ridge.csv', index=False)
logging.info("Saved predictions to 'test_transport_predictions_ridge.csv'")
