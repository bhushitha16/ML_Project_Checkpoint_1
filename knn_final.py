import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading training dataset...")
train = pd.read_csv('train.csv')

# --- DUPLICATE HANDLING ---
num_duplicates = train.duplicated().sum()
logging.info(f"Number of duplicate rows: {num_duplicates}")
train = train.drop_duplicates()
logging.info(f"Shape after removing duplicates: {train.shape}")

# Convert date columns to datetime
train['Order_Placed_Date'] = pd.to_datetime(train['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
train['Delivery_Date'] = pd.to_datetime(train['Delivery_Date'], format='%m/%d/%y', errors='coerce')

# Feature: shipping duration in days
train['Shipping_Duration'] = (train['Delivery_Date'] - train['Order_Placed_Date']).dt.days

# Numeric and categorical columns
num_cols = ['Supplier_Reliability','Equipment_Height','Equipment_Width','Equipment_Weight',
            'Equipment_Value','Base_Transport_Fee','Shipping_Duration']
cat_cols = ['Supplier_Name','Equipment_Type','CrossBorder_Shipping','Urgent_Shipping',
            'Installation_Service','Transport_Method','Fragile_Equipment','Hospital_Info','Rural_Hospital']
target = 'Transport_Cost'

# Ensure numeric columns are numeric
for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')

# Drop rows with missing target or shipping duration (essential)
train = train.dropna(subset=[target, 'Shipping_Duration'])

# Include Hospital_Id for reference
X = train[num_cols + cat_cols + ['Hospital_Id']]
y = train[target]

# --- Preprocessing Pipelines ---
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

# --- TRAIN-VALIDATION SPLIT (80-20) ---
X_train_full, X_val_full, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Drop Hospital_Id from features
X_train = X_train_full.drop(columns=['Hospital_Id'])
X_val = X_val_full.drop(columns=['Hospital_Id'])

# --- KNN Regressor ---
logging.info("Training KNN Regressor...")
knn_model = KNeighborsRegressor()

param_grid = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11, 15, 19],
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2]
}

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', knn_model)])

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X_train, y_train)

best_knn = search.best_estimator_
best_rmse = -search.best_score_
logging.info(f"Best KNN parameters: {search.best_params_} with CV RMSE: {best_rmse:.2f}")

# --- Validation Performance ---
y_pred = best_knn.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
r2_val = r2_score(y_val, y_pred)
logging.info(f"Validation RMSE: {rmse_val:.2f}")
logging.info(f"Validation R2 Score: {r2_val:.3f}")

# --- Test Data ---
logging.info("Loading test data...")
test = pd.read_csv('test.csv')
test['Order_Placed_Date'] = pd.to_datetime(test['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
test['Delivery_Date'] = pd.to_datetime(test['Delivery_Date'], format='%m/%d/%y', errors='coerce')
test['Shipping_Duration'] = (test['Delivery_Date'] - test['Order_Placed_Date']).dt.days

for col in num_cols:
    test[col] = pd.to_numeric(test[col], errors='coerce')

X_test = test[num_cols + cat_cols + ['Hospital_Id']].copy()
X_test_features = X_test.drop(columns=['Hospital_Id'])

logging.info("Predicting with KNN...")
test_preds = best_knn.predict(X_test_features)

output_df = pd.DataFrame({
    'Hospital_Id': X_test['Hospital_Id'],
    'Transport_Cost_Predicted': test_preds
})
output_df.to_csv('test_transport_predictions_knn.csv', index=False)
logging.info("Saved predictions to 'test_transport_predictions_knn.csv'")
