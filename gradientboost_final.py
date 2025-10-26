import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------
# Function to correct dates and calculate Shipping Duration
# ------------------------
def correct_dates_and_shipping_duration(df):
    mask = df['Delivery_Date'] < df['Order_Placed_Date']
    if mask.any():
        logging.info(f"Swapping dates for {mask.sum()} records where Delivery_Date < Order_Placed_Date")
        df.loc[mask, ['Order_Placed_Date', 'Delivery_Date']] = df.loc[mask, ['Delivery_Date', 'Order_Placed_Date']].values
    df['Shipping_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    return df

# ------------------------
# Load and preprocess training data
# ------------------------
logging.info("Loading training dataset...")
train = pd.read_csv('train.csv')

# Convert dates
train['Order_Placed_Date'] = pd.to_datetime(train['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
train['Delivery_Date'] = pd.to_datetime(train['Delivery_Date'], format='%m/%d/%y', errors='coerce')

# Calculate shipping duration
train = correct_dates_and_shipping_duration(train)

# Columns
num_cols = ['Supplier_Reliability','Equipment_Height','Equipment_Width','Equipment_Weight',
            'Equipment_Value','Base_Transport_Fee','Shipping_Duration']
cat_cols = ['Supplier_Name','Equipment_Type','CrossBorder_Shipping','Urgent_Shipping',
            'Installation_Service','Transport_Method','Fragile_Equipment','Hospital_Info','Rural_Hospital']
target = 'Transport_Cost'

# Convert numeric columns to numeric type
for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')

train[target] = train[target].abs()

# Drop rows with missing target or Shipping_Duration
train = train.dropna(subset=[target, 'Shipping_Duration'])

# Optional: remove duplicates
duplicates = train.duplicated().sum()
logging.info(f"Number of duplicate rows: {duplicates}")
train = train.drop_duplicates()
logging.info(f"Shape after removing duplicates: {train.shape}")

# Features and target
X = train[num_cols + cat_cols + ['Hospital_Id']]
y = train[target]

# ------------------------
# Split dataset
# ------------------------
X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train_full.drop(columns=['Hospital_Id'])
X_val = X_val_full.drop(columns=['Hospital_Id'])

# ------------------------
# Preprocessing pipelines
# ------------------------
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

# ------------------------
# Gradient Boosting Regressor (optimized for speed)
# ------------------------
gbr_model = GradientBoostingRegressor(
    n_estimators=200,    # fewer trees for faster training
    max_depth=3,         # shallower trees
    learning_rate=0.1,   # faster convergence
    subsample=0.8,       # stochastic GB
    random_state=42
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', gbr_model)])

# Log-transform target to stabilise variance
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# ------------------------
# Train
# ------------------------
logging.info("Training Gradient Boosting Regressor...")
pipeline.fit(X_train, y_train_log)

# ------------------------
# Validation performance
# ------------------------
y_pred_val_log = pipeline.predict(X_val)
y_pred_val = np.expm1(y_pred_val_log)

rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val, y_pred_val)
logging.info(f"Validation RMSE: {rmse_val:.2f}")
logging.info(f"Validation R²: {r2_val:.3f}")

# ------------------------
# Test predictions
# ------------------------
logging.info("Loading test data...")
test = pd.read_csv('test.csv')
test['Order_Placed_Date'] = pd.to_datetime(test['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
test['Delivery_Date'] = pd.to_datetime(test['Delivery_Date'], format='%m/%d/%y', errors='coerce')
test = correct_dates_and_shipping_duration(test)

for col in num_cols:
    test[col] = pd.to_numeric(test[col], errors='coerce')

X_test = test[num_cols + cat_cols + ['Hospital_Id']].copy()
X_test_features = X_test.drop(columns=['Hospital_Id'])

logging.info("Predicting with Gradient Boosting...")
y_test_pred_log = pipeline.predict(X_test_features)
y_test_pred = np.expm1(y_test_pred_log)

# Save predictions
output_df = pd.DataFrame({
    'Hospital_Id': X_test['Hospital_Id'],
    'Transport_Cost_Predicted': y_test_pred
})
output_df.to_csv('test_transport_predictions_gbr.csv', index=False)
logging.info("Saved predictions to 'test_transport_predictions_gbr.csv'")
