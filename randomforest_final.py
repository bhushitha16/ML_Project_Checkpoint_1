import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------
# Function: Correct dates and calculate Shipping Duration
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

# Shipping duration
train = correct_dates_and_shipping_duration(train)

# Columns
num_cols = ['Supplier_Reliability','Equipment_Height','Equipment_Width','Equipment_Weight',
            'Equipment_Value','Base_Transport_Fee','Shipping_Duration']
cat_cols = ['Supplier_Name','Equipment_Type','CrossBorder_Shipping','Urgent_Shipping',
            'Installation_Service','Transport_Method','Fragile_Equipment','Hospital_Info','Rural_Hospital']
target = 'Transport_Cost'

# Convert numeric columns
for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')

train[target] = train[target].abs()

# Drop rows with missing target or shipping duration
train = train.dropna(subset=[target, 'Shipping_Duration'])
train = train.drop_duplicates()
logging.info(f"Shape after cleaning: {train.shape}")

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
# Preprocessing
# ------------------------
# Impute numeric
num_imputer = SimpleImputer(strategy='median')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_val[num_cols] = num_imputer.transform(X_val[num_cols])

# Scale numeric
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

# Target encode categorical
cat_encoder = TargetEncoder(cols=cat_cols)
X_train[cat_cols] = cat_encoder.fit_transform(X_train[cat_cols], y_train)
X_val[cat_cols] = cat_encoder.transform(X_val[cat_cols])

# ------------------------
# Log-transform target
# ------------------------
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# ------------------------
# Random Forest Regressor
# ------------------------
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

logging.info("Training Random Forest Regressor...")
rf_model.fit(X_train, y_train_log)

# ------------------------
# Validation
# ------------------------
y_val_pred_log = rf_model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2_val = r2_score(y_val, y_val_pred)
logging.info(f"Validation RMSE: {rmse_val:.2f}, R²: {r2_val:.3f}")

# Feature importance
fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_})
fi_df = fi_df.sort_values(by='importance', ascending=False)
logging.info("Top 10 important features:\n" + str(fi_df.head(10)))

# ------------------------
# Load and preprocess test data
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

# Impute and scale numeric
X_test_features[num_cols] = num_imputer.transform(X_test_features[num_cols])
X_test_features[num_cols] = scaler.transform(X_test_features[num_cols])

# Encode categorical
X_test_features[cat_cols] = cat_encoder.transform(X_test_features[cat_cols])

# Predict
logging.info("Predicting with Random Forest...")
y_test_pred_log = rf_model.predict(X_test_features)
y_test_pred = np.expm1(y_test_pred_log)

# Save predictions
output_df = pd.DataFrame({
    'Hospital_Id': X_test['Hospital_Id'],
    'Transport_Cost_Predicted': y_test_pred
})
output_df.to_csv('test_transport_predictions_rf_updated.csv', index=False)
logging.info("Saved predictions to 'test_transport_predictions_rf_updated.csv'")
