import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ------------------------------
# Load the training data
# ------------------------------
print("Loading training data...")
train_df = pd.read_csv('train.csv')

# check and remove duplicates
dupes = train_df.duplicated().sum()
print(f"Duplicate rows found: {dupes}")
if dupes > 0:
    train_df = train_df.drop_duplicates()
print(f"Shape after dropping duplicates: {train_df.shape}")

# convert date columns
train_df['Order_Placed_Date'] = pd.to_datetime(train_df['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
train_df['Delivery_Date'] = pd.to_datetime(train_df['Delivery_Date'], format='%m/%d/%y', errors='coerce')

# calculate shipping duration
train_df['Shipping_Duration'] = (train_df['Delivery_Date'] - train_df['Order_Placed_Date']).dt.days

# numeric and categorical columns
num_cols = [
    'Supplier_Reliability','Equipment_Height','Equipment_Width','Equipment_Weight',
    'Equipment_Value','Base_Transport_Fee','Shipping_Duration'
]
cat_cols = [
    'Supplier_Name','Equipment_Type','CrossBorder_Shipping','Urgent_Shipping',
    'Installation_Service','Transport_Method','Fragile_Equipment','Hospital_Info','Rural_Hospital'
]
target_col = 'Transport_Cost'

# make sure numeric columns are numeric
for c in num_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors='coerce')

# drop missing target or duration
train_df = train_df.dropna(subset=[target_col, 'Shipping_Duration'])

# keep Hospital_Id for reference
X = train_df[num_cols + cat_cols + ['Hospital_Id']]
y = train_df[target_col]

# ------------------------------
# preprocessing
# ------------------------------
num_proc = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_proc = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_proc, num_cols),
    ('cat', cat_proc, cat_cols)
])

# ------------------------------
# split the data
# ------------------------------
X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train_full.drop(columns=['Hospital_Id'])
X_val = X_val_full.drop(columns=['Hospital_Id'])

# ------------------------------
# model training + tuning
# ------------------------------
print("Running RandomizedSearchCV for KNN...")

knn = KNeighborsRegressor()
param_dist = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11, 15, 19],
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2]
}

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', knn)
])

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print(f"Best CV RMSE: {-search.best_score_:.3f}")

best_model = search.best_estimator_

# ------------------------------
# validation performance
# ------------------------------
y_pred = best_model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
r2_val = r2_score(y_val, y_pred)
print(f"Validation RMSE: {rmse_val:.2f}")
print(f"Validation R2: {r2_val:.3f}")

# ------------------------------
# test set prediction
# ------------------------------
print("Loading test data...")
test_df = pd.read_csv('test.csv')

test_df['Order_Placed_Date'] = pd.to_datetime(test_df['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
test_df['Delivery_Date'] = pd.to_datetime(test_df['Delivery_Date'], format='%m/%d/%y', errors='coerce')
test_df['Shipping_Duration'] = (test_df['Delivery_Date'] - test_df['Order_Placed_Date']).dt.days

for c in num_cols:
    test_df[c] = pd.to_numeric(test_df[c], errors='coerce')

X_test = test_df[num_cols + cat_cols + ['Hospital_Id']].copy()
X_test_features = X_test.drop(columns=['Hospital_Id'])

print("Predicting on test set...")
test_preds = best_model.predict(X_test_features)

out = pd.DataFrame({
    'Hospital_Id': X_test['Hospital_Id'],
    'Transport_Cost_Predicted': test_preds
})
out.to_csv('test_transport_predictions_knn.csv', index=False)
print("Saved predictions to test_transport_predictions_knn.csv")