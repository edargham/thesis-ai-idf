import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('merged_data.csv')

# Drop date columns
date_columns = [col for col in df.columns if 'date' in col.lower()]
df = df.drop(columns=date_columns)

# Define features and target
X = df.drop(columns=['meteostat_value'])
y = df['meteostat_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
  'n_estimators': [50, 100, 200],
  'learning_rate': [0.01, 0.1, 0.3],
  'max_depth': [3, 5, 7],
  'subsample': [0.7, 0.9],
  'colsample_bytree': [0.7, 0.9]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
  estimator=xgb.XGBRegressor(),
  param_grid=param_grid,
  cv=5,
  scoring='neg_mean_squared_error',
  verbose=1,
  n_jobs=-1
)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
  'Feature': X.columns,
  'Importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)