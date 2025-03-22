import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import GridSearchCV
from scipy import stats
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv('merged_data.csv')

# Drop date columns (assuming column names contain 'date' or 'time')
date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
df = df.drop(columns=date_columns)

# Extract labels
y = df['meteostat_value'].values
X = df.drop(columns=['meteostat_value'])

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, shuffle=True)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'degree': [2, 3, 4, 5],
    'kernel': ['linear', 'rbf', 'poly']
}

xgb_params = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'colsample_bylevel': [0.5, 0.7, 1.0],
    'colsample_bynode': [0.5, 0.7, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0],
    'gamma': [0, 0.1, 0.5, 1.0],
    'min_child_weight': [1, 3, 5, 7]
}

# param_grid = {
#   'n_estimators': [50, 100, 200],
#   'learning_rate': [0.01, 0.1, 0.3],
#   'max_depth': [3, 5, 7],
#   'subsample': [0.7, 0.9],
#   'colsample_bytree': [0.7, 0.9]
# }


# Create a base model
tuner = GridSearchCV(XGBRegressor(), xgb_params, cv=5, n_jobs=-1, scoring='r2', verbose=1)
tuner.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = tuner.best_params_
print('Best parameters:', best_params)

# Score using the best model
best_model = tuner.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Evaluate training performance
y_train_pred = best_model.predict(X_train_scaled)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_evs = explained_variance_score(y_train, y_train_pred)
print('')
print('Training Mean Absolute Error:', train_mae)
print('Training Root Mean Squared Error:', train_rmse)
print('Training R^2 Score:', train_r2)
print('Training Explained Variance Score:', train_evs)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print('')
print('Testing Mean Absolute Error:', mae)
print('Testing Root Mean Squared Error:', rmse)
print('Testing R^2 Score:', r2)
print('Testing Explained Variance Score:', evs)

# concatenate the training and testing predictions
y_train_pred = y_train_pred.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)
y_pred_all = np.concatenate((y_train_pred, y_pred))

mean_true = np.mean(y)
mean_pred = np.mean(y_pred_all)
std_true = np.std(y)
std_pred = np.std(y_pred_all)
# Perform a paired t-test (if data is normally distributed) or Wilcoxon signed-rank test
# Here, we use Wilcoxon due to potential non-normality
stat, p = stats.wilcoxon(y.flatten(), y_pred_all.flatten())

print("\n -- More Stats -- \n")
print(f"Mean of actual: {mean_true}, Mean of predicted: {mean_pred}")
print(f"Standard Deviation of actual: {std_true}, value_met: {std_pred}")
print(f"Wilcoxon Test Statistic: {stat}, p-value: {p}")

# Interpret p-value
if p < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")

combined_df = pd.DataFrame({
    'actual': y.flatten(),
    'predicted': y_pred_all.flatten()
})

# Round the predictions to 1 decimal place
combined_df = combined_df.round(1)

combined_df.to_csv('data/combined_svm.csv', index=False)