import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Prompt the user for the CSV file name
csv_file = input("Enter the CSV file name: ")

# Load the data
data = pd.read_csv(csv_file)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')  # Sort the data by date

# Select Features and Target
features = ['temp', 'visibility', 'winddir', 'windspeed', 'precip', 'humidity', 'solarradiation', 'cloudcover']
target = 'Daily Mean PM2.5 Concentration'

X = data[features]
y = data[target]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Create a base model
xgb = XGBRegressor(random_state=30)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Best estimator
best_xgb = grid_search.best_estimator_

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Use the best estimator for prediction
y_pred = best_xgb.predict(X_test_scaled)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
accuracy_like_metric = np.mean(np.abs(y_test - y_pred) <= 5)
print(f'Accuracy-like metric (with tolerance ±5): {accuracy_like_metric:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared Score: {r2:.2f}')

# Feature Importance Plot
feature_importance = best_xgb.feature_importances_
sorted_idx = feature_importance.argsort()
plt.figure(figsize=(10, 6))
plt.barh(np.array(features)[sorted_idx], feature_importance[sorted_idx])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from XGBoost')
plt.show()

# Correlation Heatmap
corr_matrix = data[features + [target]].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Density Heatmap
plt.figure(figsize=(10, 6))
ax = sns.kdeplot(x=y_test, y=y_pred, cmap="Blues", fill=True, cbar=True)
ax.set_xlabel('Actual PM2.5')
ax.set_ylabel('Predicted PM2.5')
ax.set_title('Density Heatmap from XGBoost')
plt.xlim(-5, 50)
plt.ylim(-5, 50)
plt.show()

# 2D Histogram
plt.figure(figsize=(10, 6))
plt.hist2d(y_test, y_pred, bins=(30, 30), cmap='Blues')
plt.colorbar(label='Frequency')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Actual vs Predicted PM2.5 2D Histogram from XGBoost')
plt.show()

joblib.dump(best_xgb, f'xgboost_model_{csv_file.split(".")[0]}.pkl')