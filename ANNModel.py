dimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import legacy
from tensorflow.keras import backend as K

# Prompt the user for the CSV file name
csv_file = input("Enter the CSV file name: ")

# Load the data
data = pd.read_csv(csv_file)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Select Features and Target
features = ['temp', 'visibility', 'winddir', 'windspeed', 'precip', 'humidity', 'solarradiation', 'cloudcover']
target = 'Daily Mean PM2.5 Concentration'

X = data[features]
y = data[target]

# Splitting the datasetC:/Users/voice/Downloads/PM25NYRESEARCH/FortLeeCentralMERGED.csv
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to create model
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(12, input_dim=len(features), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Hyperparameters
optimizers = [legacy.Adam(learning_rate=0.01), legacy.RMSprop(learning_rate=0.01)]
batch_sizes = [10, 20, 50]
epochs = [10, 50, 100]

best_score = float('inf')
best_params = {}

for optimizer in optimizers:
    for batch_size in batch_sizes:
        for epoch in epochs:
            K.clear_session()  # Clearing the session before each model creation
            # Create and compile the model
            model = create_model(optimizer=optimizer)
            # Train the model
            model.fit(X_train_scaled, y_train, epochs=epoch, batch_size=batch_size, verbose=0)
            # Evaluate the model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            # Update best score and parameters
            if mse < best_score:
                best_score = mse
                best_params = {'optimizer': optimizer, 'batch_size': batch_size, 'epochs': epoch}

# Print best parameters
print("Best Parameters:", best_params)

# Create and train model with best parameters
K.clear_session()
best_model = create_model(optimizer=best_params['optimizer'])
best_model.fit(X_train_scaled, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

# Prediction and Evaluation
y_pred = best_model.predict(X_test_scaled).squeeze()  # Use .squeeze() to convert y_pred to 1D
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
accuracy_like_metric = np.mean(np.abs(y_test - y_pred) <= 5)
print(f'Accuracy-like metric (with tolerance ±5): {accuracy_like_metric:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared Score: {r2:.2f}')

def calculate_permutation_importance(model, X, y, metric=mean_squared_error):
    baseline_performance = metric(y, model.predict(X))
    importances = []
    for col in X.columns:
        X_permuted = X.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col])
        permuted_performance = metric(y, model.predict(X_permuted))
        importance = baseline_performance - permuted_performance
        importances.append(importance)
    return np.array(importances)

# Compute permutation importance
feature_importances = calculate_permutation_importance(best_model, pd.DataFrame(X_test_scaled, columns=features), y_test)

# Plotting feature importance
sorted_idx = np.argsort(feature_importances)
plt.figure(figsize=(10, 6))
plt.barh(np.array(features)[sorted_idx], feature_importances[sorted_idx])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from ANN')
plt.show()

# Correlation Heatmap (same as before)
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
ax.set_title('Density Heatmap from ANN')
plt.xlim(-5, 50)
plt.ylim(-5, 50)
plt.show()

# 2D Histogram
plt.figure(figsize=(10, 6))
plt.hist2d(y_test, y_pred, bins=(30, 30), cmap='Blues')
plt.colorbar(label='Frequency')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Actual vs Predicted PM2.5 2D Histogram from ANN')
plt.show()

best_model.save(f'ann_model_{csv_file.split(".")[0]}.keras')
