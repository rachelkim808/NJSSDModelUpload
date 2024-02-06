import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import joblib

# Define features
features = ['temp', 'visibility', 'winddir', 'windspeed', 'precip', 'humidity', 'solarradiation', 'cloudcover']

# Function to get user input for features
def get_user_input(features):
    user_input = []
    for feature in features:
        value = float(input(f"Enter {feature}: "))
        user_input.append(value)
    return user_input

# Function for bootstrapping confidence estimation
def bootstrap_prediction_confidence(model, X, y, user_input_scaled, num_samples=100, is_ann=False):
    bootstrap_preds = []
    for _ in range(num_samples):
        sample_indices = np.random.choice(range(len(X)), len(X), replace=True)
        sample_X = X.iloc[sample_indices]
        sample_y = y[sample_indices] if not is_ann else y
        if is_ann:
            preds = model.predict(sample_X)[:, 0]
            user_pred = model.predict(user_input_scaled)[:, 0][0]
        else:
            preds = model.predict(sample_X)
            user_pred = model.predict(user_input_scaled)[0]
        bootstrap_preds.append(np.mean(preds))
    std_dev = np.std(bootstrap_preds)
    return user_pred, std_dev

# Function to map month to season
def map_month_to_season(month):
    seasons = {
        'winter': ['December', 'January', 'February'],
        'spring': ['March', 'April', 'May'],
        'summer': ['June', 'July', 'August'],
        'fall': ['September', 'October', 'November']
    }
    for season, months in seasons.items():
        if month.title() in months:
            return season
    return None

# Ask user for the month
user_month = input("Enter the month: ")
season = map_month_to_season(user_month)

if season is None:
    print("Invalid month entered.")
else:
    # Load the corresponding models for the season
    best_rf = joblib.load(f'random_forest_model_FortLee{season.capitalize()}DecadeMERGED_updated.pkl')
    best_xgb = joblib.load(f'xgboost_model_FortLee{season.capitalize()}DecadeMERGED.pkl')
    best_ann = load_model(f'ann_model_FortLee{season.capitalize()}DecadeMERGED.keras')

    # Load the data (You need to specify the CSV file for the corresponding season)
    csv_file = f'FortLee{season.capitalize()}DecadeMERGED.csv'
    data = pd.read_csv(csv_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Select Features and Target
    target = 'Daily Mean PM2.5 Concentration'
    X = data[features]
    y = data[target]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=60)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert data for prediction
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)
    y_test_reset = y_test.reset_index(drop=True)

    # Get user input for predictions
    user_input = get_user_input(features)

    # Convert user input into DataFrame and scale
    user_input_df = pd.DataFrame([user_input], columns=features)
    user_input_scaled = scaler.transform(user_input_df)

    # Calculate model confidence based on user input
    rf_user_pred, rf_confidence = bootstrap_prediction_confidence(best_rf, X_test_scaled_df, y_test_reset, user_input_scaled)
    xgb_user_pred, xgb_confidence = bootstrap_prediction_confidence(best_xgb, X_test_scaled_df, y_test_reset, user_input_scaled)
    ann_user_pred, ann_confidence = bootstrap_prediction_confidence(best_ann, X_test_scaled_df, y_test_reset, user_input_scaled, is_ann=True)

    # Display user input predictions and confidence
    print(f"Random Forest User Prediction: {rf_user_pred}, Confidence: {rf_confidence}")
    print(f"XGBoost User Prediction: {xgb_user_pred}, Confidence: {xgb_confidence}")
    print(f"ANN User Prediction: {ann_user_pred}, Confidence: {ann_confidence}")

    # Determine the most confident model and its prediction
    confidences = [rf_confidence, xgb_confidence, ann_confidence]
    predictions = [rf_user_pred, xgb_user_pred, ann_user_pred]
    most_confident_index = np.argmin(confidences)
    most_confident_model = ["Random Forest", "XGBoost", "ANN"][most_confident_index]
    most_confident_prediction = predictions[most_confident_index]

    print(f"\nThe most confident model is {most_confident_model} with a prediction of {most_confident_prediction}")
