import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the enhanced dataset
file_path = r"C:\Users\USER\Desktop\My Projects\RFR Repository\data\processed\enhanced_data.csv"  # Replace with your enhanced dataset file path
clean_data = pd.read_csv(file_path)

# Drop unnecessary columns
new_features_1 = clean_data.drop(columns=['No', 'X1 transaction date'])

# Split data into features (X) and target (Y)
X = new_features_1.drop(columns=['Y house price of unit area'])
y = new_features_1['Y house price of unit area']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

# Save the trained model
model_filename = "random_forest_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")