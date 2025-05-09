import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from CSV files
results_df = pd.read_csv('results.csv')
transfer_df = pd.read_csv('football_transfers_players.csv')

# Filter players with more than 900 minutes played
results_df = results_df[results_df['stats_standard_Minutes'] > 900]

# Standardize player names for merging
results_df['Player'] = results_df['Player'].str.lower().str.strip()
transfer_df['player_name'] = transfer_df['player_name'].str.lower().str.strip()

# Merge the two DataFrames based on player names, keeping only matches
merged_df = pd.merge(results_df, transfer_df, left_on='Player', right_on='player_name', how='inner')

# Drop rows with missing transfer prices
merged_df = merged_df.dropna(subset=['price'])

# Save the merged DataFrame to a CSV file
merged_df.to_csv('merged_results_transfer.csv', index=False,encoding='utf-8-sig')

# Function to convert transfer price strings to floats
def convert_price(price_str):
    if pd.isna(price_str) or not isinstance(price_str, str):
        return np.nan
    price_str = price_str.replace('€', '').strip()
    if 'M' in price_str:
        return float(price_str.replace('M', ''))
    elif 'K' in price_str:
        return float(price_str.replace('K', '')) / 1000
    return np.nan

# Apply the conversion function to the 'price' column
merged_df['price'] = merged_df['price'].apply(convert_price)

# Select feature columns (stats_ columns excluding certain ones)
feature_cols = [col for col in results_df.columns if col.startswith('stats_') and col not in [
    'stats_standard_Nation', 'stats_standard_Squad', 'stats_standard_Position'
]]

# Handle the age column: extract the numeric age
def extract_age(age_str):
    if pd.isna(age_str) or not isinstance(age_str, str):
        return np.nan
    try:
        return float(age_str.split('-')[0])
    except:
        return np.nan

if 'stats_standard_Current age' in feature_cols:
    merged_df['stats_standard_Current age'] = merged_df['stats_standard_Current age'].apply(extract_age)

# Extract features
features = merged_df[feature_cols]

# Replace 'N/a' with NaN and convert to numeric
features = features.replace('N/a', np.nan)
features = features.apply(pd.to_numeric, errors='coerce')
features = features.fillna(0)

# Target variable is the converted price
target = merged_df['price']

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate loss (Mean Squared Error) and accuracy (R-squared)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R2): {r2:.2f}")

# Plotting results
plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (€M)')
plt.ylabel('Predicted Price (€M)')
plt.title('Actual vs Predicted Prices')
plt.grid(True)

# Plot 2: Error distribution
plt.subplot(1, 2, 2)
errors = y_pred - y_test
sns.histplot(errors, kde=True, color='purple')
plt.xlabel('Prediction Error (€M)')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.grid(True)

plt.tight_layout()
plt.savefig('prediction_results.png')
plt.close()

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Player': merged_df.loc[y_test.index, 'Player'],
    'Actual_Price': y_test,
    'Predicted_Price': y_pred
})
predictions_df.to_csv('price_predictions.csv', index=False,encoding='utf-8-sig')