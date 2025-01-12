import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import zscore, skew
import seaborn as sns
import matplotlib.pyplot as plt

# Define functions
def fractional_year_to_date(fractional_year):
    """Convert fractional year format to datetime."""
    year = int(fractional_year)
    days = round((fractional_year - year) * 365)
    date = datetime(year, 1, 1) + timedelta(days=days)
    return date

def reduce_skewness(column):
    """
    Reduce skewness in a given column dynamically.
    Uses log1p, square root, or no transformation based on skewness score.
    """
    skewness_score = skew(column)
    if skewness_score > 1:  # Highly positively skewed
        return np.log1p(column)
    elif 0.5 < skewness_score <= 1:  # Moderately positively skewed
        return np.sqrt(column)
    elif skewness_score < -1:  # Highly negatively skewed
        max_val = column.max()
        return np.sqrt(max_val - column)
    else:
        return column  # No transformation for nearly symmetric data

# Load dataset
file_path = r"C:\Users\USER\Desktop\My Projects\RFR Repository\data\raw\Real_estate.csv"  # Replace with your dataset's file path 
estate_data = pd.read_csv(file_path)

# Convert transaction date
estate_data['X1 transaction date'] = estate_data['X1 transaction date'].apply(fractional_year_to_date)

# Drop unnecessary columns and separate features
features = estate_data.drop(columns=['X1 transaction date', 'No'])

# Outlier detection and removal using Z-score
z_scores = features.apply(zscore)
outliers = (z_scores.abs() > 3).any(axis=1)
clean_data = estate_data[~outliers]

# Save clean data for further use
clean_data.to_csv("clean_data.csv", index=False)

# Calculate and plot correlation heatmap
correlation_matrix = features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    linewidths=0.5
)
plt.title('Correlation Heatmap')
plt.show()

# Check and transform skewness dynamically
skewness = features.apply(lambda x: skew(x), axis=0)
print("Skewness before transformation:\n", skewness)

# Dynamically reduce skewness
for column in features.columns:
    features[column] = reduce_skewness(features[column])

# Re-check skewness after transformation
transformed_skewness = features.apply(lambda x: skew(x), axis=0)
print("Skewness after transformation:\n", transformed_skewness)

# Save preprocessed features for modeling
features.to_csv(r"C:\Users\USER\Desktop\My Projects\RFR Repository\data\raw\preprocessed_features.csv", index=False)
