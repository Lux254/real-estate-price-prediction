import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Load the preprocessed dataset
file_path = r"C:\Users\USER\Desktop\My Projects\RFR Repository\data\processed\clean_estate_data.xlsx"  # Replace with your preprocessed dataset file path
clean_data = pd.read_excel(file_path)

# Ensure the transaction date column is in datetime format
clean_data['X1 transaction date'] = pd.to_datetime(clean_data['X1 transaction date'])

# 1. Create Proximity-Convenience Score
clean_data['Proximity_Convenience_Score'] = (
    clean_data['X3 distance to the nearest MRT station'] * (clean_data['X4 number of convenience stores'] + 1)
)

# 2. Group locations into clusters using KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clean_data['Neighborhood_Cluster'] = kmeans.fit_predict(clean_data[['X5 latitude', 'X6 longitude']])

# 3. Extract features from the transaction date
clean_data['Year'] = clean_data['X1 transaction date'].dt.year
clean_data['Month'] = clean_data['X1 transaction date'].dt.month
clean_data['Quarter'] = clean_data['X1 transaction date'].dt.quarter

# 4. Correlation heatmap for new features
new_features = clean_data.drop(columns=['X1 transaction date', 'No'])  # Adjust as needed
corr_matrix = new_features.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Save the enhanced dataset with new features
clean_data.to_csv(r"C:\Users\USER\Desktop\My Projects\RFR Repository\data\processed\enhanced_data.csv", index=False)