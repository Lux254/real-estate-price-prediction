# Predicting Real Estate Prices Using Random Forest Regression

This project utilizes a **Random Forest Regression model** to predict real estate prices based on various features, such as location, property age, and proximity to amenities. The model demonstrates strong predictive capabilities and highlights key insights into the factors driving housing prices.

---

## Features
- **Robust Data Preprocessing**: Addressed outliers, transformed skewed data, and extracted meaningful temporal features.
- **Feature Engineering**: Created derived features like proximity-convenience scores and performed geospatial clustering.
- **Random Forest Model**: Leveraged the model's ability to handle non-linear relationships.
- **Evaluation Metrics**: Achieved high accuracy with metrics such as R-squared (0.778) and Mean Absolute Error (MAE: 4.71).

---

## Dataset
- **Source**: Kaggle
- **Key Features**:
  - **X1**: Transaction date
  - **X2**: House age
  - **X3**: Distance to the nearest MRT station
  - **X4**: Number of nearby convenience stores
  - **X5**: Latitude
  - **X6**: Longitude
  - **Y**: House price of unit area (target variable)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/real-estate-price-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd real-estate-price-prediction
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Workflow

### 1. **Data Preprocessing**
- Handled missing values and outliers.
- Applied transformations to reduce skewness (e.g., log and square root transformations).
- Extracted temporal features (year, month, quarter) from transaction dates.

### 2. **Feature Engineering**
- **Proximity-Convenience Score**: Combined distance to MRT and nearby convenience stores.
- **Geospatial Clustering**: Clustered latitude and longitude into neighborhood categories.

### 3. **Model Training**
- Used Random Forest Regressor with 100 estimators.

### 4. **Model Evaluation**
- Metrics:
  - **Mean Absolute Error (MAE)**: 4.71
  - **Mean Squared Error (MSE)**: 42.75
  - **R-squared (R²)**: 0.778
- Visualized predicted vs. actual housing prices using scatter plots.

---

## Results
- **Performance**:
  - The model achieved an R-squared value of 0.778, demonstrating strong predictive capability.
- **Visualization**:
  - A scatter plot demonstrated close alignment between predicted and actual housing prices.

---

## Usage

1. **Run Preprocessing**:
   ```bash
   python scripts/data_preprocessing.py
   ```

2. **Train the Model**:
   ```bash
   python scripts/train_model.py
   ```

3. **View Results**:
   Generated visualizations and metrics are saved in the `results/` directory.

---

## Future Directions
- Incorporate additional datasets, such as economic indicators or property details.
- Explore advanced models like Gradient Boosting or XGBoost.
- Deploy the model using Flask or Streamlit for real-time predictions.

---

## Contact
For questions or collaborations:
- **Email**: shadrackohungo@gmail.com

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.
