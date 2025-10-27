# Vehicle Sales Prediction & Analysis

## 📊 Overview

This project is a comprehensive data science analysis of vehicle sales data, focusing on data preprocessing, exploratory data analysis, and machine learning model implementation to predict vehicle selling prices. The project demonstrates the complete data science lifecycle from data cleaning to predictive modeling.

## 🎯 Project Objectives

- Clean and preprocess vehicle sales data containing missing values and outliers
- Perform exploratory data analysis to uncover trends and patterns
- Build and evaluate machine learning models to predict vehicle selling prices
- Provide insights into vehicle market dynamics and pricing factors

## 📋 Dataset

The dataset contains historical vehicle sales records with the following features:

- **Year**: Manufacturing year of the vehicle
- **Make**: Car brand name
- **Model**: Specific model of the car
- **Trim**: Variant or version of the model
- **Body**: Type of vehicle (SUV, Sedan, etc.)
- **Transmission**: Transmission type (Automatic or Manual)
- **VIN**: Vehicle Identification Number (unique identifier)
- **State**: State where the car was sold
- **Condition**: Condition of the car at the time of sale
- **Odometer**: Number of miles driven
- **Color**: Exterior color
- **Interior**: Interior color/material
- **Seller**: Seller information
- **MMR**: Manheim Market Report value (price benchmark)
- **Selling Price**: Actual selling price (target variable)
- **Sale Date**: Date of sale

## 🛠️ Technologies Used

- **Python**: Core programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualizations
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting algorithm
- **CuML**: GPU-accelerated machine learning (for Random Forest)

## 📝 Key Features

### 1. Data Preprocessing
- **Column Standardization**: Capitalized and renamed columns for consistency
- **State Code Mapping**: Converted state codes to full state names (including US states and Canadian provinces)
- **Brand Standardization**: Clustered and standardized similar brand names
- **Missing Value Handling**: Implemented intelligent filling strategies:
  - Model, Trim, and Body: Filled based on similar vehicle profiles
  - Market Price & Selling Price: Filled using mean values of similar vehicles
  - Color and Interior: Filled with top-occurring values
- **Duplicate Removal**: Removed duplicate records based on VIN
- **Outlier Removal**: Applied IQR (Interquartile Range) method and Isolation Forest algorithm

### 2. Exploratory Data Analysis
- **Statistical Summary**: Descriptive statistics for all features
- **Visualization of Top Brands**: Analysis of most popular car brands
- **Price Distribution Analysis**: Box plots and distributions for market prices
- **Correlation Analysis**: Relationship between Market Price and Selling Price
- **Geographic Analysis**: Top 10 states by sales volume with pie charts

### 3. Statistical Hypothesis Testing
- **Transmission Analysis**: T-test to compare prices between Automatic and Manual transmissions
- **Odometer Impact**: Z-test to analyze the effect of high vs. low mileage on selling prices

### 4. Machine Learning Models

#### Implemented Models:
1. **Polynomial Regression** (degree=2)
   - R² Score: 0.95
   - Explained 95% of variance in selling prices
   
2. **XGBoost Regressor**
   - R² Score: 0.95
   - Strong predictive performance
   
3. **Random Forest Regressor**
   - R² Score: 0.93 (standard)
   - R² Score: 0.95 (with GridSearchCV optimization)
   - GPU-accelerated implementation using CuML

#### Preprocessing Pipeline:
- One-hot encoding for categorical variables
- Standard scaling for numerical features
- Train-test split (80-20)

## 📊 Key Insights

1. **Strong Price Correlation**: Market Price and Selling Price show a correlation coefficient of approximately 0.97, indicating high predictability
2. **Brand Analysis**: Luxury brands (BMW, etc.) show higher median prices with wider price ranges
3. **Popular Brands**: Ford, Chevrolet, and Nissan are among the most frequently sold brands
4. **Geographic Distribution**: Sales are concentrated in specific states, with California and Texas leading
5. **Transmission Impact**: Statistical tests reveal significant differences in prices based on transmission type

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/Vehicle_sales.git
cd Vehicle_sales
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

**Optional**: For GPU acceleration with CuML (requires CUDA)
```bash
pip install cuml-cu11  # For CUDA 11
# or
pip install cuml-cu12  # For CUDA 12
```

### Running the Project

1. Download the dataset
   - Place the `car_prices.csv` file in the project directory
   - Update the file path in the notebook if necessary

2. Launch Jupyter Notebook
```bash
jupyter notebook
```

3. Open and run `Vehicle_Sales_Final.ipynb`

## 📁 Project Structure

```
Vehicle_sales/
├── Vehicle_Sales_Final.ipynb    # Main Jupyter notebook
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── index.html                    # HTML export of the notebook
```

## 📈 Model Performance

| Model | R² Score | MAE | MSE |
|-------|----------|-----|-----|
| Polynomial Regression | 0.95 | ~1019.52 | ~2,268,165 |
| XGBoost Regressor | 0.95 | - | - |
| Random Forest | 0.93-0.95 | - | - |

## 🔍 Files Description

- **Vehicle_Sales_Final.ipynb**: Complete data analysis and modeling pipeline
- **README.md**: Project documentation
- **index.html**: HTML version of the notebook for easy viewing

## 🎓 Methodology

1. **Data Collection**: Load and inspect the raw dataset
2. **Data Cleaning**: Handle missing values, duplicates, and standardization
3. **Data Transformation**: Encode categorical variables and scale numerical features
4. **Outlier Detection**: Apply IQR and Isolation Forest methods
5. **Exploratory Analysis**: Visualizations and statistical tests
6. **Feature Engineering**: Prepare data for modeling
7. **Model Training**: Implement multiple regression algorithms
8. **Model Evaluation**: Assess performance using R², MAE, and MSE metrics
9. **Visualization**: Generate plots for predictions and residuals

## 💡 Future Enhancements

- Implement deep learning models (Neural Networks)
- Add time series analysis for price trends
- Deploy model as a web application
- Add more sophisticated feature engineering
- Implement ensemble methods combining multiple models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).


⭐ If you found this project helpful, please give it a star!

