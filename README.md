# Bedtime Screen Time & Sleep Debt — Fatigue Prediction

## Overview
This project uses machine learning to predict a person's **next-day fatigue score** based on bedtime screen time, sleep patterns, lifestyle habits, and demographic information.

## Features
- Data analysis and visualization using Pandas, Matplotlib, and Seaborn.
- Data preprocessing, missing-value imputation, scaling, and one-hot encoding.
- Training and hyperparameter tuning of multiple regression models.
- Model evaluation using MAE, MSE, RMSE, and R².
- Saving the final trained model using Joblib.

## Machine Learning Models
The project compares Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, SVR, KNN, AdaBoost, Bagging, and Voting Regressor models.

## Technologies Used
- Python
- Pandas, NumPy, DuckDB
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

## Dataset
`bedtime_screentime_sleep_debt.csv`

The dataset includes features such as bedtime phone usage, caffeine intake, physical activity, sleep duration, sleep latency, and sleep stages.

## How to Run
1. Clone this repository.
2. Install the required libraries:

   ```bash
   pip install pandas numpy duckdb scikit-learn matplotlib seaborn joblib
   ```

3. Place `bedtime_screentime_sleep_debt.csv` in the project directory.
4. Run the Python script.

The trained model is saved
