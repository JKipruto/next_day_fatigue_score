import os
import pandas as pd
import duckdb as db
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

ohecoder = OneHotEncoder()
conn = db.connect("My_database.duckdb")
if os.path.exists("bedtime_screentime_sleep_debt.csv"):
  # The target from my model will be next_day_fatigue_score-
    print("File Available")
    bssd_df = conn.sql("""select age,gender,occupation_type,chronotype,bedtime_phone_minutes,primary_bedtime_app,
    screen_brightness_pct,blue_light_filter_active,caffeine_post_5pm_mg,physical_activity_min,sleep_latency_min,total_sleep_hours,deep_sleep_pct,rem_sleep_pct,
    morning_alarm_snoozes,next_day_fatigue_score from 'bedtime_screentime_sleep_debt.csv'""").df()
    non_binary_sex = conn.sql(
        """select * from'bedtime_screentime_sleep_debt.csv' where gender not in ('Male','Female')""").df()

    print(bssd_df.isnull().sum())
    print(non_binary_sex.count())
    print(bssd_df.isna().sum())
    print(bssd_df.dtypes)

    numeric_columns = ["age", "bedtime_phone_minutes", "screen_brightness_pct",
                       "caffeine_post_5pm_mg", "physical_activity_min", "sleep_latency_min", "total_sleep_hours", "deep_sleep_pct",
                       "rem_sleep_pct", "morning_alarm_snoozes", "blue_light_filter_active", "next_day_fatigue_score"]

    for col in numeric_columns[:-1]:
        plt.figure(figsize=(12, 12))
        plt.title(f"Score againt {col} scatterplot")
        sns.scatterplot(x="next_day_fatigue_score", y=col, data=bssd_df)
        plt.xlabel(f"{col}")
        plt.ylabel("next_day_fatigue_score")
        plt.show()

    # the text data and the categorical also be used in one hot encoding
    bar_columns = bssd_df.drop(numeric_columns, axis=1)
    print(bar_columns.columns)
    for column in bar_columns:
        plt.figure(figsize=(12, 12))
        plt.title(f"{column} countplot")
        sns.countplot(x=column, data=bar_columns)
        plt.xlabel(f"{column}")
        plt.show()

    plt.figure(figsize=(12, 12))
    plt.title("next_day_fatigue_score histogram")
    sns.histplot(x="next_day_fatigue_score", data=bssd_df, kde=True)
    plt.xlabel("next day fatigue score")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.title("Correlation Heatmap")
    sns.heatmap(bssd_df[numeric_columns].corr(), annot=True, cmap="coolwarm",
                fmt=".4f")
    plt.show()

    x = bssd_df.drop("next_day_fatigue_score", axis=1)
    y = bssd_df["next_day_fatigue_score"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, test_size=20)

    numeric_columns = numeric_columns[:-1]
    processor = ColumnTransformer(
        transformers=[
            ("cat_data", OneHotEncoder(handle_unknown="ignore",
             sparse_output=False, drop="first"), bar_columns.columns.to_list),
            ("numeric_data", StandardScaler(), numeric_columns[:-1])
        ],
        remainder=numeric_columns[-1]
    )

    x_train_scaled = processor.fit_transform(x_train)
    x_test = processor.transform(x_test)

    models = {
        "Linear_reg": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "Decision_tree": DecisionTreeRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Random_forest": RandomForestRegressor(),
        "Adaboost": AdaBoostRegressor(),
        "Voting": VotingRegressor(),
        "Bagging": BaggingRegressor(),
        # "Xgboost": XGBRegressor()
    }

    voting_models = [
        ("Decision_tree", DecisionTreeRegressor(max_depth=10, max_leaf_nodes=20)),
        ("SVR", SVR()),
        ("KNN", KNeighborsRegressor(n_neighbors=10))
    ]

    bag_adaboost_estimators = [DecisionTreeRegressor(
        max_depth=10, max_leaf_nodes=20), SVR(), KNeighborsRegressor(n_neighbors=10)]
    number_est = np.arange(1, 100, 1)
    alphas = (10.0**np.array([0.0, 0.5, 1.0, 1.5,
              2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]))
    depth = np.arange(10, 100, 1)
    leaf_nodes = np.arange(2, 50, 1)
    neighbors = np.arange(2, 20, 1)
    # add more parameters of the models and understand XGboost and its parameters
else:
    print("File unavailable")
