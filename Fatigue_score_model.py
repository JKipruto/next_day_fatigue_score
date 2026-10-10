import os
import pandas as pd
import duckdb as db
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
        plt.title(f"Fatigue score againt {col} scatterplot")
        sns.scatterplot(x=col, y="next_day_fatigue_score", data=bssd_df)
        plt.xlabel(f"{col}")
        plt.ylabel("next_day_fatigue_score")
        plt.show()
        plt.close()

    # the text data and the categorical also be used in one hot encoding
    bar_columns = bssd_df.drop(numeric_columns, axis=1)
    print(bar_columns.columns)
    for column in bar_columns:
        plt.figure(figsize=(12, 12))
        plt.title(f"{column} countplot")
        sns.countplot(x=column, data=bar_columns)
        plt.xlabel(f"{column}")
        plt.show()
        plt.close()

    plt.figure(figsize=(12, 12))
    plt.title("next_day_fatigue_score histogram")
    sns.histplot(x="next_day_fatigue_score", data=bssd_df, kde=True)
    plt.xlabel("next day fatigue score")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.title("Correlation Heatmap")
    sns.heatmap(bssd_df[numeric_columns].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", vmin=-1, vmax=1)
    plt.show()
    plt.close()

    x = bssd_df.drop("next_day_fatigue_score", axis=1)
    y = bssd_df["next_day_fatigue_score"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, test_size=0.2)

    numeric_columns = numeric_columns[:-1]
    processor = ColumnTransformer(
        transformers=[
            ("cat_data", OneHotEncoder(handle_unknown="ignore",
             sparse_output=False, drop="first"), bar_columns.columns.to_list()),
            ("numeric_data", StandardScaler(), numeric_columns[:-1])
        ],
        remainder='passthrough'
    )

    def make_pipeline(model):
        return Pipeline([("prep", processor), ("model", model)])

    voting_models = [
        ("Decision_tree", DecisionTreeRegressor(
            max_depth=10, max_leaf_nodes=20, random_state=42)),
        ("SVR", SVR()),
        ("KNN", KNeighborsRegressor(n_neighbors=10))
    ]

    RANDOM_STATE = 42
    number_est = np.arange(50, 500, 10)
    alphas = np.logspace(-3, 2, 11)
    depth = np.arange(2, 20, 1)
    leaf_nodes = np.arange(2, 50, 1)

    ada_bases = [DecisionTreeRegressor(
        max_depth=d, random_state=42)for d in np.arange(2, 5, 1)]

    bag_base = [DecisionTreeRegressor(
        max_depth=10, max_leaf_nodes=20, random_state=RANDOM_STATE), KNeighborsRegressor(n_neighbors=10)]

    best_models = {}
    tuned_values = {}
    cv_rmse = {}
    search_config = {
        "Lasso": (Lasso(max_iter=10000), "grid", {"model__alpha": alphas}, None),
        "Ridge": (Ridge(), "grid", {"model__alpha": alphas}, None),
        "KNN": (KNeighborsRegressor(), "grid", {"model__n_neighbors": np.arange(2, 20)}, None),
        "Decision_tree": (DecisionTreeRegressor(random_state=RANDOM_STATE), "random",
                          {"model__max_depth": depth, "model__max_leaf_nodes": leaf_nodes}, 20),
        "SVR": (SVR(), "random",
                {"model__kernel": ["linear", "rbf"],
                 "model__epsilon": np.arange(0.0, 1.0, 0.05),
                 "model__C": np.logspace(-2, 2, 10)}, 20),
        "Random_forest": (RandomForestRegressor(random_state=RANDOM_STATE), "random",
                          {"model__max_depth": depth, "model__n_estimators": number_est,
                           "model__max_leaf_nodes": leaf_nodes}, 20),
        "Adaboost": (AdaBoostRegressor(random_state=RANDOM_STATE), "random",
                     {"model__estimator": ada_bases, "model__n_estimators": np.arange(10, 101, 10)}, 10),
        "Bagging": (BaggingRegressor(random_state=RANDOM_STATE), "random",
                    {"model__estimator": bag_base, "model__n_estimators": np.arange(10, 51, 10)}, 10),
    }

    for name, (model, kind, params, n_iter) in search_config.items():
        pipe = make_pipeline(model)
        if kind == "grid":
            search = GridSearchCV(
                pipe, params, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
        else:
            search = RandomizedSearchCV(pipe, params, n_iter=n_iter, random_state=RANDOM_STATE,
                                        scoring="neg_mean_squared_error", cv=5, n_jobs=-1)

        search.fit(x_train, y_train)
        best_models[name] = search.best_estimator_
        tuned_values[name] = search.best_params_
        cv_rmse[name] = np.sqrt(-search.best_score_)

    untuned = {
        "Baseline_mean": DummyRegressor(strategy="mean"),
        "Linear_reg": LinearRegression(),
        "Voting": VotingRegressor(estimators=voting_models),
    }

    for name, model in untuned.items():
        pipe = make_pipeline(model)
        scores = cross_val_score(pipe, x_train, y_train,
                                 scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
        cv_rmse[name] = np.sqrt(-scores.mean())
        best_models[name] = pipe.fit(x_train, y_train)

    print("Best parameters")
    print("========")
    for name, p in tuned_values.items():
        print(name, p)
    print("========")
    print("Cross-validation RMSE on the training set (lower is better)")

    print(pd.Series(cv_rmse, name="CV_RMSE").sort_values())
    print("========")

    metrics_list = []
    for name, model in best_models.items():
        model_pred = model.predict(x_test)
        metrics_list.append({
            "name": name,
            "MAE": mean_absolute_error(y_test, model_pred),
            "MSE": mean_squared_error(y_test, model_pred),
            "RMSE": root_mean_squared_error(y_test, model_pred),
            "R2_score": r2_score(y_test, model_pred),
        })

    metrics_df = pd.DataFrame(metrics_list).sort_values("RMSE")
    print(metrics_df)


else:
    print("File unavailable")
