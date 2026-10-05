import os
import pandas as pd
import duckdb as db
import numpy as np
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
    morning_alarm_snoozes,next_day_fatigue_score,sleep_debt_category from 'bedtime_screentime_sleep_debt.csv'""").df()
    non_binary_sex = conn.sql(
        """select * from'bedtime_screentime_sleep_debt.csv' where gender not in ('Male','Female')""").df()

    print(bssd_df.isnull().sum())
    print(non_binary_sex.count())
    print(bssd_df.isna().sum())
    print(bssd_df.dtypes)

    numeric_columns = ["age", "bedtime_phone_minutes", "screen_brightness_pct", "blue_light_filter_active",
                       "caffeine_post_5pm_mg", "physical_activity_min", "sleep_latency_min", "total_sleep_hours", "deep_sleep_pct",
                       "rem_sleep_pct", "morning_alarm_snoozes", "next_day_fatigue_score"]

    for col in numeric_columns[:-1]:
        plt.figure(figsize=(12, 12))
        plt.title(f"Score againt {col} scatterplot")
        sns.scatterplot(x="next_day_fatigue_score", y=col, data=bssd_df)
        plt.xlabel(f"{col}")
        plt.ylabel("next_day_fatigue_score")
        plt.show()

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

else:
    print("File unavailable")
