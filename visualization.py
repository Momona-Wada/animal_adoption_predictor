import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH = "/Users/momonawada/PycharmProjects/animal_adoption_predictor/"
ADOPTION_RECORD_CSV = "all_records.csv"

pd.set_option("display.max_columns", None)

def convert_age_to_months(age_str):
    if pd.isna(age_str) or age_str == "age":
        return np.nan

    parts = age_str.split()
    if len(parts) < 2:
        return np.nan

    num, unit = parts[0], parts[1] # "3 years" → num = "3", unit = "years"

    num = int(num)

    if "year" in unit:
        return num * 12
    elif "month" in unit:
        return num
    elif "week" in unit:
        return num // 4
    elif "day" in unit:
        return num // 30
    else:
        return np.nan

def plot_boxplots(df):
    num_features = ["Age_in_Months", "Days_length"]

    for feature in num_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="Outcome_Type", y=feature)
        plt.title(f"{feature} by Outcome_Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    df = pd.read_csv(PATH + ADOPTION_RECORD_CSV)

    # ---- convert `Age` and `Age_upon_Outcome` to Month age -------------
    df["Age_in_Months"] = df["Age"].apply(convert_age_to_months)
    # print(df[["Age", "Age_in_Months"]])

    df["Age_upon_Outcome_in_Months"] = df["Age_upon_Outcome"].apply(convert_age_to_months)
    df["Age_upon_Outcome_in_Months"] = df["Age_upon_Outcome_in_Months"].fillna(12) # impute NaN with 12 (= 1-year-old)
    # print(df[["Age_upon_Outcome", "Age_upon_Outcome_in_Months"]])

    # ---- fill `null` with `Unknown` -----------
    df["Outcome_Subtype"] = df["Outcome_Subtype"].fillna("Unknown")
    df["gender_intake"] = df["gender_intake"].fillna("Unknown")
    df["gender_outcome"] = df["gender_outcome"].fillna("Unknown")
    df["fixed_intake"] = df["fixed_intake"].fillna("Unknown")
    df["fixed_outcome"] = df["fixed_outcome"].fillna("Unknown")
    df["Sex"] = df["Sex"].fillna("Unknown")
    df["Outcome_Type"] = df["Outcome_Type"].fillna("Unknown")
    df["Sex_upon_Outcome"] = df["Sex_upon_Outcome"].fillna("Unknown")
    df["Age_upon_Outcome"] = df["Age_upon_Outcome"].fillna("12 months")
    df["Name_intake"] = df["Name_intake"].fillna("Unknown")
    df["Name_outcome"] = df["Name_outcome"].fillna("Unknown")

    # --- adjust fixed_outcome to match fixed_intake ---
    df.loc[(df["fixed_intake"].isin(["Neutered", "Spayed"])) & (df["fixed_outcome"] == "Intact"), "fixed_outcome"] = df["fixed_intake"]
    df.loc[df["fixed_intake"] == df["fixed_outcome"], "fixed_changed"] = 0

    # --- convert Age_bucket to months ---
    bucket_mapping = {
        "1-3 years": "12-36 months",
        "1-6 months": "1-6 months",
        "1-6 weeks": "0.25-1.5 months",
        "4-6 years": "48-72 months",
        "7+ years": "84+ months",
        "7-12 months": "7-12 months",
        "Less than 1 week": "Less than 0.25 months"
    }
    df["Age_Bucket_in_Months"] = df["Age_Bucket"].replace(bucket_mapping)
    # print(df[["Age_Bucket", "Age_Bucket_in_Months"]])

    # Identify rows where DateTime_length is negative
    mask = df["DateTime_length"].astype(str).str.startswith("-")
    # Swap intake and outcome dates
    df.loc[mask, ["DateTime_intake", "DateTime_outcome"]] = df.loc[mask, ["DateTime_outcome", "DateTime_intake"]].values
    # Recalculate DateTime_length
    df["DateTime_intake"] = pd.to_datetime(df["DateTime_intake"])
    df["DateTime_outcome"] = pd.to_datetime(df["DateTime_outcome"])
    df["DateTime_length"] = df["DateTime_outcome"] - df["DateTime_intake"]
    # print(df[["DateTime_intake", "DateTime_outcome", "DateTime_length"]])

    # Recalculate Days_length
    df["Days_length"] = df["DateTime_length"].dt.days

    df.drop(columns=["Unnamed: 0"], inplace=True)

    # print(df.head())
    # print(df.info())
    # print(df.isnull().sum())
    # print(df["Outcome_Type"].value_counts())
    # print(df.describe())

    plot_boxplots(df)


if __name__ == "__main__":
    main()

