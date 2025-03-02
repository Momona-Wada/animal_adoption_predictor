import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

def map_outcome(outcome):
    if outcome in ["Adoption", "Rto-Adopt"]:
        return "Adopted"
    elif outcome == "Return to Owner":
        return "Return to Owner"
    else:
        return "Others"

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("\n*** Random Forest Model Evaluation ***")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Most Important Features in Random Forest:")
    print(feature_importance.head(10))

    return rf_model

def main():
    df = pd.read_csv(PATH + ADOPTION_RECORD_CSV)

    # ---- convert `Age` and `Age_upon_Outcome` to Month age -------------
    df["Age_in_Months"] = df["Age"].apply(convert_age_to_months)
    # print(df[["Age", "Age_in_Months"]])

    df["Age_upon_Outcome_in_Months"] = df["Age_upon_Outcome"].apply(convert_age_to_months)
    df["Age_upon_Outcome_in_Months"] = df["Age_upon_Outcome_in_Months"].fillna(12) # impute NaN with 12 (= 1-year-old)
    # print(df[["Age_upon_Outcome", "Age_upon_Outcome_in_Months"]])

    # Fill missing categorical values with "Unknown"
    fill_unknown_cols = [
        "Outcome_Subtype", "gender_intake", "fixed_intake", "fixed_outcome",
        "Sex", "Outcome_Type", "Sex_upon_Outcome", "Age_upon_Outcome"
    ]
    df[fill_unknown_cols] = df[fill_unknown_cols].fillna("Unknown")

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
    # Convert categorical age bucket to numerical
    df["Age_Bucket_in_Months"] = df["Age_Bucket_in_Months"].str.extract(r'(\d+)').astype(float)

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

    df.drop(columns=["Unnamed: 0", "Animal ID", "MonthYear_intake" ,"Name_intake", "Found_Location", "Name_outcome",
                     "MonthYear_outcome", "Age", "DateTime_intake", "DateTime_outcome", "Color_intake", "Age_Bucket",
                     "gender_outcome", "DateTime_length"], inplace=True)

    # ---- Map Outcome_Type to 3 classes: Adopted, Return to Owner, Others ----
    df["Outcome_Type_3class"] = df["Outcome_Type"].apply(map_outcome)

    # print(df.head())
    # print(df.info())
    # print(df.isnull().sum())
    # print(df["Outcome_Type"].value_counts())
    # print(df.describe())

    # Prepare dataset for model training
    X = df.drop(columns=["Outcome_Type", "Outcome_Type_3class"])
    y = df["Outcome_Type_3class"]

    # Convert target variable to numerical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test, X.columns)


if __name__ == "__main__":
    main()

