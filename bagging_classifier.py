import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# File path
PATH = "/Users/momonawada/PycharmProjects/animal_adoption_predictor/"
ADOPTION_RECORD_CSV = "all_records.csv"

def evaluate_model(model, X_test, y_test, title):
    print(f"\n ******* {title} *******")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

def main():
    pd.set_option("display.max_columns", None)

    # Load dataset
    df = pd.read_csv(PATH + ADOPTION_RECORD_CSV)

    # Convert 'Age' and 'Age_upon_Outcome' to months
    def convert_age_to_months(age_str):
        if pd.isna(age_str) or age_str == "age":
            return np.nan
        parts = age_str.split()
        if len(parts) < 2:
            return np.nan
        num, unit = int(parts[0]), parts[1]
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

    df["Age_in_Months"] = df["Age"].apply(convert_age_to_months)
    df["Age_upon_Outcome_in_Months"] = df["Age_upon_Outcome"].apply(convert_age_to_months)
    df["Age_upon_Outcome_in_Months"] = df["Age_upon_Outcome_in_Months"].fillna(12)

    # Fill missing categorical values with "Unknown"
    fill_unknown_cols = [
        "Outcome_Subtype", "gender_intake", "fixed_intake", "fixed_outcome",
        "Sex", "Outcome_Type", "Sex_upon_Outcome", "Age_upon_Outcome"
    ]
    df[fill_unknown_cols] = df[fill_unknown_cols].fillna("Unknown")

    # Adjust 'fixed_outcome' to match 'fixed_intake' if necessary
    df.loc[(df["fixed_intake"].isin(["Neutered", "Spayed"])) & (df["fixed_outcome"] == "Intact"), "fixed_outcome"] = df["fixed_intake"]
    df.loc[df["fixed_intake"] == df["fixed_outcome"], "fixed_changed"] = 0

    # Convert categorical age bucket to numerical
    df["Age_Bucket_in_Months"] = df["Age_Bucket"].str.extract(r'(\d+)').astype(float)

    # Drop unnecessary columns
    df.drop(columns=[
        "Unnamed: 0", "Animal ID", "MonthYear_intake", "Name_intake", "Found_Location",
        "Name_outcome", "MonthYear_outcome", "Age", "DateTime_intake", "DateTime_outcome",
        "Color_intake", "Age_Bucket", "gender_outcome", "DateTime_length"
    ], inplace=True, errors="ignore")

    # Map 'Outcome_Type' into three classes
    def map_outcome(outcome):
        if outcome in ["Adoption", "Rto-Adopt"]:
            return "Adopted"
        elif outcome == "Return to Owner":
            return "Return to Owner"
        else:
            return "Others"

    df["Outcome_Type_3class"] = df["Outcome_Type"].apply(map_outcome)

    # ---- Feature Selection ----
    # Separate features (X) and target variable (y)
    X = df.drop(columns=["Outcome_Type", "Outcome_Type_3class"])
    y = df["Outcome_Type_3class"]

    # Apply one-hot encoding to categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Convert to float32 for memory efficiency
    X = X.astype(np.float32)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 1: Train RandomForestClassifier to get feature importance
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train_scaled, y_train)

    # Step 2: Select top 10 features
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    top_features = feature_importances["Feature"].head(10).values
    print("\nTop 10 Selected Features:")
    print(feature_importances.head(10))

    # Step 3: Filter data to use only selected features
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    # Step 4: Train Bagging Classifier with selected features
    ensemble_model = BaggingClassifier(
        estimator=LogisticRegression(max_iter=5000),
        max_features=10,
        max_samples=0.5,
        n_estimators=10,
    ).fit(X_train_selected, y_train)

    evaluate_model(ensemble_model, X_test_selected, y_test, "Bagging Classifier with Feature Selection")

    # Train standalone Logistic Regression
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_selected, y_train)
    evaluate_model(model, X_test_selected, y_test, "Linear Regression with Feature Selection")

if __name__ == "__main__":
    main()
