import pandas as pd
import numpy as np
import os


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop ID column if it exists
    df.drop(columns="customerID", inplace=True, errors="ignore")

    # Handle missing TotalCharges
    df.replace(" ", np.nan, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
    ).fillna(0)

    # Normalize "No internet/phone service" BEFORE mapping binary columns
    for col in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        if col in df.columns:
            df.loc[df[col] == "No internet service", col] = "No"

    if "MultipleLines" in df.columns:
        df.loc[df["MultipleLines"] == "No phone service", "MultipleLines"] = "No"

    # Binary columns
    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "Churn",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # Gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    # One-hot encode categorical variables (drop_first avoids multicollinearity)
    df = pd.get_dummies(
        df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True
    )

    # Convert booleans (if any) to integers
    for column in df.select_dtypes(bool).columns:
        df[column] = df[column].astype(int)


    # Final cleanup: ensure all numeric dtypes (no NaN in binary cols)
    df = df.apply(lambda col: pd.to_numeric(col, errors="ignore"))

    float_cols = ["MonthlyCharges", "TotalCharges",'tenure']
    for col in float_cols:
        col_mean = df[col].mean()
        col_std=df[col].std()
        df[col] = (df[col] - col_mean)/col_std

    return df


if __name__ == "__main__":
    # Load raw dataset
    raw_file_path = os.path.join(os.getcwd() ,"data", "dataset.csv")
    df = pd.read_csv(raw_file_path)

    # Preprocess
    df_processed = preprocess_data(df)

    # Save processed dataset
    processed_file_path = os.path.join(
        os.getcwd(), "data", "processed_dataset.csv"
    )

    df_processed.to_csv(processed_file_path, index=False)

    print(f"Processed dataset saved to {processed_file_path}")
