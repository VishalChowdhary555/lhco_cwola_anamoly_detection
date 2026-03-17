import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_cwola_dataframe(df_sr, df_sb):
    return (
        pd.concat([df_sr, df_sb], axis=0)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )


def split_dataset(X, y_cwola, y_truth, mjj_vals, random_state=42):
    X_train, X_temp, y_train, y_temp, truth_train, truth_temp, mjj_train, mjj_temp = train_test_split(
        X, y_cwola, y_truth, mjj_vals,
        test_size=0.30,
        random_state=random_state,
        stratify=y_cwola
    )

    X_val, X_test, y_val, y_test, truth_val, truth_test, mjj_val, mjj_test = train_test_split(
        X_temp, y_temp, truth_temp, mjj_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        truth_train, truth_val, truth_test,
        mjj_train, mjj_val, mjj_test
    )


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled
