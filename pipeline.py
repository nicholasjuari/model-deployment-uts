import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Classification models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv("B.csv")
    df = df.drop(columns=["student_id"])
    return df

# =========================
# BUILD PREPROCESSOR
# =========================
def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocessor

# =========================
# CLASSIFICATION PIPELINE
# =========================
def train_classification(df):
    X_cls = df.drop(columns=["placement_status", "salary_package_lpa"])
    y_cls = df["placement_status"]

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    preprocessor_cls = build_preprocessor(X_cls)

    models_cls = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        #"Random Forest": RandomForestClassifier(random_state=42),
        #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        #"Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }

    results_cls = []

    for name, model in models_cls.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor_cls),
            ("model", model)
        ])

        pipe.fit(X_train_cls, y_train_cls)
        y_pred = pipe.predict(X_test_cls)

        acc = accuracy_score(y_test_cls, y_pred)
        precision = precision_score(y_test_cls, y_pred, zero_division=0)
        recall = recall_score(y_test_cls, y_pred, zero_division=0)
        f1 = f1_score(y_test_cls, y_pred, zero_division=0)

        results_cls.append([name, acc, precision, recall, f1])

        mlflow.log_metric(f"{name}_cls_accuracy", acc)
        mlflow.log_metric(f"{name}_cls_precision", precision)
        mlflow.log_metric(f"{name}_cls_recall", recall)
        mlflow.log_metric(f"{name}_cls_f1", f1)

    results_cls_df = pd.DataFrame(
        results_cls,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
    )

    best_cls_name = results_cls_df.sort_values(by="F1", ascending=False).iloc[0]["Model"]
    best_cls_model = models_cls[best_cls_name]

    final_cls_pipeline = Pipeline([
        ("preprocessor", preprocessor_cls),
        ("model", best_cls_model)
    ])

    final_cls_pipeline.fit(X_cls, y_cls)

    with open("model_classification.pkl", "wb") as f:
        pickle.dump(final_cls_pipeline, f)

    mlflow.log_param("best_classification_model", best_cls_name)
    mlflow.sklearn.log_model(final_cls_pipeline, "classification_model")

    return results_cls_df, best_cls_name

# =========================
# REGRESSION PIPELINE
# =========================
def train_regression(df):
    df_reg = df.dropna(subset=["salary_package_lpa"]).copy()
    X_reg = df_reg.drop(columns=["salary_package_lpa", "placement_status"])
    y_reg = df_reg["salary_package_lpa"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    preprocessor_reg = build_preprocessor(X_reg)

    models_reg = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        )
    }

    results_reg = []

    for name, model in models_reg.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor_reg),
            ("model", model)
        ])

        pipe.fit(X_train_reg, y_train_reg)
        y_pred = pipe.predict(X_test_reg)

        mae = mean_absolute_error(y_test_reg, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
        r2 = r2_score(y_test_reg, y_pred)

        results_reg.append([name, mae, rmse, r2])

        mlflow.log_metric(f"{name}_reg_mae", mae)
        mlflow.log_metric(f"{name}_reg_rmse", rmse)
        mlflow.log_metric(f"{name}_reg_r2", r2)

    results_reg_df = pd.DataFrame(
        results_reg,
        columns=["Model", "MAE", "RMSE", "R2"]
    )

    best_reg_name = results_reg_df.sort_values(by="R2", ascending=False).iloc[0]["Model"]
    best_reg_model = models_reg[best_reg_name]

    final_reg_pipeline = Pipeline([
        ("preprocessor", preprocessor_reg),
        ("model", best_reg_model)
    ])

    final_reg_pipeline.fit(X_reg, y_reg)

    with open("model_regression.pkl", "wb") as f:
        pickle.dump(final_reg_pipeline, f)

    mlflow.log_param("best_regression_model", best_reg_name)
    mlflow.sklearn.log_model(final_reg_pipeline, "regression_model")

    return results_reg_df, best_reg_name

# =========================
# MAIN
# =========================
def main():
    df = load_data()

    mlflow.set_experiment("Student_Placement_and_Salary")

    with mlflow.start_run():
        results_cls_df, best_cls_name = train_classification(df)
        results_reg_df, best_reg_name = train_regression(df)

        print("=== Classification Results ===")
        print(results_cls_df)
        print("Best Classification Model:", best_cls_name)

        print("\n=== Regression Results ===")
        print(results_reg_df)
        print("Best Regression Model:", best_reg_name)

if __name__ == "__main__":
    main()