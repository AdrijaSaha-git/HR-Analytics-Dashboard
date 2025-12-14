import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ---------------- LOAD & CLEAN DATA ---------------- #

def load_hr_data(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    
    # Basic cleaning
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df.dropna(inplace=True)
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df


# ---------------- KPIs & ANALYSIS ---------------- #

def attrition_by_department(df: pd.DataFrame, reports: Path):
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="Department", hue="Attrition")
    plt.title("Attrition Count by Department")
    plt.xticks(rotation=45)
    out_path = reports / "attrition_by_dept.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def satisfaction_vs_performance(df: pd.DataFrame, reports: Path):
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df, x="JobSatisfaction", y="PerformanceRating", hue="Attrition")
    plt.title("Satisfaction vs Performance")
    out_path = reports / "satisfaction_vs_performance.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def salary_distribution(df: pd.DataFrame, reports: Path):
    plt.figure(figsize=(6,5))
    sns.histplot(df["MonthlyIncome"], bins=30, kde=True)
    plt.title("Salary Distribution")
    out_path = reports / "salary_distribution.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------- BASIC ML MODEL ---------------- #

def attrition_prediction(df: pd.DataFrame, reports: Path):
    # Prepare dataset
    features = ["Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany"]
    X = df[features]
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Save results
    result_path = reports / "attrition_prediction_results.txt"
    with open(result_path, "w") as f:
        f.write(f"Model Accuracy: {acc:.2f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, preds))

    return result_path
