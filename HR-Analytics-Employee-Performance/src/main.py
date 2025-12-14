from pathlib import Path
from hr_analytics import (
    load_hr_data,
    attrition_by_department,
    satisfaction_vs_performance,
    salary_distribution,
    attrition_prediction
)


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "ibm_hr_dataset.xlsx"
    reports = root / "reports"
    reports.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = load_hr_data(data_path)
    print(df.head())

    print("\nGenerating visual analytics...")
    print("Saved:", attrition_by_department(df, reports))
    print("Saved:", satisfaction_vs_performance(df, reports))
    print("Saved:", salary_distribution(df, reports))

    print("\nRunning attrition prediction model...")
    print("Saved:", attrition_prediction(df, reports))

    print("\nDashboard generation complete!")


if __name__ == "__main__":
    main()
