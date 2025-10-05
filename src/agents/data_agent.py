# src/agents/data_agent.py
import pandas as pd
import json

class DataAgent:
    def __init__(self, data_path):
        self.data_path = data_path

    def run(self):
        df = pd.read_csv(self.data_path)

        summary = {
            "num_rows": len(df),
            "num_features": len(df.columns),
            "target_balance": df["Churn"].value_counts(normalize=True).to_dict(),
            "numeric_features": df.select_dtypes("number").columns.tolist(),
            "categorical_features": df.select_dtypes("object").columns.tolist()
        }

        with open("reports/data_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        return df, summary
