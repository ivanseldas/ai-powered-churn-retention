# src/agents/explain_agent.py
import shap
import numpy as np
import json

class ExplainabilityAgent:
    def __init__(self, model):
        self.model = model

    def run(self, X_test):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        mean_shap = np.abs(shap_values[1]).mean(axis=0)
        top_indices = np.argsort(mean_shap)[-5:][::-1]
        top_features = X_test.columns[top_indices].tolist()
        top_values = mean_shap[top_indices].round(3).tolist()

        drivers = dict(zip(top_features, top_values))

        with open("reports/drivers.json", "w") as f:
            json.dump(drivers, f, indent=4)

        return drivers
