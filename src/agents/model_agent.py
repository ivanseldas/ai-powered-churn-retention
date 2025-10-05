# src/agents/model_agent.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import json

class ModelAgent:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )

    def run(self, df):
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        return self.model, X_test, metrics
