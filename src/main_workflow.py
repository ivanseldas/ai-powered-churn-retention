# src/main_workflow.py
from agents.data_agent import DataAgent
from agents.model_agent import ModelAgent
from agents.explain_agent import ExplainabilityAgent
from agents.report_agent import ReportAgent

def main():
    print("🚀 Running Agentic Churn Workflow...")

    data_agent = DataAgent("data/telecom_churn.csv")
    df, summary = data_agent.run()

    model_agent = ModelAgent()
    model, X_test, metrics = model_agent.run(df)

    explain_agent = ExplainabilityAgent(model)
    drivers = explain_agent.run(X_test)

    report_agent = ReportAgent(api_key="YOUR_OPENAI_API_KEY")
    report_agent.run(summary, metrics, drivers)

    print("✅ All agents completed successfully!")

if __name__ == "__main__":
    main()
