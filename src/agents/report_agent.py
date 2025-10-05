# src/agents/report_agent.py
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class ReportAgent:
    def __init__(self, api_key):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def run(self, summary, metrics, drivers):
        prompt = f"""
        You are a senior data scientist.
        Write a clear business report about customer churn analysis.
        Data Summary: {summary}
        Model Metrics: {metrics}
        Top Drivers: {drivers}
        Explain how these insights can be used for customer retention.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        report = response.choices[0].message.content

        with open("reports/churn_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        return report
