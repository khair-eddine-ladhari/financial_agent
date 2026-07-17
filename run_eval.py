from dotenv import load_dotenv
load_dotenv()

from langsmith import evaluate, Client
from agent import run_financial_analysis

client = Client()
dataset_name = "financial-agent-eval-v1"

def target(inputs: dict) -> dict:
    result = run_financial_analysis(question=inputs["question"])
    return {"report": result["report"], "raw_findings": result["raw_findings"]}

def logging_only(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "manual_review_needed", "score": None, "comment": reference_outputs.get("expected", "")}

evaluate(
    target,
    data=dataset_name,
    evaluators=[logging_only],
    experiment_prefix="baseline-run",
)