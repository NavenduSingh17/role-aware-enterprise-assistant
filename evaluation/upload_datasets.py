# python -m evaluation.upload_datasets (from the root directory)

import csv
import ast
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
client = Client()
BASE_DIR = Path(__file__).parent

def upload_rag_dataset(csv_path: Path = BASE_DIR / "rag_eval_dataset.csv", dataset_name: str = "FINSOLVE-RAG-EVALUATION"):
    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)
    dataset = client.create_dataset(dataset_name=dataset_name, description="RAG Triad evaluation")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            client.create_example(
                inputs={
                    "query": row["query"],
                    "user_role": row["user_role"],
                },
                outputs={
                    "expected_answer": row["expected_answer"],
                },
                dataset_id=dataset.id,
            )
    print(f"Uploaded '{dataset_name}' successfully.")

def upload_agent_dataset(csv_path: Path = BASE_DIR / "agent_eval_dataset.csv", dataset_name: str = "FINSOLVE-AGENT-EVALUATION"):
    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)
    dataset = client.create_dataset(dataset_name=dataset_name, description="End-to-end Agent evaluation")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            client.create_example(
                inputs={
                    "user_question": row["user_question"],
                    "user_role": row["user_role"],
                },
                outputs={
                    "expected_answer": row["expected_answer"],
                    "expected_tools": ast.literal_eval(row["expected_tools"]),
                },
                dataset_id=dataset.id,
            )
    print(f"Uploaded '{dataset_name}' successfully.")

if __name__ == "__main__":
    upload_rag_dataset()
    upload_agent_dataset()
