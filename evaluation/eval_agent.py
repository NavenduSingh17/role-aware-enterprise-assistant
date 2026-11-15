# python -m evaluation.eval_agent (from the root directory)

import time
from uuid import uuid4
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from langchain_core.prompts import ChatPromptTemplate

from chat_agent import chat

load_dotenv()
client = Client()

# I'm just giving the application (not judge) llm model (which i'm using for all) as prefix
#EXPERIMENT_TAG = "qwen/qwen3.8-27b".replace("/", "-").replace(":", "-")
EXPERIMENT_TAG = "openai/gpt-oss-120b".replace("/", "-").replace(":", "-")

#JUDGE_MODEL_NAME = "qwen/qwen3.8-27b"
JUDGE_MODEL_NAME = "openai/gpt-oss-120b"
JUDGE_LLM = ChatGroq(model=JUDGE_MODEL_NAME, temperature=0)

class JudgeGrade(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
    comment: str = Field(description="Clear explanation of why this score was given")

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system", 
            'Respond ONLY with a valid JSON object matching this exact schema: {{"score": <float between 0.0 and 1.0>, "comment": "<string>"}}, do NOT use "reasoning" or "explanation", you have to use "comment"'
    ),
    ("human", "{input}")
])

structured_judge = (
    (lambda text: {"input": text})
    | JUDGE_PROMPT
    | JUDGE_LLM.with_structured_output(JudgeGrade, method="json_mode")
)

def agent_target(inputs: dict) -> dict:
    answer = chat(
        user_question=inputs["user_question"],
        user_role=inputs["user_role"],
        session_thread_id=str(uuid4())
    )
    return {"answer": answer}

def rate_limited_examples(dataset_name: str, delay_seconds: int = 60):
    examples = list(client.list_examples(dataset_name=dataset_name))
    for idx, example in enumerate(examples):
        if idx > 0:
            print(f"[Rate Limit] Sleeping {delay_seconds}s before test {idx + 1}/{len(examples)}...")
            time.sleep(delay_seconds)
        yield example

def extract_tool_runs(run: Run) -> list[Run]:
    tool_runs = []
    def _walk(r: Run):
        if r.run_type == "tool":
            tool_runs.append(r)
        if r.child_runs:
            for child in r.child_runs:
                _walk(child)
    _walk(run)
    return tool_runs

def is_tool_failed(tool_run: Run) -> bool:
    out_str = str(tool_run.outputs or "")
    return bool(
        tool_run.error
        or "Database query failed:" in out_str
        or "Invalid query generated:" in out_str
    )

def eval_answer_correctness(run: Run, example: Example) -> dict:
    inputs = example.inputs or {}
    ref = example.outputs or {}
    pred = (run.outputs or {}).get("answer", "")

    prompt = f"""Grade the assistant's response against the reference answer.
User Role: {inputs.get('user_role')}
User Question: {inputs.get('user_question')}
Reference Answer: {ref.get('expected_answer')}
Actual Assistant Response: {pred}"""
    grade = structured_judge.invoke(prompt)
    return {"key": "answer_correctness", "score": grade.score, "comment": grade.comment}

def eval_latency(run: Run, example: Example) -> dict:
    if run.end_time and run.start_time:
        latency_sec = round((run.end_time - run.start_time).total_seconds(), 3)
    else:
        latency_sec = 0.0

    return {
        "key": "latency_seconds",
        "score": latency_sec,
        "comment": f"Execution took {latency_sec}s (excluding rate-limit sleep)."
    }

def summary_metrics(runs: list[Run], examples: list[Example]) -> list[dict]:
    total_tasks = len(runs)
    successful_tasks = 0
    total_tool_calls = 0
    failed_tool_calls = 0

    for run, example in zip(runs, examples):
        tool_runs = extract_tool_runs(run)
        called_tools = {t.name for t in tool_runs}
        expected_tools = set((example.outputs or {}).get("expected_tools", []))
        row_tool_failed = False

        for t in tool_runs:
            total_tool_calls += 1
            if is_tool_failed(t):
                failed_tool_calls += 1
                row_tool_failed = True

        corr = eval_answer_correctness(run, example)["score"]
        if corr >= 0.8 and called_tools == expected_tools and not row_tool_failed:
            successful_tasks += 1

    tool_failure_rate = (failed_tool_calls / total_tool_calls) if total_tool_calls > 0 else 0.0
    task_success_rate = (successful_tasks / total_tasks) if total_tasks > 0 else 0.0

    return {
        "results": [
            {
                "key": "overall_tool_failure_rate",
                "score": round(tool_failure_rate, 4),
                "comment": f"{failed_tool_calls} failed tool calls out of {total_tool_calls} total tool invocations."
            },
            {
                "key": "overall_task_success_rate",
                "score": round(task_success_rate, 4),
                "comment": f"{successful_tasks}/{total_tasks} tasks called expected tools, passed correctness (>=0.8), and had zero tool failures."
            }
        ]
    }

if __name__ == "__main__":
    evaluate(
        agent_target,
        data=rate_limited_examples("FINSOLVE-AGENT-EVALUATION", delay_seconds=15),
        evaluators=[eval_answer_correctness,eval_latency],
        summary_evaluators=[summary_metrics],
        experiment_prefix=f"finsolve-agent-judge-{EXPERIMENT_TAG}",
        metadata={"judge_model": JUDGE_MODEL_NAME},
        max_concurrency=1,
    )
