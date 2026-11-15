# python -m evaluation.eval_rag (from the root directory)

import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from langchain_core.prompts import ChatPromptTemplate

from knowledge_retrieval import search_company_docs

load_dotenv()
client = Client()

# I'm just giving the application (not judge) llm model (which i'm using for all) as prefix
#EXPERIMENT_TAG = "qwen/qwen3.8-27b".replace("/", "-").replace(":", "-")
EXPERIMENT_TAG = "openai/gpt-oss-120b".replace("/", "-").replace(":", "-")

#JUDGE_MODEL_NAME = "qwen/qwen3.8-27b"
JUDGE_MODEL_NAME = "openai/gpt-oss-120b"
JUDGE_LLM = ChatGroq(model=JUDGE_MODEL_NAME, temperature=0)

class JudgeGrade(BaseModel):
    score: float = Field(description="Score between 0.0 (completely fails) and 1.0 (completely passes)")
    comment: str = Field(description="Concise explanation justifying the score")

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

def rag_target(inputs: dict) -> dict:
    tool_msg = search_company_docs.invoke(
        {
            "name": "search_company_docs",
            "args": {"query": inputs["query"]},
            "id": "eval_call_1",
            "type": "tool_call",
        },
        config={"configurable": {"user_role": inputs["user_role"]}},
    )

    retrieved_docs = tool_msg.artifact or []

    return {
        "answer": tool_msg.content,
        "contexts": [
            f"[{d.metadata.get('source')} | Page: {d.metadata.get('page')}] {d.page_content}"
            for d in retrieved_docs
        ],
    }

def rate_limited_examples(dataset_name: str, delay_seconds: int = 60):
    examples = list(client.list_examples(dataset_name=dataset_name))
    for idx, example in enumerate(examples):
        if idx > 0:
            print(f"[Rate Limit] Sleeping {delay_seconds}s before test {idx + 1}/{len(examples)}...")
            time.sleep(delay_seconds)
        yield example

def eval_context_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = f"""Evaluate if the retrieved context chunks contain the information needed to answer the query.
If the reference answer states that the information is not available to the user's role, score 1.0 when no unauthorized documents are retrieved.
Query: {inputs['query']}
Reference Answer: {reference_outputs.get('expected_answer', '')}
Retrieved Contexts:
{chr(10).join(outputs.get('contexts', []))}"""
    grade = structured_judge.invoke(prompt)
    return {"key": "context_relevance", "score": grade.score, "comment": grade.comment}



def eval_groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    contexts = outputs.get('contexts', [])
    context_text = chr(10).join(contexts) if contexts else "None (No documents retrieved)"

    prompt = f"""Evaluate if the generated answer is strictly grounded in the retrieved context with zero hallucinations.

Rules:
1. If no documents were retrieved (Context is None) AND the generated answer is a refusal (e.g., stating it cannot help, the question is out of scope, or information is not available), score 1.0 because the model did not hallucinate any facts.
2. If documents were retrieved, score 1.0 only if every factual claim in the generated answer is supported by the retrieved context.
3. Score 0.0 if the generated answer invents facts not present in the retrieved context.

Query: {inputs['query']}
Reference Answer: {reference_outputs.get('expected_answer', '')}
Retrieved Contexts:
{context_text}
Generated Answer: {outputs.get('answer', '')}"""
    grade = structured_judge.invoke(prompt)
    return {"key": "groundedness", "score": grade.score, "comment": grade.comment}

def eval_answer_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = f"""Evaluate if the generated answer directly and accurately addresses the user query compared to the reference answer.
Query: {inputs['query']}
Reference Answer: {reference_outputs.get('expected_answer', '')}
Generated Answer: {outputs.get('answer', '')}"""
    grade = structured_judge.invoke(prompt)
    return {"key": "answer_relevance", "score": grade.score, "comment": grade.comment}

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

if __name__ == "__main__":
    evaluate(
        rag_target,
        data=rate_limited_examples("FINSOLVE-RAG-EVALUATION", delay_seconds=10),
        evaluators=[
            eval_context_relevance,
            eval_groundedness,
            eval_answer_relevance,
            eval_latency
        ],
        experiment_prefix=f"finsolve-rag-judge-{EXPERIMENT_TAG}",
        metadata={"judge_model": JUDGE_MODEL_NAME},
        max_concurrency=1,
    )
