import time
from difflib import SequenceMatcher
from textwrap import dedent

import ollama
import pandas as pd
import streamlit as st

MODEL = "llama3"

# ---------- Evaluation data ----------
eval_data = pd.DataFrame({
    "question": [
        "What is overfitting?",
        "What is RAG?",
        "Why do we use a validation set?",
        "What is an AI agent?",
    ],
    "context": [
        "Overfitting occurs when a model learns training data too closely and performs poorly on unseen data.",
        "RAG combines information retrieval with generation so an LLM can answer using external knowledge.",
        "A validation set is used to tune model choices and hyperparameters without using the final test set.",
        "An AI agent is a system that can reason about a task and use tools or actions to accomplish it.",
    ],
    "ground_truth": [
        "Overfitting happens when a model fits training data too closely and fails to generalize.",
        "RAG retrieves relevant information and provides it to an LLM to generate a grounded response.",
        "A validation set helps tune the model while keeping the test set unseen.",
        "An AI agent can reason about a task and use tools to take actions.",
    ],
})
eval_data.to_csv("evaluation_dataset.csv", index=False)


# ---------- Model----------
def local_model(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def llm_application(question, context, prompt_version="v1"):
    if prompt_version == "v1":
        prompt = dedent(f"""\
            Answer the question using the provided context.

            Context:
            {context}

            Question:
            {question}
            """)
    else:
        prompt = dedent(f"""\
            Answer the question using only the provided context.
            If the answer is not supported by the context, say you
            don't have enough information.

            Context:
            {context}

            Question:
            {question}
            """)
    return local_model(prompt)


def run_with_latency(question, context, prompt_version):
    start = time.perf_counter()
    answer = llm_application(question, context, prompt_version)
    latency = time.perf_counter() - start
    return answer, latency


def similarity_score(prediction, reference):
    return SequenceMatcher(None, prediction.lower(), reference.lower()).ratio()


# ---------- Evaluation ----------
def evaluate_prompt(prompt_version):
    results = []
    for _, row in eval_data.iterrows():
        answer, latency = run_with_latency(
            row["question"], row["context"], prompt_version
        )
        results.append({
            "prompt": prompt_version,
            "question": row["question"],
            "answer": answer,
            "quality": similarity_score(answer, row["ground_truth"]),
            "latency": latency,
        })
    return pd.DataFrame(results)


@st.cache_data(show_spinner="Running evaluation...")
def run_comparison():
    comparison = pd.concat(
        [evaluate_prompt("v1"), evaluate_prompt("v2")],
        ignore_index=True,
    )
    comparison.to_csv("evaluation_results.csv", index=False)
    return comparison


# ---------- Dashboard ----------
st.title("LLM Evaluation Dashboard")

comparison = run_comparison()
by_prompt = comparison.groupby("prompt")[["quality", "latency"]].mean()

col1, col2 = st.columns(2)
col1.metric("Average Quality", f"{comparison['quality'].mean():.2f}")
col2.metric("Average Latency", f"{comparison['latency'].mean():.2f}s")

st.subheader("Quality by Prompt")
st.bar_chart(by_prompt["quality"])

st.subheader("Latency by Prompt")
st.bar_chart(by_prompt["latency"])

st.dataframe(comparison)