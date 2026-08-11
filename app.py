"""Streamlit chatbot UI for an externally hosted AfriMedQA Llama endpoint."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert African health AI. Answer medical questions concisely, "
    "accurately, and with appropriate caution for African healthcare contexts. "
    "Encourage urgent clinical care when symptoms suggest an emergency."
)


def get_secret(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


def build_prompt(messages: list[dict[str, str]], system_prompt: str) -> str:
    prompt_parts = [f"System: {system_prompt.strip()}"]
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        prompt_parts.append(f"{role}: {message['content'].strip()}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


def call_hugging_face_endpoint(
    *,
    endpoint_url: str,
    hf_token: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False,
        },
    }

    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("generated_text", "").strip()
    if isinstance(data, dict):
        if "generated_text" in data:
            return str(data["generated_text"]).strip()
        if "error" in data:
            raise RuntimeError(str(data["error"]))
    return str(data).strip()


def main() -> None:
    st.set_page_config(page_title="AfriMedQA Chatbot", layout="centered")
    st.title("AfriMedQA Clinical Chatbot")
    st.caption("Fine-tuned Llama 3.2 interface for African healthcare Q&A.")

    st.warning(
        "This chatbot is for demonstration and decision-support testing only. "
        "It is not a substitute for clinical judgement, emergency care, or local treatment guidelines.",
    )

    endpoint_url = get_secret("HF_ENDPOINT_URL")
    hf_token = get_secret("HF_TOKEN")

    with st.sidebar:
        st.header("Generation")
        max_new_tokens = st.slider("Max new tokens", min_value=64, max_value=1024, value=384, step=64)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
        system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=180)

    if not endpoint_url or not hf_token:
        st.error(
            "Missing Hugging Face configuration. Add HF_ENDPOINT_URL and HF_TOKEN "
            "to Streamlit secrets or environment variables."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a clinical question")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    prompt = build_prompt(st.session_state.messages, system_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                answer = call_hugging_face_endpoint(
                    endpoint_url=endpoint_url,
                    hf_token=hf_token,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as exc:
                answer = f"Endpoint request failed: {exc}"
                st.error(answer)
            else:
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
