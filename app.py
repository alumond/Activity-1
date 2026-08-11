"""Streamlit chatbot UI backed by the Gemini API."""

from __future__ import annotations

import os
import requests
import streamlit as st


DEFAULT_MODEL = "gemini-3.5-flash-lite"
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


def call_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:500] if response.text else str(exc)
        if response.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' was not found or is not available for this API key. "
                "Try setting GEMINI_MODEL to gemini-3.5-flash-lite or gemini-3.5-flash."
            ) from exc
        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {detail}") from exc
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        feedback = data.get("promptFeedback", {})
        raise RuntimeError(f"Gemini returned no candidates. Feedback: {feedback}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError("Gemini returned a response without text.")
    return "\n".join(text_parts).strip()


def main() -> None:
    st.set_page_config(page_title="AfriMedQA Chatbot", layout="centered")
    st.title("AfriMedQA Clinical Chatbot")
    st.caption("Gemini-powered interface for African healthcare Q&A.")

    st.warning(
        "This chatbot is for demonstration and decision-support testing only. "
        "It is not a substitute for clinical judgement, emergency care, or local treatment guidelines.",
    )

    gemini_api_key = get_secret("GEMINI_API_KEY")
    gemini_model = get_secret("GEMINI_MODEL", DEFAULT_MODEL)

    with st.sidebar:
        st.header("Generation")
        st.text_input("Model", value=gemini_model, disabled=True)
        max_output_tokens = st.slider("Max output tokens", min_value=64, max_value=2048, value=512, step=64)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
        system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=180)

    if not gemini_api_key:
        st.error(
            "Missing Gemini configuration. Add GEMINI_API_KEY to Streamlit "
            "secrets or environment variables."
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
                answer = call_gemini(
                    api_key=gemini_api_key,
                    model=gemini_model,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as exc:
                answer = f"Gemini request failed: {exc}"
                st.error(answer)
            else:
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
