"""Shared Gemini helpers for AfriMedQA channels."""

from __future__ import annotations

from typing import Any

import requests


DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_SYSTEM_PROMPT = (
    "You are AfriMedQA, a careful clinical guidance assistant for African "
    "healthcare contexts. Give practical, plain-language guidance. Start with "
    "the likely urgency level when symptoms may be serious. Include what the "
    "user should do now, what to avoid, and which warning signs require urgent "
    "care. Do not claim to diagnose. Keep responses concise unless the user "
    "asks for detail."
)
FOCUSED_MAX_OUTPUT_TOKENS = 512
DETAILED_MAX_OUTPUT_TOKENS = 900
TEMPERATURE = 0.25
TOP_P = 0.9


def build_prompt(messages: list[dict[str, str]], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
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
    max_output_tokens: int = FOCUSED_MAX_OUTPUT_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    timeout: int = 120,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload: dict[str, Any] = {
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

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:500] if response.text else str(exc)
        if response.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' was not found or is not available for this API key."
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


def build_clinical_response(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    model: str = DEFAULT_MODEL,
    depth: str = "focused",
) -> str:
    depth_instruction = (
        "Use a concise triage-first response for Telegram. Keep it readable on mobile."
        if depth == "focused"
        else "Use a fuller response with short sections and practical detail."
    )
    max_output_tokens = FOCUSED_MAX_OUTPUT_TOKENS if depth == "focused" else DETAILED_MAX_OUTPUT_TOKENS
    prompt = build_prompt(messages, f"{DEFAULT_SYSTEM_PROMPT}\n\n{depth_instruction}")
    return call_gemini(
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
    )
