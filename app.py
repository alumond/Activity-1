"""Streamlit chatbot UI backed by the Gemini API."""

from __future__ import annotations

import os
import requests
import streamlit as st


DEFAULT_MODEL = "gemini-3.5-flash-lite"
FOCUSED_MAX_OUTPUT_TOKENS = 512
DETAILED_MAX_OUTPUT_TOKENS = 900
TEMPERATURE = 0.25
TOP_P = 0.9
DEFAULT_SYSTEM_PROMPT = (
    "You are AfriMedQA, a careful clinical guidance assistant for African "
    "healthcare contexts. Give practical, plain-language guidance. Start with "
    "the likely urgency level when symptoms may be serious. Include what the "
    "user should do now, what to avoid, and which warning signs require urgent "
    "care. Do not claim to diagnose. Keep responses concise unless the user "
    "asks for detail."
)
EXAMPLE_PROMPTS = [
    "Yellow eyes, dark urine, and body pain",
    "Fever, chills, headache, and weakness for two days",
    "Child vomiting repeatedly and unable to drink",
]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #0f1720;
            --panel: #151f2b;
            --panel-soft: #1b2735;
            --text: #f6f8fb;
            --muted: #a8b3c2;
            --accent: #22c55e;
            --accent-soft: rgba(34, 197, 94, .14);
            --warning: #f5c542;
            --warning-soft: rgba(245, 197, 66, .13);
            --border: rgba(255, 255, 255, .09);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, .12), transparent 30rem),
                radial-gradient(circle at 90% 10%, rgba(56, 189, 248, .08), transparent 26rem),
                linear-gradient(180deg, #0b1118 0%, #101820 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 980px;
            padding-top: 2.2rem;
            padding-bottom: 7rem;
        }

        [data-testid="stSidebar"] {
            background: #0b1118;
            border-right: 1px solid var(--border);
        }

        .hero {
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.15rem;
            margin-bottom: 1rem;
        }

        .eyebrow {
            color: var(--accent);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .45rem;
        }

        .hero h1 {
            font-size: clamp(2rem, 5vw, 3.4rem);
            line-height: 1.03;
            letter-spacing: 0;
            margin: 0;
        }

        .hero p {
            color: var(--muted);
            max-width: 700px;
            margin-top: .7rem;
            font-size: 1rem;
        }

        .safety-note {
            background: var(--warning-soft);
            border: 1px solid rgba(245, 197, 66, .34);
            border-radius: 8px;
            padding: .9rem 1rem;
            margin: 1.2rem 0 1rem;
            color: #f7edbd;
        }

        .workspace {
            background: rgba(15, 23, 32, .64);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }

        .metric-row {
            display: grid;
            gap: .75rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 1rem 0 1.25rem;
        }

        .metric {
            background: rgba(21, 31, 43, .78);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .85rem .95rem;
        }

        .metric span {
            color: var(--muted);
            display: block;
            font-size: .78rem;
            margin-bottom: .25rem;
        }

        .metric strong {
            font-size: .95rem;
        }

        .example-grid {
            display: grid;
            gap: .7rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: .75rem;
        }

        .example {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--muted);
            padding: .8rem;
            font-size: .9rem;
        }

        .section-label {
            color: var(--muted);
            font-size: .82rem;
            font-weight: 700;
            margin: .35rem 0 .55rem;
            text-transform: uppercase;
        }

        [data-testid="stChatMessage"] {
            background: rgba(21, 31, 43, .82);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .75rem;
            margin-bottom: .85rem;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: var(--panel-soft);
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid var(--border);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid var(--border);
            min-height: 2.65rem;
        }

        @media (max-width: 760px) {
            .metric-row,
            .example-grid {
                grid-template-columns: 1fr;
            }
            .block-container {
                padding-top: 1.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_secret(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


def format_transcript(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "No conversation yet.\n"
    blocks = ["# AfriMedQA Chat Transcript\n"]
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        blocks.append(f"## {role}\n\n{message['content'].strip()}\n")
    return "\n".join(blocks)


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
    apply_theme()

    gemini_api_key = get_secret("GEMINI_API_KEY")
    gemini_model = get_secret("GEMINI_MODEL", DEFAULT_MODEL)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    with st.sidebar:
        st.header("Conversation")
        response_depth = st.radio(
            "Response depth",
            ["Focused", "Detailed"],
            horizontal=True,
        )
        st.download_button(
            "Download transcript",
            data=format_transcript(st.session_state.messages),
            file_name="afrimedqa-chat-transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">AfriMedQA assistant</div>
            <h1>AfriMedQA Clinical Chatbot</h1>
            <p>
                A polished health guidance workspace for African healthcare contexts,
                built to surface urgency, next steps, and warning signs with clarity.
            </p>
        </section>
        <div class="safety-note">
            <strong>Safety note:</strong> This tool is for demonstration and decision-support
            testing only. It does not replace a clinician, emergency care, or local
            treatment guidelines.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not gemini_api_key:
        st.error(
            "Missing Gemini configuration. Add GEMINI_API_KEY to Streamlit "
            "secrets or environment variables."
        )
        st.stop()

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric"><span>Mode</span><strong>Clinical guidance</strong></div>
            <div class="metric"><span>Response</span><strong>{response_depth}</strong></div>
            <div class="metric"><span>Safety</span><strong>Urgent symptoms flagged</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown('<div class="workspace">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Start with a common scenario</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, prompt_text in zip(cols, EXAMPLE_PROMPTS):
            with col:
                if st.button(prompt_text, use_container_width=True):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.session_state.pending_prompt or st.chat_input("Describe symptoms, duration, age, and location")
    if not user_question:
        return
    st.session_state.pending_prompt = None

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    depth_instruction = (
        "Use a concise triage-first response."
        if response_depth == "Focused"
        else "Use a fuller response with short sections and practical detail."
    )
    max_output_tokens = FOCUSED_MAX_OUTPUT_TOKENS if response_depth == "Focused" else DETAILED_MAX_OUTPUT_TOKENS
    effective_system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{depth_instruction}"
    prompt = build_prompt(st.session_state.messages, effective_system_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                answer = call_gemini(
                    api_key=gemini_api_key,
                    model=gemini_model,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
            except Exception as exc:
                answer = f"Gemini request failed: {exc}"
                st.error(answer)
            else:
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
