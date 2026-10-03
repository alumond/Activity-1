"""Streamlit chatbot UI backed by the Gemini API."""

from __future__ import annotations

import os
import json
from pathlib import Path

import requests
import streamlit as st


DEFAULT_MODEL = "gemini-3.5-flash-lite"
FOCUSED_MAX_OUTPUT_TOKENS = 512
DETAILED_MAX_OUTPUT_TOKENS = 900
TEMPERATURE = 0.25
TOP_P = 0.9
DEFAULT_SYSTEM_PROMPT = (
    "You are Health for All, a careful clinical guidance assistant for African "
    "healthcare contexts. Give practical, plain-language guidance. Start with "
    "the likely urgency level when symptoms may be serious. Include what the "
    "user should do now, what to avoid, and which warning signs require urgent "
    "care. Do not claim to diagnose. Keep responses concise unless the user "
    "asks for detail."
)
EXAMPLE_PROMPTS = [
    ("Yellow eyes + dark urine", "Yellow eyes, dark urine, and body pain"),
    ("Fever + chills for 2 days", "Fever, chills, headache, and weakness for two days"),
    ("Child unable to drink", "Child vomiting repeatedly and unable to drink"),
]
MEMORY_FILE = Path("streamlit_memory.json")
MAX_MEMORY_MESSAGES = 20


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #f7faf9;
            --muted: #9eaaa7;
            --muted-strong: #c3cdca;
            --canvas: #07100e;
            --canvas-2: #0a1512;
            --panel: rgba(16, 31, 27, .76);
            --panel-solid: #10201b;
            --panel-raised: #142722;
            --emerald: #55e6a5;
            --emerald-deep: #103d2e;
            --emerald-soft: rgba(85, 230, 165, .10);
            --gold: #e7c777;
            --warning-soft: rgba(231, 199, 119, .10);
            --line: rgba(224, 255, 243, .10);
            --line-strong: rgba(224, 255, 243, .17);
            --shadow: 0 30px 80px rgba(0, 0, 0, .34);
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 72% -12%, rgba(49, 178, 124, .16), transparent 34rem),
                radial-gradient(circle at 26% 35%, rgba(41, 117, 91, .08), transparent 32rem),
                linear-gradient(145deg, var(--canvas) 0%, #091411 45%, #07100e 100%);
            color: var(--ink);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .13;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 160 160' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.18'/%3E%3C/svg%3E");
        }

        .block-container {
            max-width: 1080px;
            padding-top: 1.45rem;
            padding-bottom: 9rem;
            padding-left: clamp(1.25rem, 4vw, 3.5rem);
            padding-right: clamp(1.25rem, 4vw, 3.5rem);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 10% 0%, rgba(85, 230, 165, .09), transparent 20rem),
                linear-gradient(180deg, #0a1512 0%, #08110f 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.3rem;
        }

        [data-testid="stSidebar"] .block-container {
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }

        .sidebar-brand,
        .topbar-brand {
            display: flex;
            align-items: center;
            gap: .75rem;
        }

        .sidebar-brand {
            padding: .35rem 0 1.6rem;
        }

        .brand-mark {
            position: relative;
            display: grid;
            place-items: center;
            width: 2.35rem;
            height: 2.35rem;
            border: 1px solid rgba(101, 244, 178, .30);
            border-radius: 12px;
            background: linear-gradient(145deg, rgba(85, 230, 165, .18), rgba(85, 230, 165, .04));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 10px 28px rgba(0,0,0,.22);
            color: var(--emerald);
        }

        .brand-mark svg {
            width: 1.3rem;
            height: 1.3rem;
        }

        .brand-name {
            color: var(--ink);
            font-size: .98rem;
            font-weight: 690;
            letter-spacing: -.02em;
        }

        .brand-meta {
            color: var(--muted);
            font-size: .68rem;
            letter-spacing: .075em;
            margin-top: .08rem;
            text-transform: uppercase;
        }

        .side-label {
            color: #74827e;
            font-size: .65rem;
            font-weight: 700;
            letter-spacing: .12em;
            margin: 1.2rem 0 .55rem;
            text-transform: uppercase;
        }

        .privacy-card {
            background: rgba(255,255,255,.025);
            border: 1px solid var(--line);
            border-radius: 14px;
            color: var(--muted);
            font-size: .76rem;
            line-height: 1.55;
            margin-top: 1.3rem;
            padding: .85rem .9rem;
        }

        .privacy-card strong {
            color: var(--muted-strong);
            display: block;
            font-size: .78rem;
            margin-bottom: .15rem;
        }

        .topbar {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin: .25rem 0 1rem;
            min-height: 3rem;
        }

        .topbar .brand-mark {
            width: 2rem;
            height: 2rem;
            border-radius: 10px;
        }

        .topbar-name {
            color: var(--ink);
            font-size: .92rem;
            font-weight: 670;
            letter-spacing: -.015em;
        }

        .status-pill {
            align-items: center;
            background: rgba(255,255,255,.025);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--muted-strong);
            display: inline-flex;
            font-size: .7rem;
            font-weight: 560;
            gap: .45rem;
            padding: .42rem .7rem;
        }

        .status-dot {
            background: var(--emerald);
            border-radius: 50%;
            box-shadow: 0 0 0 4px rgba(85, 230, 165, .10), 0 0 12px rgba(85,230,165,.35);
            height: .42rem;
            width: .42rem;
        }

        .empty-hero {
            margin: clamp(.75rem, 2vh, 1.6rem) auto 1.45rem;
            max-width: 820px;
            text-align: center;
        }

        .eyebrow {
            align-items: center;
            color: var(--emerald);
            display: inline-flex;
            font-size: .68rem;
            font-weight: 720;
            gap: .5rem;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .85rem;
        }

        .eyebrow::before {
            content: "";
            background: var(--emerald);
            border-radius: 999px;
            height: 1px;
            opacity: .65;
            width: 1.6rem;
        }

        .empty-hero h1 {
            color: var(--ink);
            font-size: clamp(2.55rem, 6vw, 4.15rem);
            font-weight: 580;
            line-height: .99;
            letter-spacing: -.055em;
            margin: 0;
        }

        .empty-hero h1 span {
            background: linear-gradient(100deg, #f7faf9 20%, #9dd8bf 92%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .empty-hero p {
            color: var(--muted);
            font-size: clamp(.98rem, 2vw, 1.11rem);
            line-height: 1.72;
            margin: .95rem auto 0;
            max-width: 625px;
        }

        .trust-row {
            align-items: center;
            color: #7f8d89;
            display: flex;
            flex-wrap: wrap;
            font-size: .7rem;
            gap: .55rem 1.15rem;
            justify-content: center;
            letter-spacing: .035em;
            margin-top: 1rem;
        }

        .trust-row span {
            align-items: center;
            display: inline-flex;
            gap: .4rem;
        }

        .trust-row span::before {
            color: var(--emerald);
            content: "✓";
            font-size: .68rem;
        }

        .safety-note {
            background: var(--warning-soft);
            border: 1px solid rgba(231, 199, 119, .20);
            border-radius: 14px;
            color: #cdbf97;
            font-size: .76rem;
            line-height: 1.55;
            margin: 1.15rem auto 0;
            max-width: 720px;
            padding: .8rem 1rem;
        }

        .safety-note strong {
            color: var(--gold);
        }

        .starter-heading {
            color: #778580;
            font-size: .66rem;
            font-weight: 700;
            letter-spacing: .12em;
            margin: 0 0 .6rem;
            text-align: center;
            text-transform: uppercase;
        }

        .st-key-starter_cards {
            margin: 0 auto;
            max-width: 900px;
        }

        .st-key-starter_cards [data-testid="stButton"] > button {
            align-items: flex-start;
            background: linear-gradient(145deg, rgba(22, 42, 36, .80), rgba(13, 28, 24, .64));
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.035), 0 18px 45px rgba(0,0,0,.14);
            color: var(--muted-strong);
            font-size: .82rem;
            font-weight: 520;
            justify-content: flex-start;
            min-height: 4.65rem;
            padding: 1rem 1.05rem;
            text-align: left;
            transition: border-color .2s ease, background .2s ease, transform .2s ease, color .2s ease;
            white-space: normal;
        }

        .st-key-starter_cards [data-testid="stButton"] > button::after {
            color: var(--emerald);
            content: "↗";
            font-size: 1rem;
            margin-left: auto;
            opacity: .55;
        }

        .st-key-starter_cards [data-testid="stButton"] > button:hover {
            background: linear-gradient(145deg, rgba(27, 52, 44, .92), rgba(15, 33, 28, .76));
            border-color: rgba(85, 230, 165, .28);
            color: var(--ink);
            transform: translateY(-2px);
        }

        .conversation-rule {
            border-top: 1px solid var(--line);
            margin: .15rem 0 1.55rem;
        }

        .conversation-meta {
            color: #70807a;
            font-size: .65rem;
            font-weight: 680;
            letter-spacing: .12em;
            margin-bottom: .85rem;
            text-transform: uppercase;
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            border: 0;
            margin-bottom: 1.3rem;
            padding: .15rem 0;
            gap: .8rem;
        }

        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
            border: 1px solid var(--line-strong);
            box-shadow: 0 8px 22px rgba(0,0,0,.18);
        }

        [data-testid="stChatMessageContent"] {
            color: #dfe7e4;
            font-size: .94rem;
            line-height: 1.72;
            padding-top: .2rem;
        }

        [data-testid="stChatMessageContent"] p {
            margin-bottom: .72rem;
        }

        [data-testid="stChatMessageContent"] strong {
            color: #f4f8f6;
            font-weight: 660;
        }

        [data-testid="stChatMessageContent"] h1,
        [data-testid="stChatMessageContent"] h2,
        [data-testid="stChatMessageContent"] h3 {
            color: var(--ink);
            font-weight: 620;
            letter-spacing: -.025em;
            margin-top: 1.3rem;
        }

        [data-testid="stChatMessageContent"] li {
            margin-bottom: .38rem;
        }

        [data-testid="stChatMessageContent"] blockquote {
            background: rgba(231, 199, 119, .075);
            border-left: 2px solid var(--gold);
            border-radius: 0 10px 10px 0;
            color: #d5c99f;
            padding: .75rem .9rem;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(145deg, rgba(25, 50, 42, .88), rgba(18, 38, 32, .78));
            border: 1px solid rgba(85, 230, 165, .13);
            border-radius: 18px 18px 5px 18px;
            margin-left: auto;
            max-width: min(78%, 720px);
            padding: .65rem .85rem;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
            color: #edf5f2;
        }

        [data-testid="stBottomBlockContainer"] {
            background: linear-gradient(180deg, rgba(7,16,14,0) 0%, rgba(7,16,14,.94) 30%, #07100e 66%);
            padding-top: 2.4rem;
            padding-bottom: 1.05rem;
        }

        [data-testid="stBottomBlockContainer"] > div {
            max-width: 940px;
            margin: 0 auto;
        }

        [data-testid="stChatInput"] > div {
            background: rgba(18, 36, 30, .92);
            border: 1px solid rgba(223, 255, 243, .16);
            border-radius: 18px;
            box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,.055);
            min-height: 3.7rem;
            transition: border-color .2s ease, box-shadow .2s ease;
        }

        [data-testid="stChatInput"] > div:focus-within {
            border-color: rgba(85, 230, 165, .38);
            box-shadow: 0 26px 72px rgba(0,0,0,.40), 0 0 0 3px rgba(85,230,165,.06);
        }

        [data-testid="stChatInput"] textarea {
            color: var(--ink) !important;
            font-size: .91rem;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #7e8d88 !important;
        }

        [data-testid="stChatInputSubmitButton"] {
            background: var(--emerald) !important;
            border-radius: 11px !important;
            color: #062017 !important;
            margin-right: .38rem;
        }

        [data-testid="stSpinner"] {
            color: var(--muted);
        }

        .stButton > button,
        .stDownloadButton > button {
            background: rgba(255,255,255,.025);
            border: 1px solid var(--line);
            border-radius: 12px;
            color: var(--muted-strong);
            min-height: 2.55rem;
            transition: all .2s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: rgba(85,230,165,.07);
            border-color: rgba(85,230,165,.24);
            color: var(--ink);
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            background: rgba(255,255,255,.025);
            border: 1px solid var(--line);
            border-radius: 12px;
            gap: .2rem;
            padding: .28rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 8px;
            flex: 1;
            justify-content: center;
            min-width: 0;
            padding: .28rem .18rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label p {
            font-size: .72rem !important;
            white-space: nowrap;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: var(--muted);
            font-size: .78rem;
        }

        [data-testid="stAlert"] {
            background: rgba(115, 34, 45, .18);
            border: 1px solid rgba(246, 112, 126, .22);
            border-radius: 14px;
            color: #f4c5cb;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: .85rem;
            }
            .empty-hero {
                margin-top: 2.75rem;
            }
            .empty-hero h1 {
                font-size: clamp(2.35rem, 13vw, 3.4rem);
            }
            .trust-row {
                gap: .45rem .8rem;
            }
            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
                max-width: 92%;
            }
            .status-pill {
                padding: .38rem .58rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:
        # Local development commonly uses environment variables without a
        # .streamlit/secrets.toml file. Streamlit raises before we can fall back.
        pass
    return os.getenv(name, default)


def load_memory() -> list[dict[str, str]]:
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [
        {"role": item["role"], "content": item["content"]}
        for item in data
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ][-MAX_MEMORY_MESSAGES:]


def save_memory(messages: list[dict[str, str]]) -> None:
    trimmed = messages[-MAX_MEMORY_MESSAGES:]
    MEMORY_FILE.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")


def format_transcript(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "No conversation yet.\n"
    blocks = ["# Health for All Chat Transcript\n"]
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
    st.set_page_config(
        page_title="Health for All — Clinical Guidance",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    gemini_api_key = get_secret("GEMINI_API_KEY")
    gemini_model = get_secret("GEMINI_MODEL", DEFAULT_MODEL)

    if "messages" not in st.session_state:
        st.session_state.messages = load_memory()
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    brand_mark = (
        '<span class="brand-mark" aria-hidden="true">'
        '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M12 3.25V20.75M3.25 12H20.75" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round"/>'
        '<path d="M7.5 7.5L16.5 16.5M16.5 7.5L7.5 16.5" stroke="currentColor" '
        'stroke-width="1.15" stroke-linecap="round" opacity=".42"/>'
        "</svg></span>"
    )

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                {brand_mark}
                <div>
                    <div class="brand-name">Health for All</div>
                    <div class="brand-meta">Clinical guidance</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("＋  New conversation", use_container_width=True, key="new_conversation"):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            save_memory([])
            st.rerun()

        st.markdown('<div class="side-label">Response style</div>', unsafe_allow_html=True)
        response_depth = st.radio(
            "Response depth",
            ["Focused", "Detailed"],
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown('<div class="side-label">Conversation</div>', unsafe_allow_html=True)
        st.download_button(
            "↓  Export transcript",
            data=format_transcript(st.session_state.messages),
            file_name="health-for-all-chat-transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.markdown(
            """
            <div class="privacy-card">
                <strong>Designed for careful guidance</strong>
                Responses prioritise urgency, immediate next steps, and warning signs.
                Avoid sharing names or other identifying details.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-brand">
                {brand_mark}
                <span class="topbar-name">Health for All</span>
            </div>
            <div class="status-pill"><span class="status-dot"></span> Guidance online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <section class="empty-hero">
                <div class="eyebrow">Thoughtful care starts here</div>
                <h1>Health guidance,<br><span>made beautifully clear.</span></h1>
                <p>
                    Describe what is happening in your own words. Health for All will help
                    you understand the urgency, what to do next, and the warning signs
                    that should never be ignored.
                </p>
                <div class="trust-row">
                    <span>Triage-first</span>
                    <span>Plain language</span>
                    <span>African health contexts</span>
                </div>
                <div class="safety-note">
                    <strong>Important:</strong> Guidance is informational and cannot replace
                    a clinician, emergency service, or local treatment protocol.
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="starter-heading">Or begin with a common concern</div>', unsafe_allow_html=True)
        with st.container(key="starter_cards"):
            cols = st.columns(3, gap="medium")
            for col, (label, prompt_text) in zip(cols, EXAMPLE_PROMPTS):
                with col:
                    if st.button(label, use_container_width=True, key=f"starter_{label}"):
                        st.session_state.pending_prompt = prompt_text
                        st.rerun()
    else:
        st.markdown(
            f"""
            <div class="conversation-rule"></div>
            <div class="conversation-meta">Current conversation · {len(st.session_state.messages)} messages</div>
            """,
            unsafe_allow_html=True,
        )

    if not gemini_api_key:
        st.error(
            "Gemini is not configured yet. Add GEMINI_API_KEY to Streamlit "
            "secrets or to the local environment to start a conversation."
        )
        st.stop()

    for message in st.session_state.messages:
        avatar = "🩺" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    user_question = st.session_state.pending_prompt or st.chat_input(
        "Describe symptoms, duration, age, and location…"
    )
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
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Reviewing the details carefully…"):
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
    save_memory(st.session_state.messages)


if __name__ == "__main__":
    main()
