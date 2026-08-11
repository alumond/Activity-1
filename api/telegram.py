"""Telegram webhook for AfriMedQA, designed for Vercel Python Functions."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gemini_client import DEFAULT_MODEL, build_clinical_response  # noqa: E402


MAX_HISTORY_MESSAGES = 8
HISTORY_TTL_SECONDS = 60 * 60 * 24 * 7
WELCOME_MESSAGE = (
    "Welcome to AfriMedQA. Send symptoms, duration, age, and location if relevant. "
    "If this is an emergency, seek urgent medical care immediately."
)
UNSUPPORTED_MESSAGE = "Please send a text message describing the health question or symptoms."
CONFIG_ERROR_MESSAGE = "The bot is not fully configured. Please contact the administrator."


def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    raw_body = handler.rfile.read(length) if length else b"{}"
    if not raw_body:
        return {}
    return json.loads(raw_body.decode("utf-8"))


def get_message(update: dict[str, Any]) -> dict[str, Any] | None:
    return update.get("message") or update.get("edited_message")


def get_chat_id_and_text(update: dict[str, Any]) -> tuple[int | None, str | None]:
    message = get_message(update)
    if not message:
        return None, None
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text")
    if chat_id is None:
        return None, None
    return chat_id, text.strip() if isinstance(text, str) else None


def telegram_api_url(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def send_telegram_message(token: str, chat_id: int, text: str) -> None:
    chunks = split_telegram_message(text)
    for chunk in chunks:
        response = requests.post(
            telegram_api_url(token, "sendMessage"),
            json={
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            },
            timeout=30,
        )
        response.raise_for_status()


def split_telegram_message(text: str, limit: int = 3900) -> list[str]:
    clean_text = text.strip() or "I could not generate a response."
    if len(clean_text) <= limit:
        return [clean_text]

    chunks = []
    remaining = clean_text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def kv_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def kv_command(kv_url: str, kv_token: str, command: list[Any]) -> Any:
    response = requests.post(kv_url.rstrip("/"), headers=kv_headers(kv_token), json=command, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload.get("result")


def history_key(chat_id: int) -> str:
    return f"telegram:chat:{chat_id}:messages"


def load_history(chat_id: int) -> list[dict[str, str]]:
    kv_url = env("KV_REST_API_URL")
    kv_token = env("KV_REST_API_TOKEN")
    if not kv_url or not kv_token:
        return []

    result = kv_command(kv_url, kv_token, ["GET", history_key(chat_id)])
    if not result:
        return []
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [
        {"role": item["role"], "content": item["content"]}
        for item in parsed
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and item.get("content")
    ][-MAX_HISTORY_MESSAGES:]


def save_history(chat_id: int, messages: list[dict[str, str]]) -> None:
    kv_url = env("KV_REST_API_URL")
    kv_token = env("KV_REST_API_TOKEN")
    if not kv_url or not kv_token:
        return

    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    kv_command(
        kv_url,
        kv_token,
        ["SET", history_key(chat_id), json.dumps(trimmed), "EX", HISTORY_TTL_SECONDS],
    )


def verify_webhook_secret(handler: BaseHTTPRequestHandler) -> bool:
    expected_secret = env("TELEGRAM_WEBHOOK_SECRET")
    if not expected_secret:
        return False
    received_secret = handler.headers.get("X-Telegram-Bot-Api-Secret-Token")
    return received_secret == expected_secret


def required_config_present() -> bool:
    required = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_WEBHOOK_SECRET",
        "GEMINI_API_KEY",
    ]
    return all(env(name) for name in required)


def handle_text_message(chat_id: int, text: str) -> str:
    if text.lower().startswith("/start"):
        return WELCOME_MESSAGE
    if text.lower().startswith("/help"):
        return (
            "Send a health question or symptoms in one message. Include age, duration, "
            "location, pregnancy status, medicines, and warning signs when relevant."
        )

    gemini_api_key = env("GEMINI_API_KEY")
    if not gemini_api_key:
        return CONFIG_ERROR_MESSAGE

    model = env("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    history = load_history(chat_id)
    messages = history + [{"role": "user", "content": text}]
    answer = build_clinical_response(
        messages=messages,
        api_key=gemini_api_key,
        model=model,
        depth="focused",
    )
    save_history(chat_id, messages + [{"role": "assistant", "content": answer}])
    return answer


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        json_response(self, 200, {"ok": True, "service": "afrimedqa-telegram"})

    def do_POST(self) -> None:
        if not verify_webhook_secret(self):
            json_response(self, 401, {"ok": False, "error": "invalid webhook secret"})
            return

        if not required_config_present():
            json_response(self, 500, {"ok": False, "error": "missing server configuration"})
            return

        try:
            update = read_json_body(self)
            chat_id, text = get_chat_id_and_text(update)
            if chat_id is None:
                json_response(self, 200, {"ok": True, "ignored": "no chat id"})
                return

            telegram_token = env("TELEGRAM_BOT_TOKEN")
            if not text:
                send_telegram_message(telegram_token, chat_id, UNSUPPORTED_MESSAGE)
                json_response(self, 200, {"ok": True, "handled": "unsupported"})
                return

            answer = handle_text_message(chat_id, text)
            send_telegram_message(telegram_token, chat_id, answer)
            json_response(self, 200, {"ok": True})
        except Exception as exc:
            json_response(self, 500, {"ok": False, "error": str(exc)})


def webhook_registration_url(vercel_url: str, bot_token: str, secret: str) -> str:
    webhook_url = f"{vercel_url.rstrip('/')}/api/telegram"
    return (
        f"https://api.telegram.org/bot{bot_token}/setWebhook"
        f"?url={quote(webhook_url, safe='')}"
        f"&secret_token={quote(secret, safe='')}"
    )
