import os
import unittest
from unittest.mock import patch

from api.telegram import (
    format_for_telegram,
    get_chat_id_and_text,
    required_config_present,
    split_telegram_message,
    webhook_registration_url,
)
from gemini_client import build_prompt


class TelegramBotTests(unittest.TestCase):
    def test_extracts_chat_id_and_text(self):
        update = {
            "message": {
                "chat": {"id": 12345},
                "text": " yellow eyes and body pain ",
            }
        }
        self.assertEqual(get_chat_id_and_text(update), (12345, "yellow eyes and body pain"))

    def test_ignores_non_text_message(self):
        update = {"message": {"chat": {"id": 12345}, "photo": []}}
        self.assertEqual(get_chat_id_and_text(update), (12345, None))

    def test_splits_long_telegram_messages(self):
        chunks = split_telegram_message("x" * 8000, limit=3900)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 3900 for chunk in chunks))

    def test_prompt_contains_context(self):
        prompt = build_prompt(
            [
                {"role": "user", "content": "I have fever"},
                {"role": "assistant", "content": "How long has it lasted?"},
            ],
            "System instruction",
        )
        self.assertIn("System: System instruction", prompt)
        self.assertIn("User: I have fever", prompt)
        self.assertIn("Assistant: How long has it lasted?", prompt)

    def test_webhook_registration_url_encodes_values(self):
        url = webhook_registration_url("https://example.vercel.app", "123:abc", "secret value")
        self.assertIn("https://api.telegram.org/bot123:abc/setWebhook", url)
        self.assertIn("url=https%3A%2F%2Fexample.vercel.app%2Fapi%2Ftelegram", url)
        self.assertIn("secret_token=secret%20value", url)

    def test_required_config_does_not_require_kv(self):
        fake_env = {
            "TELEGRAM_BOT_TOKEN": "telegram-token",
            "TELEGRAM_WEBHOOK_SECRET": "webhook-secret",
            "GEMINI_API_KEY": "gemini-key",
        }
        with patch.dict(os.environ, fake_env, clear=True):
            self.assertTrue(required_config_present())

    def test_formats_common_markdown_for_telegram_html(self):
        formatted = format_for_telegram("**Urgency Level:** urgent\n* Go to a clinic\n- Avoid self-medication")
        self.assertIn("<b>Urgency Level:</b> urgent", formatted)
        self.assertIn("• Go to a clinic", formatted)
        self.assertIn("• Avoid self-medication", formatted)
        self.assertNotIn("**", formatted)

    def test_escapes_telegram_html(self):
        formatted = format_for_telegram("Use <test> & check **now**")
        self.assertEqual(formatted, "Use &lt;test&gt; &amp; check <b>now</b>")


if __name__ == "__main__":
    unittest.main()
