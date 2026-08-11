import json
import unittest

from api.telegram import get_chat_id_and_text, split_telegram_message, webhook_registration_url
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


if __name__ == "__main__":
    unittest.main()
