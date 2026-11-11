# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that Qwen's template-only final newline is not a training target."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare import format_training_example


class TemplateTokenizer:
    def __init__(self, trailing="\n"):
        self.trailing = trailing

    def apply_chat_template(self, messages, **kwargs):
        return [10, 11, 12] if len(messages) == 1 else [10, 11, 12, 42, 99, 13]

    def convert_tokens_to_ids(self, token):
        return 99

    def decode(self, token_ids, **kwargs):
        return self.trailing if token_ids else ""


class TrainingFormatTest(unittest.TestCase):
    def test_template_newline_is_removed_before_length_check(self):
        encoded = format_training_example(TemplateTokenizer(), "Question", "Answer", 5)
        self.assertEqual(encoded["input_ids"], [10, 11, 12, 42, 99])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 42, 99])
        self.assertEqual(encoded["valid_token_count"], 2)

    def test_nonwhitespace_after_end_turn_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected content"):
            format_training_example(TemplateTokenizer("extra"), "Question", "Answer", 10)


if __name__ == "__main__":
    unittest.main()
