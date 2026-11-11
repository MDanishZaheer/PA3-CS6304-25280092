# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that the supplied course prompts retain their three input formats."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.qualitative import load_qualitative_prompts
from models.qwen import assert_non_thinking_prompt


class QualitativePromptTest(unittest.TestCase):
    def test_course_yaml_and_json_system_message(self):
        source = Path(__file__).resolve().parents[1] / "data" / "first_plot_questions.yaml"
        prompts = load_qualitative_prompts(source)
        self.assertEqual(len(prompts), 24)
        self.assertEqual(len({item["question_id"] for item in prompts}), 8)
        self.assertEqual([sum(item["format"] == name for item in prompts)
                          for name in ("plain", "json", "template")], [8, 8, 8])

        class Recorder:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return [1, 2, 3]

        recorder = Recorder()
        json_prompt = next(item for item in prompts if item["format"] == "json")
        assert_non_thinking_prompt(recorder, json_prompt["prompt"], json_prompt["system"])
        self.assertEqual([item["role"] for item in recorder.messages], ["system", "user"])
        self.assertEqual(recorder.messages[0]["content"], json_prompt["system"])
        self.assertFalse(recorder.kwargs["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
