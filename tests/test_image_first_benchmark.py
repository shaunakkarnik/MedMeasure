"""Regression checks for the benchmark's image-first chat adapter."""
from copy import deepcopy
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from scripts.image_first_benchmark import image_first_messages, main


class ImageFirstBenchmarkTests(unittest.TestCase):
    def conversation(self):
        return [
            {'role': 'system', 'content': [{'type': 'text', 'text': 'system'}]},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'question'},
                {'type': 'image_url', 'image_url': {'url': 'image-1'}},
                {'type': 'image_url', 'image_url': {'url': 'image-2'}}]}]

    def test_single_conversation_is_reordered_without_mutation(self):
        messages = self.conversation()
        original = deepcopy(messages)
        ordered = image_first_messages(messages)
        self.assertEqual(messages, original)
        self.assertEqual(ordered[0], original[0])
        self.assertEqual([part['type'] for part in ordered[1]['content']],
                         ['image_url', 'image_url', 'text'])
        self.assertEqual(ordered[1]['content'][:2], original[1]['content'][1:])
        self.assertEqual(image_first_messages(ordered), ordered)

    def test_batches_empty_and_plain_text_are_supported(self):
        messages = self.conversation()
        ordered = image_first_messages([messages, messages])
        self.assertEqual(ordered, [image_first_messages(messages)] * 2)
        self.assertEqual(image_first_messages([]), [])
        plain = [{'role': 'user', 'content': 'question'}]
        self.assertEqual(image_first_messages(plain), plain)

    def test_entrypoint_preserves_backend_name_and_forwards_chat_arguments(self):
        backend = ModuleType('lmms_eval.models.vllm_qwen25vl')
        entry = ModuleType('lmms_eval.__main__')
        vllm = ModuleType('vllm')
        received = {}

        class LLM:
            def chat(self, messages, *args, **kwargs):
                received.update(messages=messages, args=args, kwargs=kwargs)
                return 'original-result'

        vllm.LLM = LLM

        def evaluate():
            return backend.LLM().chat([self.conversation()], 'sampling', use_tqdm=False)

        entry.cli_evaluate = evaluate
        with patch.dict(sys.modules, {
            'vllm': vllm,
            'lmms_eval.models.vllm_qwen25vl': backend,
            'lmms_eval.__main__': entry,
        }):
            self.assertEqual(main(), 'original-result')
        self.assertEqual(received['messages'][0][1]['content'][0]['type'], 'image_url')
        self.assertEqual(received['args'], ('sampling',))
        self.assertEqual(received['kwargs'], {'use_tqdm': False})


if __name__ == '__main__':
    unittest.main()
