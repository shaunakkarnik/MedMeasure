"""Keep the pinned benchmark backend and place images before question text.

The upstream task formatter, vLLM engine, sampling and scoring stay in charge.
A dependency subclass changes only the user message content order at chat time.
"""
import importlib


def image_first_messages(messages):
    """Return single or batched conversations without mutating the input."""
    if not messages:
        return messages
    if isinstance(messages[0], list):
        return [image_first_messages(conversation) for conversation in messages]
    result = []
    for message in messages:
        content = message.get('content')
        if message['role'] == 'user' and isinstance(content, list):
            images = [part for part in content if part.get('type') in ('image', 'image_url', 'image_pil')]
            others = [part for part in content if part.get('type') not in ('image', 'image_url', 'image_pil')]
            message = {**message, 'content': images + others}
        result.append(message)
    return result


def main():
    from vllm import LLM

    backend = importlib.import_module('lmms_eval.models.vllm_qwen25vl')

    class ImageFirstLLM(LLM):
        def chat(self, messages, *args, **kwargs):
            return super().chat(image_first_messages(messages), *args, **kwargs)

    # Keep the registered model name: upstream image/spacing formatting uses it.
    backend.LLM = ImageFirstLLM
    evaluate = importlib.import_module('lmms_eval.__main__').cli_evaluate
    return evaluate()


if __name__ == '__main__':
    main()
