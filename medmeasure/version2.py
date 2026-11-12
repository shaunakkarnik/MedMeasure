"""Auxiliary mask supervision for MedVision's text measurement pathway."""
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from medmeasure.version1 import restore_spatial_tokens, segmentation_loss


class VisionLoRA(nn.Module):
    """Add a trainable low-rank update to a frozen vision projection."""

    def __init__(self, projection, rank):
        super().__init__()
        self.projection = projection
        device = projection.weight.device
        self.down = nn.Linear(projection.in_features, rank, bias=False, device=device)
        self.up = nn.Linear(rank, projection.out_features, bias=False, device=device)
        nn.init.zeros_(self.up.weight)
        # Alpha equals rank, so the LoRA multiplier is one.

    def forward(self, x):
        update = self.up(self.down(x.float()))
        return self.projection(x) + update.to(x.dtype)


def add_vision_adapters(backbone, first_block, rank):
    blocks = backbone.visual.blocks
    if not 0 <= first_block < len(blocks) or rank < 1:
        raise ValueError('Invalid vision adapter configuration')
    for block in blocks[first_block:]:
        block.attn.qkv = VisionLoRA(block.attn.qkv, rank)


def sizing_inputs(processor, image, spacing, device):
    # NumPy axes are row/column, while the prompt gives width/height.
    height_mm, width_mm = spacing
    templates = json.loads((Path(__file__).resolve().parents[1] /
                            'configs/version2-prompts.json').read_text())
    question = templates['sizing'].format(width=image.width, height=image.height,
                                          width_mm=width_mm, height_mm=height_mm)
    messages = [
        {'role': 'system', 'content': templates['system']},
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors='pt')
    return {name: value.to(device) for name, value in inputs.items()}


def references(row):
    return [float(np.float16(row[axis])) for axis in ('major_mm', 'minor_mm')]


def answer_inputs(inputs, tokenizer, row):
    major, minor = references(row)
    # The prompt and completion are tokenized separately so the mask query
    # can always use the final prompt position.
    answer = tokenizer.encode(f'<answer>{major:.8f}, {minor:.8f}</answer>',
                              add_special_tokens=False)
    completion = inputs['input_ids'].new_tensor([answer + [tokenizer.eos_token_id]])
    prompt = inputs['input_ids']
    ids = torch.cat((prompt, completion), dim=1)
    attention = torch.cat((inputs['attention_mask'], torch.ones_like(completion)), dim=1)
    labels = torch.cat((torch.full_like(prompt, -100), completion), dim=1)
    return {**inputs, 'input_ids': ids, 'attention_mask': attention, 'labels': labels}


def training_losses(backbone, decoder, layers, prompt, tokenizer, row, target):
    inputs = answer_inputs(prompt, tokenizer, row)
    if decoder is None:
        return backbone(**inputs, use_cache=False).loss, None

    visual = backbone.visual
    grid = inputs['image_grid_thw']
    window_index, _ = visual.get_window_index(grid)
    captured = {}

    def capture(index):
        def hook(module, args, output):
            captured[index] = output
        return hook

    handles = [visual.blocks[index].register_forward_hook(capture(index)) for index in layers]
    try:
        output = backbone(**inputs, use_cache=False, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    maps = [restore_spatial_tokens(captured[index], grid, window_index,
                                   visual.spatial_merge_size) for index in layers]
    # Causal attention prevents this state from seeing the teacher-forced answer.
    query = output.hidden_states[-1][:, prompt['input_ids'].shape[1] - 1]
    logits = decoder(maps, query, target.shape[-2:])
    return output.loss, segmentation_loss(logits, target)


def score_answer(row, answer):
    ref = references(row)
    result = {'sample_id': row['sample_id'], 'case_id': row['case_id'],
              'answer': answer, 'reference_mm': ref, 'prediction_mm': None,
              'avgMAE': {'MAE': None, 'success': False},
              'avgMRE': {'MRE': None, 'success': False}}
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    match = re.search(rf'<answer>\s*(?:({number})\s*,\s*({number})|\(\s*({number})\s*,\s*({number})\s*\))\s*</answer>', answer)
    if match:
        prediction = [float(value) for value in match.groups() if value is not None]
        errors = [abs(value - expected) for value, expected in zip(prediction, ref)]
        mae = sum(error / 2 for error in errors)
        mre = sum(error / expected / 2 for error, expected in zip(errors, ref))
        if math.isfinite(mae) and math.isfinite(mre):
            result.update(prediction_mm=prediction, avgMAE={'MAE': mae, 'success': True},
                          avgMRE={'MRE': mre, 'success': True})
    return result
