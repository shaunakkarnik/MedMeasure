"""Frozen Qwen2.5-VL features and a question-conditioned dense mask decoder.

The adapter targets transformers 4.51.3 explicitly. Only image/question inputs
reach the backbone; segmentation labels and physical spacing stay outside it.
"""
import torch
from torch import nn
from torch.nn import functional as F


def restore_spatial_tokens(tokens, grid_thw, window_index, merge_size):
    """Undo window packing, then undo the processor's merge-block packing."""
    if tuple(grid_thw.shape) != (1, 3):
        raise ValueError('Version 1 pilot expects exactly one image per forward')
    time, height, width = (int(x) for x in grid_thw[0])
    if time != 1 or height % merge_size or width % merge_size:
        raise ValueError('Expected a still image with merge-divisible spatial grid')
    channels = tokens.shape[-1]
    if tokens.shape != (height * width, channels):
        raise ValueError('Vision token count does not match the image grid')
    blocks = tokens.reshape(-1, merge_size * merge_size, channels)
    blocks = blocks[torch.argsort(window_index).to(tokens.device)]
    raster = blocks.reshape(height // merge_size, width // merge_size,
                            merge_size, merge_size, channels).permute(0, 2, 1, 3, 4)
    return raster.reshape(height, width, channels).permute(2, 0, 1).unsqueeze(0)


class FrozenMedVisionFeatures(nn.Module):
    def __init__(self, backbone, vision_layers):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.vision_layers = tuple(vision_layers)
        if not self.vision_layers or len(set(self.vision_layers)) != len(self.vision_layers):
            raise ValueError('Choose distinct vision block indices')
        if any(i < 0 or i >= len(backbone.visual.blocks) for i in self.vision_layers):
            raise ValueError('Vision layer index outside backbone depth')

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, inputs):
        allowed = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'}
        if set(inputs) != allowed:
            raise ValueError('Expected only processor image/question tensors; answers are forbidden')
        ids, attention = inputs['input_ids'], inputs['attention_mask']
        if ids.shape[0] != 1 or not attention.bool().any():
            raise ValueError('Expected one nonempty prompt')
        grid = inputs['image_grid_thw']
        visual = self.backbone.visual
        window_index, _ = visual.get_window_index(grid)
        captured = {}
        handles = []
        for index in self.vision_layers:
            def capture(module, args, output, index=index):
                captured[index] = output
            handles.append(visual.blocks[index].register_forward_hook(capture))
        try:
            image_features = visual(inputs['pixel_values'].to(visual.dtype), grid_thw=grid)
        finally:
            for handle in handles:
                handle.remove()
        maps = [restore_spatial_tokens(captured[i], grid, window_index,
                                       visual.spatial_merge_size) for i in self.vision_layers]
        embeddings = self.backbone.model.embed_tokens(ids)
        image_tokens = ids == self.backbone.config.image_token_id
        if int(image_tokens.sum()) != image_features.shape[0]:
            raise ValueError('Image placeholder count differs from visual features')
        embeddings = embeddings.masked_scatter(
            image_tokens.unsqueeze(-1).expand_as(embeddings), image_features.to(embeddings))
        positions, _ = self.backbone.get_rope_index(
            ids, image_grid_thw=grid, attention_mask=attention)
        # Call the text model directly: no vocabulary logits or generated answer.
        output = self.backbone.model(inputs_embeds=embeddings, attention_mask=attention,
                                     position_ids=positions, use_cache=False, return_dict=True)
        last = torch.where(attention[0].bool())[0][-1]
        query = output.last_hidden_state[:, last]
        return maps, query


class TargetMaskDecoder(nn.Module):
    def __init__(self, vision_dim, language_dim, num_layers, dim=128, heads=4):
        super().__init__()
        if dim % 8 or dim % heads:
            raise ValueError('Decoder dimension must be divisible by 8 and attention heads')
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(vision_dim, dim, 1), nn.GroupNorm(8, dim), nn.GELU())
            for _ in range(num_layers)])
        self.fusion = nn.Sequential(nn.Conv2d(dim * num_layers, dim, 1), nn.GELU(),
                                    nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.query_projection = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.mask_embedding = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pixel_embedding = nn.Conv2d(dim, dim, 3, padding=1)
        self.pixel_bias = nn.Conv2d(dim, 1, 1)
        self.scale = dim ** -.5

    def forward(self, maps, query, output_size):
        if len(maps) != len(self.projections):
            raise ValueError('Feature layer count differs from decoder projections')
        device = self.pixel_bias.weight.device
        fused = self.fusion(torch.cat([project(x.to(device=device, dtype=torch.float32))
                                      for project, x in zip(self.projections, maps)], dim=1))
        target = self.query_projection(query.to(device=device, dtype=torch.float32)).unsqueeze(1)
        pixels = fused.flatten(2).transpose(1, 2)
        attended, _ = self.attention(target, pixels, pixels, need_weights=False)
        target = self.mask_embedding(self.query_norm(target + attended)).squeeze(1)
        logits = torch.einsum('bc,bchw->bhw', target, self.pixel_embedding(fused))[:, None] * self.scale
        logits = logits + self.pixel_bias(fused)
        # Processor resizes the whole native slice, with no crop/pad/reorientation.
        return F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)


def segmentation_loss(logits, target):
    target = target.to(device=logits.device, dtype=torch.float32)
    if target.shape != logits.shape:
        raise ValueError('Target must be aligned with native-grid logits')
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = logits.sigmoid()
    dimensions = (1, 2, 3)
    dice = (2 * (probabilities * target).sum(dimensions) + 1e-6) / (
        probabilities.sum(dimensions) + target.sum(dimensions) + 1e-6)
    return bce + 1 - dice.mean()
