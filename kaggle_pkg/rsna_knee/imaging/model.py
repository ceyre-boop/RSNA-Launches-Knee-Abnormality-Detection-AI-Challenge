"""DINOv2 backbone + per-target attention over slot embeddings.

Each slot contributes one embedding (CLS token concatenated with mean-pooled
patch tokens). Twelve learnable target queries attend over the available slots,
so routing is learned rather than hardcoded from anatomical priors.

Requires the optional ``train`` dependency group.
"""

from __future__ import annotations

from typing import Sequence

import timm
import torch
import torch.nn as nn

from ..constants import TARGET_LABELS

DEFAULT_BACKBONE = "vit_small_patch14_dinov2"
DEFAULT_UNFROZEN_BLOCKS = 6

# ImageNet statistics; DINOv2 checkpoints are trained with these.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class SlotAttentionHead(nn.Module):
    """One learnable query per target, softmax-attending over present slots."""

    def __init__(self, embed_dim: int, n_targets: int, attn_dim: int = 256) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_targets, attn_dim) * 0.02)
        self.key_proj = nn.Linear(embed_dim, attn_dim)
        self.value_proj = nn.Linear(embed_dim, attn_dim)
        self.norm = nn.LayerNorm(attn_dim)
        self.classifier = nn.Parameter(torch.zeros(n_targets, attn_dim))
        self.bias = nn.Parameter(torch.zeros(n_targets))
        nn.init.normal_(self.classifier, std=0.02)
        self.scale = attn_dim**-0.5

    def forward(self, slot_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``slot_embeddings`` is [B, S, D]; ``mask`` is [B, S] with 1 for present."""
        keys = self.key_proj(slot_embeddings)
        values = self.value_proj(slot_embeddings)
        logits = torch.einsum("td,bsd->bts", self.queries, keys) * self.scale
        blocked = mask.unsqueeze(1) < 0.5
        logits = logits.masked_fill(blocked, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        # A study with zero usable slots would otherwise produce NaNs.
        weights = torch.nan_to_num(weights) * mask.unsqueeze(1)
        pooled = self.norm(torch.einsum("bts,bsd->btd", weights, values))
        return (pooled * self.classifier).sum(dim=-1) + self.bias


class KneeSlotModel(nn.Module):
    """Shared DINOv2 encoder over slots, then per-target slot attention -> logits."""

    def __init__(
        self,
        *,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        img_size: int = 224,
        n_targets: int = len(TARGET_LABELS),
        unfrozen_blocks: int = DEFAULT_UNFROZEN_BLOCKS,
        attn_dim: int = 256,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            in_chans=in_chans,
        )
        embed_dim = int(getattr(self.backbone, "embed_dim", self.backbone.num_features))
        self.head = SlotAttentionHead(embed_dim * 2, n_targets, attn_dim=attn_dim)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)
        self.freeze_backbone(unfrozen_blocks)

    def freeze_backbone(self, unfrozen_blocks: int) -> None:
        """Train only the final ``unfrozen_blocks`` transformer blocks (plus norm)."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        blocks: Sequence[nn.Module] = getattr(self.backbone, "blocks", [])
        for block in list(blocks)[len(blocks) - unfrozen_blocks :] if unfrozen_blocks > 0 else []:
            for parameter in block.parameters():
                parameter.requires_grad = True
        final_norm = getattr(self.backbone, "norm", None)
        if unfrozen_blocks > 0 and isinstance(final_norm, nn.Module):
            for parameter in final_norm.parameters():
                parameter.requires_grad = True

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` [N, 3, H, W] in [0, 1] -> [N, 2 * embed_dim] embeddings."""
        normalized = (images - self.mean) / self.std
        tokens = self.backbone.forward_features(normalized)
        if tokens.ndim == 4:  # convnet-style fallback: [N, C, H, W]
            pooled = tokens.mean(dim=(2, 3))
            return torch.cat([pooled, pooled], dim=-1)
        prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        cls_token = tokens[:, 0] if prefix > 0 else tokens.mean(dim=1)
        patches = tokens[:, prefix:].mean(dim=1)
        return torch.cat([cls_token, patches], dim=-1)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``images`` [B, S, 3, H, W], ``mask`` [B, S] -> logits [B, n_targets]."""
        batch, slots = images.shape[0], images.shape[1]
        flat = images.reshape(batch * slots, *images.shape[2:])
        embeddings = self.encode(flat).reshape(batch, slots, -1)
        return self.head(embeddings, mask)


def parameter_groups(model: KneeSlotModel, backbone_lr: float, head_lr: float):
    """Split trainable parameters into backbone / head groups for the optimizer."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    groups.append({"params": head_params, "lr": head_lr})
    return groups
