"""Generic timm backbone + per-target attention over slot embeddings.

Each slot contributes one embedding. Two encoder families are supported and are
distinguished at build time by probing the backbone with a dummy forward:

* **Token backbones** (ViT / DINOv2): ``forward_features`` returns ``[N, T, D]``.
  The slot embedding is the CLS token concatenated with the mean of the patch
  tokens, giving ``2 * D``.
* **Spatial backbones** (CNNs such as EfficientNet / ConvNeXt / ResNet):
  ``forward_features`` returns ``[N, C, H, W]``. The slot embedding is global
  average pooling concatenated with global max pooling (``2 * C``), then a
  ``Linear`` projection down to ``attn_dim`` so the slot-attention head always
  sees a consistent dimension.

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

# Trailing modules that sit after the last stage of common timm CNNs. They are
# cheap to train and belong with the unfrozen tail.
_CNN_TAIL_ATTRS = ("conv_head", "bn2", "norm", "norm_pre", "final_conv", "head_norm")

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


def _create_backbone(name: str, *, pretrained: bool, img_size: int, in_chans: int) -> nn.Module:
    """Build a timm model, tolerating backbones that reject ``img_size``."""
    try:
        return timm.create_model(
            name, pretrained=pretrained, num_classes=0, img_size=img_size, in_chans=in_chans
        )
    except TypeError:
        # Most CNNs are fully convolutional and take no img_size argument.
        return timm.create_model(name, pretrained=pretrained, num_classes=0, in_chans=in_chans)


class KneeSlotModel(nn.Module):
    """Shared timm encoder over slots, then per-target slot attention -> logits."""

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
        self.backbone = _create_backbone(
            backbone, pretrained=pretrained, img_size=img_size, in_chans=in_chans
        )
        self.is_token_backbone, feature_dim = self._probe_backbone(img_size, in_chans)
        if self.is_token_backbone:
            # CLS token + mean patch token.
            self.slot_proj: nn.Module = nn.Identity()
            slot_dim = feature_dim * 2
        else:
            # Global avg-pool + global max-pool, projected to a fixed width so the
            # head sees the same dimension regardless of backbone width.
            self.slot_proj = nn.Linear(feature_dim * 2, attn_dim)
            slot_dim = attn_dim
        self.head = SlotAttentionHead(slot_dim, n_targets, attn_dim=attn_dim)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)
        self.freeze_backbone(unfrozen_blocks)

    def _probe_backbone(self, img_size: int, in_chans: int) -> tuple[bool, int]:
        """Run one dummy CPU forward to learn the output rank and feature width.

        ``num_features`` is used as the expected width and cross-checked against
        the real output; the runtime shape wins if they disagree. This avoids
        name heuristics entirely.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, in_chans, img_size, img_size)
                features = self.backbone.forward_features(dummy)
        finally:
            self.backbone.train(was_training)
        declared = int(getattr(self.backbone, "num_features", 0) or 0)
        if features.ndim == 3:  # [N, T, D]
            return True, int(features.shape[-1]) or declared
        if features.ndim == 4:  # [N, C, H, W]
            return False, int(features.shape[1]) or declared
        raise ValueError(
            f"Unsupported backbone feature rank {features.ndim}; expected 3 (tokens) or 4 (maps)."
        )

    def _cnn_stages(self) -> Sequence[nn.Module]:
        """Best-effort list of CNN stages (``.stages``, else ``.blocks``)."""
        for attr in ("stages", "blocks", "layers"):
            candidate = getattr(self.backbone, attr, None)
            if isinstance(candidate, (nn.Sequential, nn.ModuleList)) and len(candidate) > 0:
                return list(candidate)
        return []

    def freeze_backbone(self, unfrozen_blocks: int) -> None:
        """Freeze the encoder, then re-enable the last ``unfrozen_blocks`` units.

        * Token backbones: the last N transformer blocks plus the final norm.
        * CNN backbones: the last N *stages* — ``.stages`` for ConvNeXt/ResNet-v2
          style models, ``.blocks`` for EfficientNet (whose ``blocks`` is a
          ``Sequential`` of stages) — plus trailing head modules
          (``conv_head``/``bn2``/``norm``). If no stage container can be found,
          fall back to unfreezing the last ~40% of parameter tensors.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if unfrozen_blocks <= 0:
            return

        if self.is_token_backbone:
            blocks: Sequence[nn.Module] = list(getattr(self.backbone, "blocks", []))
            for block in blocks[max(0, len(blocks) - unfrozen_blocks) :]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            final_norm = getattr(self.backbone, "norm", None)
            if isinstance(final_norm, nn.Module):
                for parameter in final_norm.parameters():
                    parameter.requires_grad = True
            return

        stages = self._cnn_stages()
        if stages:
            for stage in stages[max(0, len(stages) - unfrozen_blocks) :]:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
            for attr in _CNN_TAIL_ATTRS:
                module = getattr(self.backbone, attr, None)
                if isinstance(module, nn.Module):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
            return

        # Unknown structure: unfreeze the trailing ~40% of parameter tensors.
        parameters = list(self.backbone.parameters())
        tail = max(1, int(round(len(parameters) * 0.4)))
        for parameter in parameters[len(parameters) - tail :]:
            parameter.requires_grad = True

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` [N, 3, H, W] in [0, 1] -> [N, slot_dim] embeddings."""
        normalized = (images - self.mean) / self.std
        features = self.backbone.forward_features(normalized)
        if not self.is_token_backbone:
            avg_pooled = features.mean(dim=(2, 3))
            max_pooled = features.amax(dim=(2, 3))
            return self.slot_proj(torch.cat([avg_pooled, max_pooled], dim=-1))
        prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        cls_token = features[:, 0] if prefix > 0 else features.mean(dim=1)
        patches = features[:, prefix:].mean(dim=1)
        return torch.cat([cls_token, patches], dim=-1)

    def head_parameters(self):
        """Trainable non-backbone parameters (slot projection + attention head)."""
        yield from self.slot_proj.parameters()
        yield from self.head.parameters()

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``images`` [B, S, 3, H, W], ``mask`` [B, S] -> logits [B, n_targets]."""
        batch, slots = images.shape[0], images.shape[1]
        flat = images.reshape(batch * slots, *images.shape[2:])
        embeddings = self.encode(flat).reshape(batch, slots, -1)
        return self.head(embeddings, mask)


def parameter_groups(model: KneeSlotModel, backbone_lr: float, head_lr: float):
    """Split trainable parameters into backbone / head groups for the optimizer."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head_parameters() if p.requires_grad]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    groups.append({"params": head_params, "lr": head_lr})
    return groups
