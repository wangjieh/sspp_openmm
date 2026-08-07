"""Configurable-patch DINOv3 backbone for the existing MMSeg pipeline.

``DINOv3BackboneMmseg`` remains unchanged for all existing p16 experiments.
This separately registered subclass adds two safeguards needed to fine-tune a
checkpoint at a different patch size:

* input dimensions must be divisible by both the Adapter pyramid stride (32)
  and the requested patch size; and
* the patch-projection kernel is resized when its checkpoint and target patch
  sizes differ, while every other incompatible tensor is skipped explicitly.

The Adapter still returns the normal stride-4/8/16/32 feature pyramid, so no
decode-head change is required when this backbone is selected in a config.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS

from custom_models.dinov3_backbone import DINOv3BackboneMmseg


def _normalize_checkpoint_state(checkpoint: str) -> Mapping[str, Any]:
    """Read the model-state mapping used by this project's DINOv3 checkpoints."""

    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(checkpoint_data, dict):
        for key in ('model', 'state_dict', 'teacher', 'student'):
            candidate = checkpoint_data.get(key)
            if isinstance(candidate, dict):
                checkpoint_data = candidate
                break
    if not isinstance(checkpoint_data, dict):
        raise TypeError(f'Checkpoint {checkpoint!r} does not contain a state dictionary.')
    return checkpoint_data


def _resample_patch_projection(weight: Tensor, target_size: Tuple[int, int]) -> Tensor:
    """Resize a Conv2D patch projector while preserving its DC response.

    Each output/input-channel kernel is resized independently in float32.
    Area compensation keeps the summed response of a constant image close to
    that of the source kernel; this is a stable initialization for subsequent
    fine-tuning, not a substitute for a checkpoint pretrained at that patch.
    """

    if weight.ndim != 4:
        raise ValueError('patch_embed.proj.weight must be a 4D convolution kernel.')
    source_height, source_width = weight.shape[-2:]
    target_height, target_width = target_size
    if (source_height, source_width) == target_size:
        return weight

    source_dtype = weight.dtype
    flattened = weight.float().reshape(-1, 1, source_height, source_width)
    resized = F.interpolate(
        flattened, size=target_size, mode='bicubic', align_corners=False
    )
    resized = resized * ((source_height * source_width) / (target_height * target_width))
    return resized.reshape(*weight.shape[:2], target_height, target_width).to(dtype=source_dtype)


@MODELS.register_module()
class DINOv3BackboneFlexiblePatch(DINOv3BackboneMmseg):
    """DINOv3 Adapter backbone that can be configured with a chosen patch size.

    Args:
        patch_size: Positive integer patch size, for example 16, 12, or 8.
        img_size: Configured training crop size.  It must be divisible by
            ``lcm(32, patch_size)`` when strict input checking is enabled.
        resample_patch_embed: Resize a checkpoint's patch-projection kernel
            when it was pretrained with another patch size.
        enforce_input_divisibility: Validate every runtime image size against
            ``lcm(32, patch_size)``.  This prevents PatchEmbed from silently
            discarding a border of a non-divisible image.

    A patch-size-matched checkpoint remains the preferred initialization.
    Resampling is intended for controlled p12/p8 fine-tuning experiments.
    """

    def __init__(
        self,
        patch_size: int = 16,
        img_size: int = 512,
        resample_patch_embed: bool = True,
        enforce_input_divisibility: bool = True,
        init_cfg: Any = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f'patch_size must be a positive integer, got {patch_size!r}.')
        if not isinstance(img_size, int) or img_size <= 0:
            raise ValueError(f'img_size must be a positive integer, got {img_size!r}.')

        self.configured_patch_size = patch_size
        self.input_size_divisor = math.lcm(32, patch_size)
        self.resample_patch_embed = bool(resample_patch_embed)
        self.enforce_input_divisibility = bool(enforce_input_divisibility)
        if self.enforce_input_divisibility and img_size % self.input_size_divisor:
            raise ValueError(
                f'img_size={img_size} is incompatible with patch_size={patch_size}. '
                f'Use a multiple of lcm(32, {patch_size})={self.input_size_divisor}; '
                'for example, use 480 or 576 for patch_size=12.'
            )

        super().__init__(
            patch_size=patch_size,
            img_size=img_size,
            init_cfg=init_cfg,
            **kwargs,
        )

    def _load_backbone_checkpoint(self, checkpoint: str) -> None:
        """Load all compatible DINOv3 weights and adapt the patch projector."""

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'DINOv3 checkpoint does not exist: {checkpoint_path}')

        checkpoint_state = _normalize_checkpoint_state(str(checkpoint_path))
        source_state: Dict[str, Tensor] = {}
        for key, value in checkpoint_state.items():
            if not isinstance(value, Tensor):
                continue
            if key.startswith('backbone.'):
                source_state[key[len('backbone.'):]] = value
            elif key.startswith('module.backbone.'):
                source_state[key[len('module.backbone.'):]] = value

        if not source_state:
            raise KeyError(
                'No backbone.* tensors were found in the checkpoint. '
                'Expected the same checkpoint layout used by DINOv3BackboneMmseg.'
            )

        target_state = self.backbone.state_dict()
        loadable_state: Dict[str, Tensor] = {}
        resized_keys: List[str] = []
        skipped_keys: List[str] = []
        for key, value in source_state.items():
            target_value = target_state.get(key)
            if target_value is None:
                skipped_keys.append(key)
                continue
            if value.shape == target_value.shape:
                loadable_state[key] = value
                continue
            if (
                key == 'patch_embed.proj.weight'
                and self.resample_patch_embed
                and value.ndim == 4
                and value.shape[:2] == target_value.shape[:2]
            ):
                loadable_state[key] = _resample_patch_projection(value, target_value.shape[-2:])
                resized_keys.append(key)
                continue
            skipped_keys.append(key)

        incompatible = self.backbone.load_state_dict(loadable_state, strict=False)
        print(
            '[DINOv3BackboneFlexiblePatch] checkpoint loaded: '
            f'loaded={len(loadable_state)} '
            f'resized={resized_keys or "none"} '
            f'missing={len(incompatible.missing_keys)} '
            f'unexpected={len(incompatible.unexpected_keys)} '
            f'skipped={len(skipped_keys)}'
        )
        if skipped_keys:
            preview = ', '.join(skipped_keys[:8])
            suffix = ' ...' if len(skipped_keys) > 8 else ''
            print(f'[DINOv3BackboneFlexiblePatch] skipped incompatible keys: {preview}{suffix}')

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.enforce_input_divisibility:
            height, width = x.shape[-2:]
            if height % self.input_size_divisor or width % self.input_size_divisor:
                raise ValueError(
                    f'Input size {(height, width)} is incompatible with patch_size='
                    f'{self.configured_patch_size}; both dimensions must be divisible by '
                    f'{self.input_size_divisor}.'
                )
        return super().forward(x)
