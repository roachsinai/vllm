# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal WhisperVQ encoder used by Kimi-Audio speech tokenization.

This module intentionally vendors only the encoder-side inference path needed
to extract discrete speech tokens from `THUDM/glm-4-voice-tokenizer`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperPreTrainedModel,
)


@dataclass
class QuantizedBaseModelOutput(BaseModelOutput):
    quantized_token_ids: torch.LongTensor | None = None


class WhisperVQConfig(WhisperConfig):
    def __init__(
        self,
        pooling_kernel_size=None,
        pooling_type="max",
        pooling_position=0,
        quantize_vocab_size=None,
        quantize_position=16,
        quantize_commit_coefficient=0.25,
        quantize_loss_scale=1.0,
        quantize_ema_decay=None,
        quantize_restart_interval=None,
        quantize_encoder_only=False,
        quantize_causal_encoder=False,
        quantize_causal_block_size=None,
        skip_language_detection=False,
        encoder_causal_attention=False,
        encoder_causal_convolution=False,
        **kwargs,
    ):
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_commit_coefficient = quantize_commit_coefficient
        self.quantize_loss_scale = quantize_loss_scale
        self.quantize_ema_decay = quantize_ema_decay
        self.quantize_restart_interval = quantize_restart_interval
        self.quantize_encoder_only = quantize_encoder_only
        self.quantize_causal_encoder = quantize_causal_encoder
        self.quantize_causal_block_size = quantize_causal_block_size
        self.skip_language_detection = skip_language_detection
        self.encoder_causal_attention = encoder_causal_attention
        self.encoder_causal_convolution = encoder_causal_convolution
        super().__init__(**kwargs)


def vector_quantize(
    inputs: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding_size = codebook.size(1)
    flattened = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook**2, dim=1)
    inputs_sqr = torch.sum(flattened**2, dim=1, keepdim=True)
    distances = torch.addmm(
        codebook_sqr + inputs_sqr,
        flattened,
        codebook.t(),
        alpha=-2.0,
        beta=1.0,
    )
    indices = torch.argmin(distances, dim=1)
    codes = torch.index_select(codebook, dim=0, index=indices).view_as(inputs)
    return codes, indices


def sinusoids(
    length: int,
    channels: int,
    max_timescale: float = 10000,
) -> torch.Tensor:
    if channels % 2 != 0:
        raise ValueError(
            "Number of channels must be divisible by 2 for sinusoidal "
            f"positional embeddings, got {channels}."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2)
    )
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(inp, (self.left_padding, 0)))


class WhisperAttention(nn.Module):
    """Minimal Whisper self-attention used by the WhisperVQ encoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads, got "
                f"embed_dim={embed_dim} and num_heads={num_heads}."
            )
        self.scaling = self.head_dim**-0.5
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(
        self,
        tensor: torch.Tensor,
        seq_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        return (
            tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, target_length, _ = hidden_states.size()

        query_states = self._shape(
            self.q_proj(hidden_states) * self.scaling,
            target_length,
            batch_size,
        )
        key_states = self._shape(
            self.k_proj(hidden_states),
            -1,
            batch_size,
        )
        value_states = self._shape(
            self.v_proj(hidden_states),
            -1,
            batch_size,
        )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[..., :key_states.shape[-2]]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )
        attn_output = torch.matmul(attn_probs, value_states)
        if attn_output.size() != (
            batch_size,
            self.num_heads,
            target_length,
            self.head_dim,
        ):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(batch_size, self.num_heads, target_length, self.head_dim)}, "
                f"but is {attn_output.size()}."
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class WhisperSdpaAttention(WhisperAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, target_length, _ = hidden_states.size()
        query_states = self._shape(
            self.q_proj(hidden_states),
            target_length,
            batch_size,
        )
        key_states = self._shape(
            self.k_proj(hidden_states),
            -1,
            batch_size,
        )
        value_states = self._shape(
            self.v_proj(hidden_states),
            -1,
            batch_size,
        )

        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[..., :key_states.shape[-2]]
        is_causal = self.is_causal and causal_mask is None and target_length > 1

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        if attn_output.size() != (
            batch_size,
            self.num_heads,
            target_length,
            self.head_dim,
        ):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(batch_size, self.num_heads, target_length, self.head_dim)}, "
                f"but is {attn_output.size()}."
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


WHISPER_ATTENTION_CLASSES = {
    "eager": WhisperAttention,
    "sdpa": WhisperSdpaAttention,
}


class WhisperVQEncoderLayer(nn.Module):

    def __init__(self, config: WhisperVQConfig, is_causal: bool = False):
        super().__init__()
        self.embed_dim = config.d_model
        attn_cls = WHISPER_ATTENTION_CLASSES.get(config._attn_implementation)
        if attn_cls is None:
            raise ValueError(
                "WhisperVQ supports only eager or sdpa attention, got "
                f"{config._attn_implementation!r}."
            )
        self.self_attn = attn_cls(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_causal=is_causal,
        )
        self.is_causal = is_causal
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask if not self.is_causal else None,
        )
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states,
                min=-clamp_value,
                max=clamp_value,
            )

        return hidden_states


class WhisperVQEncoder(WhisperPreTrainedModel):
    config_class = WhisperVQConfig

    def __init__(self, config: WhisperVQConfig):
        if not getattr(config, "_attn_implementation", None):
            config._attn_implementation = (
                "sdpa" if hasattr(F, "scaled_dot_product_attention") else "eager"
            )
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        conv_class = CausalConv1d if config.encoder_causal_convolution else nn.Conv1d
        self.conv1 = conv_class(
            self.num_mel_bins,
            embed_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = conv_class(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        num_layers = (
            config.quantize_position
            if config.quantize_encoder_only
            else config.encoder_layers
        )
        self.layers = nn.ModuleList([
            WhisperVQEncoderLayer(
                config,
                is_causal=(
                    config.encoder_causal_attention
                    or (
                        config.quantize_causal_encoder
                        and (
                            config.quantize_encoder_only
                            or layer_idx < config.quantize_position
                        )
                    )
                ),
            )
            for layer_idx in range(num_layers)
        ])
        self.layer_norm = (
            None
            if config.quantize_encoder_only
            else nn.LayerNorm(config.d_model)
        )

        self.gradient_checkpointing = False
        self.pooling_layer: nn.Module | None = None
        self.codebook: nn.Embedding | None = None
        self.embed_positions2: nn.Embedding | None = None

        self._init_pooling_layer()
        self._init_quantize_layer()
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, WhisperVQEncoder):
            with torch.no_grad():
                module.embed_positions.weight.copy_(
                    sinusoids(*module.embed_positions.weight.shape)
                )
                if module.embed_positions2 is not None:
                    max_positions = module.embed_positions2.weight.shape[0]
                    module.embed_positions2.weight.copy_(
                        module.embed_positions.weight[:max_positions]
                    )

    def _init_pooling_layer(self) -> None:
        if self.config.pooling_kernel_size is None:
            return
        if self.config.pooling_type == "max":
            self.pooling_layer = nn.MaxPool1d(
                kernel_size=self.config.pooling_kernel_size
            )
        elif self.config.pooling_type == "avg":
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.config.pooling_kernel_size
            )
        else:
            raise NotImplementedError(
                f"Pooling type {self.config.pooling_type} not implemented"
            )

    def _init_quantize_layer(self) -> None:
        if self.config.quantize_vocab_size is None:
            return
        if self.config.pooling_position is not None:
            assert self.config.quantize_position >= self.config.pooling_position
        self.codebook = nn.Embedding(
            self.config.quantize_vocab_size,
            self.config.d_model,
        )
        max_source_positions = self.max_source_positions
        if self.config.pooling_kernel_size is not None:
            max_source_positions = math.ceil(
                max_source_positions / self.config.pooling_kernel_size
            )
        self.embed_positions2 = nn.Embedding(
            max_source_positions,
            self.config.d_model,
        )
        if self.config.quantize_ema_decay is not None:
            self.codebook.weight.requires_grad = False
            self.register_buffer(
                "ema_count",
                torch.ones(self.config.quantize_vocab_size, dtype=torch.float),
            )
            self.register_buffer(
                "ema_weight",
                self.codebook.weight.data.clone().float(),
            )

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def get_block_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        block_size: int,
    ) -> torch.Tensor:
        dtype = self.conv1.weight.dtype
        _, seq_length = attention_mask.shape
        causal_mask = torch.tril(
            torch.ones(
                1,
                seq_length,
                seq_length,
                dtype=torch.bool,
                device=attention_mask.device,
            )
        )
        block_square_mask = []
        for start in range(0, seq_length, block_size):
            end = min(start + block_size, seq_length)
            length = end - start
            block_square_mask.append(causal_mask.new_ones((length, length)))
        block_square_mask = torch.block_diag(*block_square_mask)
        block_causal_mask = causal_mask | block_square_mask
        block_causal_mask = block_causal_mask & attention_mask[:, None, :]
        block_causal_mask = block_causal_mask.to(dtype=dtype)
        block_causal_mask = (1.0 - block_causal_mask) * torch.finfo(dtype).min
        return block_causal_mask.unsqueeze(1)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> QuantizedBaseModelOutput:
        batch_size, _, seq_length = input_features.shape
        stride = self.conv1.stride[0] * self.conv2.stride[0]
        seq_length = seq_length // stride

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_features.shape[-1]),
                dtype=torch.long,
                device=input_features.device,
            )
        attention_mask = attention_mask[:, ::stride]

        if self.config.quantize_causal_block_size is not None:
            extended_attention_mask = self.get_block_causal_attention_mask(
                attention_mask,
                block_size=self.config.quantize_causal_block_size,
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                (batch_size, seq_length),
            )

        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        hidden_states = inputs_embeds.permute(0, 2, 1)
        hidden_states = hidden_states + self.embed_positions.weight[:seq_length]
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )

        quantized_token_ids: torch.LongTensor | None = None

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, extended_attention_mask)

            if (
                idx + 1 == self.config.pooling_position
                and self.config.pooling_kernel_size is not None
            ):
                hidden_states = hidden_states.permute(0, 2, 1)
                if (
                    hidden_states.shape[-1] % self.config.pooling_kernel_size
                    != 0
                ):
                    hidden_states = F.pad(
                        hidden_states,
                        (
                            0,
                            self.config.pooling_kernel_size
                            - hidden_states.shape[-1]
                            % self.config.pooling_kernel_size,
                        ),
                    )
                assert self.pooling_layer is not None
                hidden_states = self.pooling_layer(hidden_states).permute(0, 2, 1)
                attention_mask = attention_mask[:, ::self.config.pooling_kernel_size]
                if self.config.quantize_causal_block_size is not None:
                    extended_attention_mask = self.get_block_causal_attention_mask(
                        attention_mask,
                        block_size=(
                            self.config.quantize_causal_block_size
                            // self.config.pooling_kernel_size
                        ),
                    )
                else:
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask,
                        (
                            batch_size,
                            seq_length // self.config.pooling_kernel_size,
                        ),
                    )

            if (
                idx + 1 == self.config.quantize_position
                and self.config.quantize_vocab_size is not None
            ):
                assert self.codebook is not None
                assert self.embed_positions2 is not None
                hidden_quantized, flat_indices = vector_quantize(
                    hidden_states,
                    self.codebook.weight,
                )
                quantized_token_ids = flat_indices.reshape(
                    batch_size,
                    hidden_quantized.shape[1],
                )
                hidden_states = hidden_quantized
                hidden_states = hidden_states + self.embed_positions2.weight[
                    : hidden_states.shape[1]
                ]

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        return QuantizedBaseModelOutput(
            last_hidden_state=hidden_states,
            quantized_token_ids=quantized_token_ids,
        )
