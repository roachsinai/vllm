# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from vllm.logger import init_logger

logger = init_logger(__name__)


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


class StandaloneInternVisionEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=getattr(config, "num_channels", 3),
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim)
        )

    def _get_pos_embed(self, pos_embed: torch.Tensor, height: int, width: int):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(
                1,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        pos_embed = F.interpolate(
            pos_embed, size=(height, width), mode="bicubic", align_corners=False
        )
        return pos_embed.reshape(1, -1, height * width).permute(0, 2, 1).to(
            target_dtype
        )

    def _get_position_embedding(self, height: int, width: int) -> torch.Tensor:
        position_embedding = self.position_embedding
        if self.num_patches == height * width:
            return position_embedding
        return torch.cat(
            [
                position_embedding[:, :1, :],
                self._get_pos_embed(position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(target_dtype))
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        return embeddings + self._get_position_embedding(height, width).to(target_dtype)


class StandaloneInternAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = hidden_states.shape
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.qk_normalization:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        hidden_states = attn @ v
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, seq_len, channels
        )
        return self.proj(hidden_states)


class StandaloneInternMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.fc2(hidden_states)


class StandaloneInternVisionEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.attn = StandaloneInternAttention(config)
        self.mlp = StandaloneInternMLP(config)
        norm_cls = NORM2FN[config.norm_type]
        self.norm1 = norm_cls(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = norm_cls(config.hidden_size, eps=config.layer_norm_eps)
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(config.hidden_size))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states)) * self.ls1
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2
        return hidden_states


class StandaloneInternVisionEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig, num_hidden_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            StandaloneInternVisionEncoderLayer(config)
            for _ in range(num_hidden_layers)
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class StandaloneInternVisionModel(nn.Module):
    def __init__(self, config: PretrainedConfig, num_hidden_layers: int | None = None):
        super().__init__()
        self.config = config
        if num_hidden_layers is None:
            num_hidden_layers = config.num_hidden_layers
        self.embeddings = StandaloneInternVisionEmbeddings(config)
        self.encoder = StandaloneInternVisionEncoder(config, num_hidden_layers)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        return self.encoder(inputs_embeds=hidden_states)


def export_intern_vit_onnx(
    *,
    model: nn.Module,
    output_path: Path,
    vision_config: PretrainedConfig,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    opset_version: int = 18,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_values = torch.ones(
        batch_size,
        getattr(vision_config, "num_channels", 3),
        vision_config.image_size,
        vision_config.image_size,
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values,),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamic_shapes={
                "pixel_values": {0: torch.export.Dim("batch_size")},
            },
        )


def build_intern_vit_engine(
    *,
    onnx_path: Path,
    engine_path: Path,
    vision_config: PretrainedConfig,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    workspace_gb: float,
) -> None:
    import tensorrt as trt

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    logger_trt = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger_trt)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with builder.create_network(network_flags) as network:
        parser = trt.OnnxParser(network, logger_trt)
        if not parser.parse_from_file(str(onnx_path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"Failed to parse InternViT ONNX model {onnx_path}: {errors}"
            )

        config = builder.create_builder_config()
        workspace_bytes = int(workspace_gb * (1 << 30))
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config.max_workspace_size = workspace_bytes

        config.set_flag(trt.BuilderFlag.FP16)
        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            min=(
                min_batch,
                getattr(vision_config, "num_channels", 3),
                vision_config.image_size,
                vision_config.image_size,
            ),
            opt=(
                opt_batch,
                getattr(vision_config, "num_channels", 3),
                vision_config.image_size,
                vision_config.image_size,
            ),
            max=(
                max_batch,
                getattr(vision_config, "num_channels", 3),
                vision_config.image_size,
                vision_config.image_size,
            ),
        )
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build InternViT TensorRT engine.")
        engine_path.write_bytes(serialized_engine)


class InternVisionTRTModel(nn.Module):
    def __init__(
        self,
        engine_path: Path,
        device: torch.device,
        max_input_shape: tuple[int, ...],
        output_dtype: torch.dtype,
    ):
        super().__init__()
        import tensorrt as trt

        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.device = device
        runtime_logger = trt.Logger(trt.Logger.WARNING)
        with trt.Runtime(runtime_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.input_name = None
        self.output_name = None
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Failed to find TensorRT input/output tensors.")

        if not self.context.set_input_shape(self.input_name, max_input_shape):
            raise RuntimeError(
                f"Failed to set max input shape: {self.input_name}={max_input_shape}"
            )
        max_output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in max_output_shape):
            raise RuntimeError(
                f"Max output shape is still dynamic: "
                f"{self.output_name}={max_output_shape}"
            )
        self.output_buffer = torch.empty(
            max_output_shape,
            dtype=output_dtype,
            device=self.device,
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        pixel_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pixel_embeds is not None:
            raise ValueError("InternVisionTRTModel only supports pixel_values input.")
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if pixel_values.device != self.device:
            raise RuntimeError(
                f"InternVisionTRTModel expected input on {self.device}, "
                f"got {pixel_values.device}"
            )

        trt_input = pixel_values.contiguous()
        assert self.input_name is not None
        assert self.output_name is not None
        if not self.context.set_input_shape(self.input_name, tuple(trt_input.shape)):
            raise RuntimeError(
                f"Failed to set input shape: {self.input_name}={tuple(trt_input.shape)}"
            )

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(
                f"Output shape is still dynamic: {self.output_name}={output_shape}"
            )

        trt_output = self.output_buffer[: output_shape[0]]
        self.context.set_tensor_address(self.input_name, trt_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buffer.data_ptr())
        stream = torch.cuda.current_stream(device=self.device)
        self.context.execute_async_v3(stream.cuda_stream)
        return trt_output

