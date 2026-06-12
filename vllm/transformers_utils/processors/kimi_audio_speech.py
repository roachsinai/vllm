# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Discrete speech-token extraction helpers for Kimi-Audio."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from glob import glob
from math import gcd
from typing import Any

import numpy as np
import safetensors
import torch
from huggingface_hub import snapshot_download
from transformers import WhisperFeatureExtractor

from vllm.transformers_utils.processors.kimi_audio_whisper_vq import (
    WhisperVQConfig,
    WhisperVQEncoder,
)

KIMIA_SPEECH_TOKENIZER_REPO = "THUDM/glm-4-voice-tokenizer"
KIMIA_SPEECH_TOKENIZER_ENV = "KIMI_AUDIO_SPEECH_TOKENIZER_PATH"
KIMIA_SPEECH_TOKENIZER_DEVICE_ENV = "KIMI_AUDIO_SPEECH_TOKENIZER_DEVICE"


def _resolve_speech_tokenizer_path(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    env_path = os.environ.get(KIMIA_SPEECH_TOKENIZER_ENV)
    if env_path and os.path.isdir(env_path):
        return env_path

    try:
        return snapshot_download(model_name_or_path, local_files_only=True)
    except Exception:
        return model_name_or_path


def _load_local_quantize_encoder(
    model_path: str, device: torch.device | str
) -> Any:
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    weight_paths = sorted(glob(os.path.join(model_path, "model*.safetensors")))
    if not weight_paths:
        raise FileNotFoundError(
            f"No model*.safetensors files found in Kimi-Audio speech tokenizer "
            f"path: {model_path}"
        )
    for path in weight_paths:
        with safetensors.safe_open(path, framework="pt", device="cpu") as weights_file:
            for key in tuple(weights_file.keys()):
                state_dict[key] = weights_file.get_tensor(key)

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


class KimiAudioSpeechTokenizer:
    def __init__(
        self,
        model_name_or_path: str = KIMIA_SPEECH_TOKENIZER_REPO,
        *,
        token_offset: int = 152064,
        device: torch.device | str | None = None,
        model: Any | None = None,
        feature_extractor: WhisperFeatureExtractor | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.token_offset = token_offset
        env_device = os.environ.get(KIMIA_SPEECH_TOKENIZER_DEVICE_ENV)
        self.device = (
            device or env_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = model
        self._feature_extractor = feature_extractor

    def _ensure_loaded(self) -> tuple[Any, WhisperFeatureExtractor]:
        resolved_model_path = _resolve_speech_tokenizer_path(self.model_name_or_path)
        if self._model is None:
            self._model = _load_local_quantize_encoder(
                resolved_model_path,
                self.device,
            )
        if self._feature_extractor is None:
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                resolved_model_path,
                local_files_only=resolved_model_path != self.model_name_or_path,
            )
        return self._model, self._feature_extractor

    def _resample_if_needed(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_sr: int,
    ) -> np.ndarray:
        if sample_rate == target_sr:
            return audio
        from scipy.signal import resample_poly

        sample_gcd = gcd(int(sample_rate), int(target_sr))
        resampled = resample_poly(
            audio,
            up=target_sr // sample_gcd,
            down=sample_rate // sample_gcd,
        )
        return np.asarray(resampled, dtype=np.float32)

    def _to_mono_audio(self, audio: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim <= 1:
            return audio_array.reshape(-1)
        if audio_array.shape[0] <= audio_array.shape[-1]:
            return audio_array[0]
        return audio_array[:, 0]

    def encode(
        self,
        audios: Sequence[np.ndarray | torch.Tensor],
        *,
        sampling_rate: int = 16000,
    ) -> list[list[int]]:
        model, feature_extractor = self._ensure_loaded()
        dtype = model.conv1.weight.dtype
        target_sr = int(feature_extractor.sampling_rate)

        segments: list[np.ndarray] = []
        segment_to_audio_idx: list[int] = []
        for audio_idx, audio in enumerate(audios):
            mono_audio = self._to_mono_audio(audio)
            mono_audio = self._resample_if_needed(
                mono_audio,
                sampling_rate,
                target_sr,
            )
            if mono_audio.size == 0:
                continue

            time_step = 0
            while time_step * target_sr < mono_audio.shape[0]:
                segment = mono_audio[
                    time_step * target_sr : (time_step + 30) * target_sr
                ]
                segments.append(segment)
                segment_to_audio_idx.append(audio_idx)
                time_step += 30

        all_speech_tokens: list[list[int]] = [[] for _ in range(len(audios))]
        if not segments:
            return all_speech_tokens

        pooling_kernel_size = getattr(model.config, "pooling_kernel_size", 1) or 1
        stride = (
            model.conv1.stride[0]
            * model.conv2.stride[0]
            * pooling_kernel_size
            * feature_extractor.hop_length
        )

        batch_size = 128
        with torch.no_grad():
            for start in range(0, len(segments), batch_size):
                batch_segments = segments[start : start + batch_size]
                features = feature_extractor(
                    batch_segments,
                    sampling_rate=target_sr,
                    return_attention_mask=True,
                    return_tensors="pt",
                    padding="longest",
                    pad_to_multiple_of=stride,
                )
                input_features = features["input_features"].to(self.device).to(dtype)
                attention_mask = features["attention_mask"].to(self.device)
                outputs = model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                )
                speech_tokens = outputs.quantized_token_ids
                attention_mask = attention_mask[
                    :, :: model.conv1.stride[0] * model.conv2.stride[0]
                ]
                attention_mask = attention_mask[:, ::pooling_kernel_size]

                for i in range(len(speech_tokens)):
                    audio_idx = segment_to_audio_idx[start + i]
                    valid_tokens = speech_tokens[i][attention_mask[i].bool()].tolist()
                    all_speech_tokens[audio_idx].extend(
                        int(token) + int(self.token_offset) for token in valid_tokens
                    )

        return all_speech_tokens


@lru_cache(maxsize=4)
def cached_get_kimi_audio_speech_tokenizer(
    token_offset: int = 152064,
    model_name_or_path: str = KIMIA_SPEECH_TOKENIZER_REPO,
) -> KimiAudioSpeechTokenizer:
    return KimiAudioSpeechTokenizer(
        model_name_or_path=model_name_or_path,
        token_offset=token_offset,
    )
