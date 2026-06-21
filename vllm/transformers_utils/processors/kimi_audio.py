# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The Moonshot AI team and the HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Processor for Kimi-Audio ASR and text-output modes."""

from collections.abc import Sequence

import numpy as np
import torch
from transformers import BatchFeature, ProcessorMixin
from transformers.audio_utils import AudioInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class KimiAudioProcessor(ProcessorMixin):
    # Required for ProcessorMixin
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    # Special token IDs
    KIMIA_MEDIA_BEGIN: int = 151661
    KIMIA_MEDIA_END: int = 151663
    KIMIA_TEXT_BLANK: int = 151666
    KIMIA_TEXT_EOS: int = 151667
    KIMIA_SPEECH_CT_ID: int = 151675
    KIMIA_SPEECH_CTD_ID: int = 151676

    # Audio processing constants
    AUDIO_SEQ_LEN: int = 376

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        speech_tokenizer=None,
        **kwargs,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.speech_tokenizer = speech_tokenizer

    def _normalize_audio(
        self,
        audio: AudioInput | None,
    ) -> tuple[list[np.ndarray], list[int]]:
        if audio is None:
            return [], []
        if isinstance(audio, np.ndarray):
            audio = [audio]

        normalized_audio: list[np.ndarray] = []
        audio_lengths: list[int] = []
        for aud in audio:
            if isinstance(aud, torch.Tensor):
                aud = aud.detach().cpu().numpy()
            aud_np = np.asarray(aud, dtype=np.float32)
            normalized_audio.append(aud_np)
            audio_lengths.append(int(aud_np.shape[-1]))
        return normalized_audio, audio_lengths

    def _build_speech_token_tensors(
        self,
        speech_token_lists: Sequence[Sequence[int]],
    ) -> dict[str, torch.Tensor]:
        max_len = max((len(tokens) for tokens in speech_token_lists), default=0)
        if max_len <= 0:
            return {}

        speech_token_ids = torch.full(
            (len(speech_token_lists), max_len),
            fill_value=-1,
            dtype=torch.long,
        )
        speech_attention_mask = torch.zeros(
            (len(speech_token_lists), max_len),
            dtype=torch.long,
        )
        for row_idx, token_ids in enumerate(speech_token_lists):
            token_count = len(token_ids)
            if token_count == 0:
                continue
            speech_token_ids[row_idx, :token_count] = torch.tensor(
                token_ids,
                dtype=torch.long,
            )
            speech_attention_mask[row_idx, :token_count] = 1

        return {
            "speech_token_ids": speech_token_ids,
            "speech_attention_mask": speech_attention_mask,
        }

    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = None,
        audio: AudioInput | None = None,
        return_tensors: str = "pt",
        return_discrete_audio_token_ids: bool | None = None,
        messages: Sequence[dict[str, object]] | None = None,
        output_type: str = "text",
        audio_sampling_rate: int = 16000,
        **kwargs,
    ) -> BatchFeature:
        if return_discrete_audio_token_ids is None:
            return_discrete_audio_token_ids = output_type == "text"

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(
                text, return_tensors=return_tensors, padding=True
            )
        else:
            text_inputs = {}

        padded_audio, audio_lengths = self._normalize_audio(audio)
        if padded_audio:
            audio_inputs = self.feature_extractor(
                padded_audio,
                sampling_rate=audio_sampling_rate,
                padding="max_length",
                return_attention_mask=True,
                return_tensors=return_tensors,
            )
            if "input_features" in audio_inputs:
                audio_inputs["whisper_input_features"] = audio_inputs.pop(
                    "input_features"
                )
            if "attention_mask" in audio_inputs:
                audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                    "attention_mask"
                )
            audio_inputs["audio_sample_lengths"] = torch.tensor(
                audio_lengths,
                dtype=torch.long,
            )
        else:
            audio_inputs = {}

        speech_token_lists: list[list[int]] = []
        need_speech_tokens = (
            padded_audio
            and self.speech_tokenizer is not None
            and return_discrete_audio_token_ids
        )
        if need_speech_tokens:
            speech_tokenizer = self.speech_tokenizer
            assert speech_tokenizer is not None
            speech_token_lists = speech_tokenizer.encode(
                padded_audio,
                sampling_rate=audio_sampling_rate,
            )

        if return_discrete_audio_token_ids and speech_token_lists:
            audio_inputs.update(self._build_speech_token_tensors(speech_token_lists))

        return BatchFeature(
            data={**text_inputs, **audio_inputs},
            tensor_type=return_tensors,
        )
