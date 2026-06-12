# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prompt builder helpers for the Kimi-Audio text-output subset."""

from __future__ import annotations

from collections.abc import Sequence


class KimiAudioPromptBuilder:
    """Build official-style text-output prompts for Kimi-Audio.

    This builder intentionally targets the subset currently supported by vLLM:
    user text, user audio, assistant text, and assistant generation prompts.
    """

    USER_START = "<|im_kimia_user_msg_start|>"
    ASSISTANT_START = "<|im_kimia_assistant_msg_start|>"
    MSG_END = "<|im_msg_end|>"
    MEDIA_BEGIN = "<|im_media_begin|>"
    MEDIA_END = "<|im_media_end|>"
    TEXT_BLANK = "<|im_kimia_text_blank|>"
    TEXT_EOS = "<|im_kimia_text_eos|>"
    SPEECH_CT = "<|im_kimia_speech_ct_id|>"
    SPEECH_CTD = "<|im_kimia_speech_ctd_id|>"

    AUDIO_PLACEHOLDER = f"{MEDIA_BEGIN}{TEXT_BLANK}{MEDIA_END}{SPEECH_CT}"

    @classmethod
    def build_audio_placeholder(cls, *, audio_count: int = 1) -> str:
        if audio_count < 0:
            raise ValueError("audio_count must be non-negative")
        return cls.AUDIO_PLACEHOLDER * audio_count

    @classmethod
    def build_user_audio_content(
        cls,
        request_prompt: str = "",
        *,
        audio_count: int = 1,
    ) -> str:
        audio_placeholder = cls.build_audio_placeholder(audio_count=audio_count)
        request_prompt = request_prompt.strip()
        if request_prompt:
            return f"{request_prompt}\n{audio_placeholder}"
        return audio_placeholder

    @classmethod
    def build_message(
        cls,
        *,
        role: str,
        content: str = "",
        message_type: str = "text",
        add_text_eos: bool | None = None,
        add_msg_end: bool = True,
        output_type: str = "text",
        audio_count: int = 1,
    ) -> str:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")
        if message_type not in {"text", "audio", None}:
            raise ValueError(f"Unsupported message_type: {message_type}")
        if output_type not in {"text", "both"}:
            raise ValueError(f"Unsupported output_type: {output_type}")
        if audio_count < 0:
            raise ValueError("audio_count must be non-negative")

        role_prefix = cls.USER_START if role == "user" else cls.ASSISTANT_START

        if add_text_eos is None:
            add_text_eos = role == "assistant" and message_type == "text"

        body = content.strip() if message_type == "text" and content else content
        if message_type == "audio":
            control_token = cls.SPEECH_CT if output_type == "text" else cls.SPEECH_CTD
            body = (
                f"{cls.MEDIA_BEGIN}{cls.TEXT_BLANK}{cls.MEDIA_END}{control_token}"
                * audio_count
            )
            if content:
                text_content = content.strip()
                body = f"{text_content}\n{body}" if text_content else body
        elif message_type is None:
            body = ""

        suffix = ""
        if add_text_eos:
            suffix += cls.TEXT_EOS
        if add_msg_end:
            suffix += cls.MSG_END

        return f"{role_prefix}{body}{suffix}"

    @classmethod
    def build_prompt_from_messages(
        cls,
        messages: Sequence[dict[str, object]],
        *,
        add_generation_prompt: bool = True,
        output_type: str = "text",
    ) -> str:
        prompt_parts: list[str] = []

        for message in messages:
            role = str(message["role"])
            message_type = str(message.get("message_type") or "text")
            content = str(message.get("content") or "")
            audio_count_value = message.get("audio_count", 1)
            audio_count = 1 if audio_count_value is None else int(audio_count_value)
            prompt_parts.append(
                cls.build_message(
                    role=role,
                    content=content,
                    message_type=message_type,
                    add_msg_end=True,
                    output_type=output_type,
                    audio_count=audio_count,
                )
            )

        if add_generation_prompt:
            prompt_parts.append(
                cls.build_message(
                    role="assistant",
                    message_type=None,
                    add_text_eos=False,
                    add_msg_end=False,
                    output_type=output_type,
                )
            )

        return "".join(prompt_parts)

    @classmethod
    def build_transcription_prompt(
        cls,
        request_prompt: str = "",
        *,
        audio_count: int = 1,
    ) -> str:
        return cls.build_prompt_from_messages(
            [
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": request_prompt,
                    "audio_count": audio_count,
                }
            ],
            add_generation_prompt=True,
            output_type="text",
        )
