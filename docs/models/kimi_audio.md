# Kimi-Audio

vLLM supports the text-output subset of
`moonshotai/Kimi-Audio-7B-Instruct` through
`KimiAudioForConditionalGeneration`.

## Supported Use Cases

- Audio-to-text transcription through the OpenAI-compatible
  `/v1/audio/transcriptions` and `/v1/audio/translations` endpoints.
- Audio understanding with text-only responses.
- Offline generation with audio inputs.
- Batched text-output inference.

This implementation intentionally does not support Kimi-Audio audio output
generation. Requests that require the official audio detokenizer, vocoder, or
`output_type="both"` path are outside the current vLLM support boundary.

## Serving

Use the Kimi-Audio tokenizer mode when serving the model:

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct \
  --tokenizer-mode kimi_audio \
  --trust-remote-code \
  --limit-mm-per-prompt '{"audio":1}'
```

The discrete speech tokenizer is loaded from `THUDM/glm-4-voice-tokenizer` by
default. To use a local copy, set:

```bash
export KIMI_AUDIO_SPEECH_TOKENIZER_PATH=/path/to/glm-4-voice-tokenizer
```

You can also set `KIMI_AUDIO_SPEECH_TOKENIZER_DEVICE` to force the speech
tokenizer device.

## Prompt Format

Kimi-Audio text-output requests must use the Kimi prompt format with the text
audio control token:

```text
<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|><|im_kimia_speech_ct_id|>
```

The vLLM implementation builds this prompt for transcription requests. For
offline generation examples, use `KimiAudioPromptBuilder` and pass structured
`messages` through `mm_processor_kwargs`.

## Scope Notes

The current implementation aligns text-output embeddings with the official
Kimi-Audio path by combining:

- discrete speech token embeddings from the GLM-4-Voice tokenizer;
- projected Whisper continuous audio features;
- the Kimi text stream blank token embedding.

The model still uses vLLM's standard multimodal embedding merge path, so CUDA
graph capture and chunked prefill can be used by the language-model forward
path.
