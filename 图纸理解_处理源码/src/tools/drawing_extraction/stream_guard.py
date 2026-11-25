"""Stop degenerate OCR streams so native-resolution crop recovery can proceed."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletion


PERIODIC_TAIL_CHARACTERS = 512
CONSECUTIVE_NUMBER_LINES = 128
POLICY = "ocr_stream_repetition_v1"


@dataclass
class RepetitionGuard:
    """Flag suspicious generation, never establish that source content is absent."""

    tail: str = ""
    line: str = ""
    line_overflow: bool = False
    last_number: int | None = None
    number_run: int = 0

    def feed(self, delta: str) -> str | None:
        for char in delta:
            if char != "\n":
                if len(self.line) < 32:
                    self.line += char
                else:
                    self.line_overflow = True
                continue
            text = self.line.strip()
            if (not self.line_overflow and text.isascii() and text.isdecimal()
                    and len(text) <= 9 and (text == "0" or not text.startswith("0"))):
                number = int(text)
                self.number_run = (self.number_run + 1
                                   if self.last_number is not None
                                   and number == self.last_number + 1 else 1)
                self.last_number = number
            else:
                self.last_number = None
                self.number_run = 0
            self.line = ""
            self.line_overflow = False
            if self.number_run >= CONSECUTIVE_NUMBER_LINES:
                return "consecutive_number_lines"

        self.tail = (self.tail + delta)[-PERIODIC_TAIL_CHARACTERS:]
        if (len(self.tail) == PERIODIC_TAIL_CHARACTERS and self.tail.strip()
                and len(set(self.tail)) <= 3):
            for period in range(1, 5):
                if all(char == self.tail[index % period]
                       for index, char in enumerate(self.tail)):
                    return "periodic_character_tail"
        return None


async def guarded_completion(
    create: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any,
) -> ChatCompletion:
    """Consume a vLLM stream within the caller's existing concurrency limit."""

    if kwargs.get("stream"):
        raise ValueError("OCR guard expects a non-streaming caller")
    options = dict(kwargs.pop("stream_options", None) or {})
    options.update(include_usage=True, continuous_usage_stats=True)
    stream = await create(*args, **kwargs, stream=True, stream_options=options)
    guard = RepetitionGuard()
    parts: list[str] = []
    usage = None
    first = None
    server_finish = None
    abort_reason = None
    async with stream:
        async for chunk in stream:
            if first is None:
                first = chunk
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            if len(chunk.choices) != 1 or chunk.choices[0].index != 0:
                raise ValueError("OCR guard requires exactly one completion")
            choice = chunk.choices[0]
            delta = choice.delta.content or ""
            parts.append(delta)
            if choice.finish_reason is not None:
                server_finish = choice.finish_reason
            if server_finish is None:
                abort_reason = guard.feed(delta)
                if abort_reason is not None:
                    break
    if first is None or (server_finish is None and abort_reason is None):
        raise RuntimeError("OCR stream ended without a completion status")

    # The SDK's response enum has no client-abort value. Preserve its distinction
    # explicitly; the evidence collector records client_abort and rejects it.
    return ChatCompletion.model_validate({
        "id": first.id, "created": first.created, "model": first.model,
        "object": "chat.completion",
        "choices": [{"index": 0, "logprobs": None,
                     "message": {"role": "assistant", "content": "".join(parts)},
                     "finish_reason": server_finish or "length"}],
        "usage": usage.model_dump() if usage is not None else None,
        "stream_guard": {
            "policy": POLICY,
            "status": "aborted" if abort_reason else "completed",
            "reason": abort_reason,
            "server_finish_reason": server_finish,
            "generated_characters": sum(map(len, parts)),
            "usage_complete": abort_reason is None and usage is not None,
        },
    })


def install_stream_guard(recognizer: Any) -> None:
    """Adapt the pinned PaddleX client without changing prompts or input images."""

    client = recognizer.genai_client
    if client.backend != "vllm-server":
        raise ValueError("OCR stream guard requires the vLLM backend")
    completions = client.openai_client.chat.completions
    if getattr(completions, "_cad_stream_guard_enabled", False):
        return
    create = completions.create

    async def guarded_create(*args: Any, **kwargs: Any) -> ChatCompletion:
        return await guarded_completion(create, *args, **kwargs)

    completions.create = guarded_create
    completions._cad_stream_guard_enabled = True
