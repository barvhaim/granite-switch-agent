# SPDX-License-Identifier: Apache-2.0
"""Thin client over llama.cpp's OpenAI-compatible server for Granite Switch.

The one thing this adds over a plain OpenAI client is the ability to *activate an
embedded adapter function* for a single request. Granite Switch selects an adapter
by placing its control token in the prompt; the model's chat template does that when
it receives an ``adapter_name`` template variable. llama-server forwards
``chat_template_kwargs`` straight into the Jinja render, so activating the guardian
adapter is just::

    client.chat(messages, adapter_name="guardian-core")

With no ``adapter_name`` the base model runs normally (ReAct reasoning + tool calls).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

# The 12 adapters embedded in granite-switch-4.1-3b-preview, as declared in the
# model's own chat-template ``adapter_map``. Names are the values you pass as
# ``adapter_name``; the template maps each to its control token.
ADAPTERS = (
    "citations",
    "query_rewrite",
    "query_clarification",
    "hallucination_detection",
    "answerability",
    "factuality-detection",
    "policy-guardrails",
    "factuality-correction",
    "guardian-core",
    "uncertainty",
    "requirement-check",
    "context-attribution",
)


@dataclass
class ChatResult:
    """One assistant turn: free text plus any tool calls the model emitted."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class LlamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        model: str = "granite-switch",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LlamaClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        adapter_name: str | None = None,
        documents: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResult:
        """One completion.

        ``adapter_name`` (one of :data:`ADAPTERS`) activates an embedded adapter
        for this request only. ``documents`` are passed through the same
        ``chat_template_kwargs`` channel so the adapter/base model can ground on
        them (the template renders a ``<documents>`` block).
        """
        template_kwargs: dict[str, Any] = {}
        if adapter_name:
            if adapter_name not in ADAPTERS:
                raise ValueError(f"unknown adapter {adapter_name!r}; known: {ADAPTERS}")
            template_kwargs["adapter_name"] = adapter_name
        if documents is not None:
            template_kwargs["documents"] = documents

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs

        resp = self._http.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        return ChatResult(
            content=msg.get("content") or "",
            tool_calls=msg.get("tool_calls") or [],
            raw=data,
        )

    def health(self) -> bool:
        try:
            r = self._http.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False


def parse_json_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Extract the JSON arguments object from an OpenAI-style tool call."""
    args = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return {}
