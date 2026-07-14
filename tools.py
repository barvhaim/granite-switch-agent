# SPDX-License-Identifier: Apache-2.0
"""Tools the Granite Switch ReAct agent can call.

Two kinds live here:

1. **External tools** — ordinary actions the base model chooses to run. Here that's
   ``search_docs`` over a tiny in-memory corpus (a stand-in for a real retriever).

2. **Adapter tools** — thin wrappers whose *implementation* is a second call to the
   same model with an embedded adapter activated (``adapter_name``). The base model
   picks these like any tool; the harness (agent.py) executes them by re-invoking
   llama-server. Only the envelope-free RAG adapters are exposed:
   ``query_rewrite``, ``answerability``, ``query_clarification``.

Each tool has an OpenAI-style JSON schema (for the model) and a Python callable
(for the harness). ADAPTER_TOOLS names which tools are adapter-backed.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from llama_client import LlamaClient

# ---------------------------------------------------------------------------
# Corpus — a handful of documents. A real agent would hit a vector store; this
# keeps the demo self-contained and deterministic.
# ---------------------------------------------------------------------------

CORPUS: list[dict[str, str]] = [
    {
        "doc_id": "france",
        "text": (
            "France is a country in Western Europe. Its capital is Paris, which "
            "sits on the Seine river. France uses the euro as its currency."
        ),
    },
    {
        "doc_id": "everest",
        "text": (
            "Mount Everest is the tallest mountain on Earth, standing 8,849 "
            "meters above sea level. It lies on the border between Nepal and Tibet."
        ),
    },
    {
        "doc_id": "python",
        "text": (
            "Python is a high-level programming language created by Guido van "
            "Rossum and first released in 1991. It emphasizes code readability."
        ),
    },
    {
        "doc_id": "granite",
        "text": (
            "Granite Switch is an IBM Research system that embeds multiple LoRA "
            "adapter functions into one checkpoint, activated by control tokens, "
            "so a small model gains many specialized capabilities."
        ),
    },
]


# Stable 1-based label per doc_id, fixed by corpus order (france -> "Doc 1").
# Used for human-friendly trace output; a doc keeps the same number across searches.
_DOC_NUMBER: dict[str, int] = {d["doc_id"]: i for i, d in enumerate(CORPUS, 1)}


def doc_label(doc: dict[str, Any]) -> str:
    """Human-friendly label for a retrieved doc, e.g. ``Doc 1 (france)``."""
    doc_id = doc["doc_id"]
    return f"Doc {_DOC_NUMBER[doc_id]} ({doc_id})"


def search_docs(query: str, top_k: int = 2) -> list[dict[str, Any]]:
    """Keyword-overlap retrieval over the in-memory corpus.

    Returns the ``top_k`` documents scored by how many query words they contain.
    Deliberately simple — the point of the demo is the adapter flow, not the
    retriever.
    """
    words = {w.lower().strip("?.,!") for w in query.split() if len(w) > 2}
    scored = []
    for doc in CORPUS:
        text = doc["text"].lower()
        score = sum(1 for w in words if w in text)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# Adapter-backed tools — implemented by re-invoking the model with an adapter.
# The signature (client, args, documents) is uniform so the harness can call any
# of them the same way.
# ---------------------------------------------------------------------------


def _adapter_query_rewrite(
    client: LlamaClient, args: dict[str, Any], documents: list[dict] | None
) -> Any:
    """Decontextualize a follow-up question using conversation history.

    ``args`` carries ``history`` (list of {role, content}) and ``question``.
    """
    history = args.get("history") or []
    question = args.get("question", "")
    messages = list(history) + [{"role": "user", "content": question}]
    out = client.chat(messages, adapter_name="query_rewrite", max_tokens=128)
    parsed = _try_json(out.content)
    if isinstance(parsed, dict) and "rewritten_question" in parsed:
        return parsed["rewritten_question"]
    return out.content.strip()


def _adapter_answerability(
    client: LlamaClient, args: dict[str, Any], documents: list[dict] | None
) -> Any:
    """Decide whether the retrieved documents can answer the question."""
    question = args.get("question", "")
    out = client.chat(
        [{"role": "user", "content": question}],
        adapter_name="answerability",
        documents=documents or [],
        max_tokens=16,
    )
    return _try_json(out.content) or out.content.strip().strip('"')


def _adapter_query_clarification(
    client: LlamaClient, args: dict[str, Any], documents: list[dict] | None
) -> Any:
    """Return CLEAR if the docs suffice, else a clarifying follow-up question."""
    question = args.get("question", "")
    out = client.chat(
        [{"role": "user", "content": question}],
        adapter_name="query_clarification",
        documents=documents or [],
        max_tokens=128,
    )
    parsed = _try_json(out.content)
    if isinstance(parsed, dict) and "clarification" in parsed:
        return parsed["clarification"]
    return out.content.strip()


def _try_json(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Registry — schemas advertised to the model + the callables the harness runs.
# ---------------------------------------------------------------------------

# Adapter tools take (client, args, documents); external tools take just args and
# are dispatched separately in the harness. ADAPTER_IMPLS maps tool name -> fn.
ADAPTER_IMPLS: dict[str, Callable[[LlamaClient, dict, list | None], Any]] = {
    "query_rewrite": _adapter_query_rewrite,
    "answerability": _adapter_answerability,
    "query_clarification": _adapter_query_clarification,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Retrieve relevant documents from the knowledge base for a query. "
                "Call this to gather evidence before answering a factual question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_rewrite",
            "description": (
                "Rewrite a follow-up question into a standalone, self-contained "
                "question using the conversation history. Use when the user's "
                "question contains pronouns or references to earlier turns "
                "(it, that, the capital, etc.) before searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "history": {
                        "type": "array",
                        "description": "Prior turns as {role, content} objects",
                        "items": {"type": "object"},
                    },
                    "question": {
                        "type": "string",
                        "description": "The follow-up question to rewrite",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answerability",
            "description": (
                "Check whether the documents just retrieved actually contain the "
                "information needed to answer the question. Returns 'answerable' "
                "or 'unanswerable'. Call this after search_docs and before "
                "committing to an answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_clarification",
            "description": (
                "Given the retrieved documents, decide if the question is clear "
                "enough to answer (returns CLEAR) or needs a clarifying follow-up "
                "question. Use when the question seems ambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
]
