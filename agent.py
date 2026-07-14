# SPDX-License-Identifier: Apache-2.0
"""A ReAct agent whose brain is Granite Switch running on llama.cpp.

The base model does the reasoning and decides which tools to call. Two tool
kinds are dispatched differently by the harness:

  * ``search_docs`` — an ordinary external tool (retrieval).
  * ``query_rewrite`` / ``answerability`` / ``query_clarification`` — *adapter*
    tools. The model chooses them like any tool, but the harness executes each by
    re-invoking the SAME model with that embedded adapter activated
    (``adapter_name``), then feeds the adapter's structured output back as the
    tool observation.

So the model selects the adapter (LLM-selected wiring) while the harness performs
the mechanically-correct activation. Retrieved documents are threaded through the
loop so adapter tools and the final answer stay grounded on the same evidence.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from llama_client import LlamaClient, parse_json_arguments
from tools import ADAPTER_IMPLS, TOOL_SCHEMAS, doc_label, search_docs

SYSTEM_PROMPT = (
    "You are a careful research assistant.\n"
    "For greetings, small talk, or questions about your own capabilities, reply "
    "directly in one sentence WITHOUT calling any tools.\n"
    "For factual questions, answer using ONLY information you retrieve with the "
    "tools, following this workflow:\n"
    "1. If the question refers to earlier turns (pronouns, 'the capital', 'it'), "
    "call query_rewrite first to make it standalone.\n"
    "2. Call search_docs to gather evidence.\n"
    "3. Call answerability to confirm the documents can answer the question.\n"
    "4. If answerable, give a concise grounded answer. If unanswerable, say so "
    "plainly rather than guessing.\n"
    "For factual questions, do not answer from prior knowledge; rely on retrieved "
    "documents."
)

MAX_STEPS = 8


class Colors:
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


class Agent:
    def __init__(self, client: LlamaClient, *, verbose: bool = True) -> None:
        self.client = client
        self.verbose = verbose
        self.retrieved: list[dict[str, Any]] = []
        # Persists across turns: user/assistant pairs only. The per-turn ReAct
        # scratchpad (tool calls + observations) is kept out to stay lean.
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.history = []
        self.retrieved = []

    def run(self, question: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": question},
        ]
        self.retrieved = []

        for step in range(1, MAX_STEPS + 1):
            result = self.client.chat(messages, tools=TOOL_SCHEMAS, max_tokens=512)

            if not result.tool_calls:
                # No tool call → the model is answering.
                self._log(f"\n{_c('ANSWER', Colors.BOLD + Colors.GREEN)}")
                answer = result.content.strip()
                self._commit(question, answer)
                return answer

            # Record the assistant turn (with its tool calls) before observations.
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": result.tool_calls,
                }
            )

            for call in result.tool_calls:
                name = call.get("function", {}).get("name", "")
                args = parse_json_arguments(call)
                observation = self._dispatch(step, name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "content": json.dumps(observation, default=str),
                    }
                )

        stopped = "(stopped: reached max steps without a final answer)"
        self._commit(question, stopped)
        return stopped

    def _commit(self, question: str, answer: str) -> None:
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

    # -- tool dispatch ------------------------------------------------------

    def _dispatch(self, step: int, name: str, args: dict[str, Any]) -> Any:
        if name == "search_docs":
            return self._run_search(step, args)
        if name in ADAPTER_IMPLS:
            return self._run_adapter(step, name, args)
        self._log(_c(f"  [step {step}] unknown tool {name!r}", Colors.YELLOW))
        return {"error": f"unknown tool {name}"}

    def _run_search(self, step: int, args: dict[str, Any]) -> Any:
        query = args.get("query", "")
        docs = search_docs(query)
        # Accumulate unique docs for grounding downstream adapters / the answer.
        seen = {d["doc_id"] for d in self.retrieved}
        for d in docs:
            if d["doc_id"] not in seen:
                self.retrieved.append(d)
        labels = ", ".join(doc_label(d) for d in docs) or "(no matches)"
        self._log(
            _c(f"  [step {step}] TOOL  search_docs({query!r})", Colors.CYAN)
            + _c(f" → {labels}", Colors.DIM)
        )
        return docs

    def _run_adapter(self, step: int, name: str, args: dict[str, Any]) -> Any:
        impl = ADAPTER_IMPLS[name]
        # Adapter tools operate on the docs retrieved so far.
        out = impl(self.client, args, self.retrieved)
        self._log(
            _c(f"  [step {step}] ADAPTER  {name}", Colors.MAGENTA)
            + _c(f"  (adapter_name={name!r})", Colors.DIM)
            + _c(f" → {out!r}", Colors.DIM)
        )
        return out

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


def _repl(agent: Agent) -> None:
    print(_c("Granite Switch ReAct agent", Colors.BOLD))
    print(_c("Type a question. /reset clears history, /exit quits.\n", Colors.DIM))
    while True:
        try:
            question = input(_c("you> ", Colors.BOLD + Colors.YELLOW)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in ("/exit", "/quit"):
            break
        if question == "/reset":
            agent.reset()
            print(_c("(history cleared)\n", Colors.DIM))
            continue
        answer = agent.run(question)
        print(f"\n{answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        help="ask one question and exit; omit for an interactive conversation",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide the trace")
    args = parser.parse_args()

    with LlamaClient(args.url) as client:
        if not client.health():
            raise SystemExit(
                f"No llama-server at {args.url}. Start it with ./run.sh first."
            )
        agent = Agent(client, verbose=not args.quiet)
        if args.question:
            print(_c(f"Q: {args.question}\n", Colors.BOLD))
            print(f"\n{agent.run(args.question)}")
        else:
            _repl(agent)


if __name__ == "__main__":
    main()
