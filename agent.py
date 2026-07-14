# SPDX-License-Identifier: Apache-2.0
"""A ReAct agent whose brain is Granite Switch running on llama.cpp.

The base model does the reasoning and decides which tools to call. Two tool
kinds are dispatched differently by the harness:

  * ``search_docs`` — an ordinary external tool (retrieval).
  * ``query_rewrite`` / ``answerability`` — *adapter* tools. The model chooses
    them like any tool, but the harness executes each by
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
    "You are a careful research assistant that answers ONLY from documents you "
    "retrieve this turn, using the tools below. You have no reliable knowledge of "
    "your own; treat anything not in the retrieved documents as unknown.\n"
    "\n"
    "For greetings, small talk, or questions about your own capabilities, reply "
    "directly in one sentence WITHOUT calling any tools.\n"
    "\n"
    "For every factual question, follow these MANDATORY rules in order:\n"
    "1. query_rewrite — if the question refers back to an earlier turn (it, that, "
    "he, she, they, 'the capital', a bare 'how tall'), you MUST call query_rewrite "
    "FIRST to turn it into a standalone question. Pass the prior turns as history.\n"
    "2. search_docs — you MUST call search_docs every factual turn, including "
    "follow-ups. NEVER answer a follow-up from earlier turns, the conversation, or "
    "your own memory; always re-retrieve. Search with the standalone question.\n"
    "3. answerability — after search_docs you MUST call answerability before you "
    "answer. Do not write an answer until you have.\n"
    "4. If answerability is 'answerable', give a concise answer grounded ONLY in "
    "the retrieved text — do not add facts, figures, or unit conversions that are "
    "not literally in a document. If 'unanswerable', say the documents do not "
    "contain the answer, and do not guess.\n"
    "\n"
    "Always call a tool by emitting a real tool call, never by writing its name as "
    "text."
)

MAX_STEPS = 8

# Tool names, to catch a model that writes a name instead of calling the tool.
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}

# Adapters that need retrieved docs; called with none they only ever say 'unanswerable'.
DOC_DEPENDENT_ADAPTERS = {"answerability"}


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
        self.current_question = question  # updated if query_rewrite decontextualizes it
        shown = 0  # displayed step number; skips silent nudge iterations

        for _ in range(MAX_STEPS):
            result = self.client.chat(messages, tools=TOOL_SCHEMAS, max_tokens=512)

            if not result.tool_calls:
                answer = result.content.strip()
                # Model wrote a tool name as text instead of calling it: nudge, don't return it.
                if answer in TOOL_NAMES:
                    messages.append({"role": "assistant", "content": answer})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"To use the {answer} tool you must emit a real "
                                "tool call, not its name as text. Do this now. If "
                                "the retrieved documents do not contain the answer, "
                                "say so plainly — do not answer from prior knowledge."
                            ),
                        }
                    )
                    continue
                # No tool call → the model is answering.
                self._log(f"\n{_c('ANSWER', Colors.BOLD + Colors.GREEN)}")
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

            shown += 1
            for call in result.tool_calls:
                name = call.get("function", {}).get("name", "")
                args = parse_json_arguments(call)
                observation = self._dispatch(shown, name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "content": json.dumps(observation, default=str),
                    }
                )
                # query_rewrite gives a standalone question; use it for later checks.
                if name == "query_rewrite" and isinstance(observation, str):
                    self.current_question = observation
                # Always verify retrieval: chain answerability after every search.
                if name == "search_docs":
                    shown += 1
                    messages.append(self._auto_answerability(shown, call))

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

    def _run_adapter(
        self, step: int, name: str, args: dict[str, Any], *, auto: bool = False
    ) -> Any:
        if name in DOC_DEPENDENT_ADAPTERS and not self.retrieved:
            self._log(
                _c(f"  [step {step}] SKIP  {name} (no documents yet)", Colors.YELLOW)
            )
            return {"note": f"No documents retrieved yet. Call search_docs before {name}."}
        impl = ADAPTER_IMPLS[name]
        out = impl(self.client, args, self.retrieved)
        tag = "ADAPTER*" if auto else "ADAPTER"
        self._log(
            _c(f"  [step {step}] {tag}  {name}", Colors.MAGENTA)
            + _c(f"  (adapter_name={name!r})", Colors.DIM)
            + _c(f" → {out!r}", Colors.DIM)
        )
        return out

    def _auto_answerability(self, step: int, search_call: dict[str, Any]) -> dict[str, Any]:
        """Run answerability on the current question right after a search."""
        verdict = self._run_adapter(
            step, "answerability", {"question": self.current_question}, auto=True
        )
        return {
            "role": "tool",
            "tool_call_id": f"auto_answerability_{search_call.get('id', step)}",
            "content": json.dumps({"answerability": verdict}, default=str),
        }

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
