# Granite Switch ReAct Agent

A minimal ReAct agent whose brain is **[Granite Switch](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview)** running locally on **llama.cpp**. The base model does the reasoning and picks tools; several of the model's own **embedded adapter functions** are exposed as LLM-selectable tools and executed by re-invoking the model with that adapter activated.

## The idea

Granite Switch packs 12 LoRA "adapter functions" into one checkpoint, each activated by a control token. This demo answers the question *"can that model be the brain of a ReAct agent, and can its adapters be the agent's tools?"* — using nothing but `llama-server` and ~350 lines of Python.

Two kinds of tool, dispatched differently:

| Tool | Kind | How the harness runs it |
|------|------|-------------------------|
| `search_docs` | external | plain Python over an in-memory corpus |
| `query_rewrite` | adapter | re-invoke the model with `adapter_name="query_rewrite"` |
| `answerability` | adapter | re-invoke with `adapter_name="answerability"` + `documents` |

The base model *selects* an adapter tool like any other; the harness performs the mechanically-correct activation (an adapter is a per-request property, not something the model emits inline) and feeds the adapter's structured output back as the observation.

### How adapter activation works

llama.cpp's server forwards `chat_template_kwargs` into the model's Jinja chat template. The template maps an `adapter_name` to that adapter's control token and places it in the prompt:

```python
client.chat(messages, adapter_name="answerability", documents=docs)
# → POST /v1/chat/completions  { ..., "chat_template_kwargs": {"adapter_name": "answerability", "documents": [...]} }
```

**Why only these two adapters?** They are *envelope-free* — their trained input is just messages + documents, which the chat template already produces, and they emit tight, well-formed output over the raw template. The other adapters (`citations`, `guardian-core`, and the judge family) need a specific trained input envelope (the `<guardian>` criteria protocol, `<c0>/<r0>` sentence tagging) that llama.cpp's template does **not** build. Adding those means pulling in [Mellea](https://mellea.ai)'s `IntrinsicsRewriter` to construct the envelope — intentionally out of scope here to keep the demo dependency-free.

> `query_clarification` was tried and dropped: its rendered template is identical to the two above (so token placement is correct), but on vague inputs it degrades — leaking a role token mid-generation and never reliably emitting the clarifying question that is its whole purpose. Its open-ended output appears more sensitive to the exact trained input framing than the two tight-output adapters, so it doesn't work cleanly over the raw template.

## Prerequisites

1. **llama.cpp** built from the `feature/granite-switch` branch (HEAD using the `graniteswitch.adapters.*` metadata namespace):
   ```bash
   cmake --build ~/Desktop/IBM/projects/llama.cpp/build --target llama-server -j
   ```
2. **A GGUF** converted with that same branch's converter (already present here as `granite-switch-4.1-3b.f16.gguf`, 8.4 GB). To regenerate:
   ```bash
   cd ~/Desktop/IBM/projects/llama.cpp
   .venv-convert/bin/python convert_hf_to_gguf.py \
     <hf-snapshot-of ibm-granite/granite-switch-4.1-3b-preview> \
     --outfile granite-switch-4.1-3b.f16.gguf --outtype f16
   ```
   > Note: the binary and the GGUF must agree on metadata keys. The older `graniteswitch.num_adapters` keys won't load on current HEAD, which expects `graniteswitch.adapters.*`.

## Run

```bash
uv sync

# terminal 1 — boot the model server
./run.sh

# terminal 2 — ask the agent
uv run agent.py "What is the capital of France, and what river runs through it?"
uv run agent.py "Who created Python?"
```

Environment overrides for `run.sh`: `LLAMA_DIR`, `LLAMA_SERVER`, `GS_MODEL`, `GS_PORT`.

## Files

- `llama_client.py` — thin OpenAI-compatible client with the `adapter_name` passthrough.
- `tools.py` — the corpus + `search_docs`, and the three adapter-backed tools with their schemas.
- `agent.py` — the ReAct loop and tool dispatch (color-traced for demoing).
- `run.sh` — boots `llama-server` on the GGUF.
