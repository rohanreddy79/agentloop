# agentloop

**A minimal, fully tested LLM agent runtime: typed tools, a budgeted tool-calling loop with parallel execution, structured output with validation-retry, and JSONL traces, in ~1,400 lines of core Python you can read in an afternoon.**

[![ci](https://github.com/rohanreddy79/agentloop/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanreddy79/agentloop/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why

Agent frameworks hide a small core under a lot of surface area: a provider adapter, a tool
registry with schema generation, a loop with a stopping rule, memory, and observability.
Everything else (planners, RAG, multi-agent) is built on those five things. This repo
implements exactly those five, with the decisions that matter in production made explicit:
tool failures go back to the model instead of crashing the loop, budgets are hard limits that
still hand you the partial result, structured output is validated and retried inside the same
loop, and every run leaves a trace with latency, usage and cost. A deterministic fake provider
makes all of it runnable and testable offline; the Anthropic provider plugs into the real
Messages API.

## Features

- `@tool` derives name, description and a JSON schema from a function's type hints and
  docstring (`str/int/float/bool`, `list[T]`, `Optional`, `Literal`, nested pydantic models);
  sync and async functions.
- `ToolRegistry` validates arguments, enforces per-tool timeouts, and turns every failure
  (unknown tool, bad arguments, timeout, exception) into an `is_error` tool result.
- `Agent.run()` / `arun()`: call model, run all tool calls of a turn concurrently with
  `asyncio.gather`, append results, repeat until a text-only answer.
- `Budget(max_steps, max_tokens, max_cost_usd, max_seconds)`; exceeding one raises
  `BudgetExceeded` carrying the partial `RunResult`.
- `agent.run(task, output_model=MyModel)` registers a hidden `final_answer` tool with the
  model's schema; validation errors are returned to the model and retried up to N times.
- `Memory` with an optional sliding window that never splits a tool call from its result, plus
  a `summarize` hook for what gets dropped.
- JSONL tracing (`run_start`, `step`, `tool_call`, `run_end`) with per-call latency, token
  usage and USD cost from a dated price table; `agentloop trace show` renders it.
- Providers: `AnthropicProvider` (official SDK, parallel `tool_use` blocks, `is_error`
  results, cache usage counters), `FakeProvider` (scripted), `RuleBasedFakeProvider`
  (deterministic planner that solves the demo offline).
- Built-in tools: AST-safe `calculator`; `read_file`/`write_file`/`list_dir` sandboxed to a
  root (traversal and symlink escapes rejected); `http_get` with host allowlist, size cap and
  no redirects; opt-in `python_exec` in an isolated subprocess with a timeout.
- Typer CLI: `agentloop run "<task>" --provider fake|anthropic --tools calc,fs --root ./sandbox`.

## Quickstart

```bash
git clone https://github.com/rohanreddy79/agentloop && cd agentloop
uv sync
uv run python examples/offline_demo.py
uv run agentloop trace show traces/offline_demo.jsonl
uv run agentloop run "What is 2 ** 16 - 1?" --tools calc --json
```

The demo sandbox contains `notes.md` but the task asks for `notes.txt`, so the first read
fails and the "model" has to recover. Output of the third command (no key, no network):

```text
task: Compute 17 * 23 + 5, read notes.txt, and write the result and the notes to result.txt.

step 1: calculator(expression='17 * 23 + 5'), read_file(path='notes.txt') -> ok, error
step 2: list_dir(path='.') -> ok
step 3: read_file(path='notes.md') -> ok
step 4: write_file(path='result.txt', content='17 * 23 + 5 = 396\n\n--- note...) -> ok
step 5: text answer

answer: Done: 17 * 23 + 5 = 396; read notes.md (58 chars); wrote result.txt; recovered from missing notes.txt by listing the directory and reading notes.md.
steps=5 tool_calls=5 tokens=1018 elapsed=0.004s
trace: traces/offline_demo.jsonl
```

(The token counts come from the fake provider's deterministic estimate, about one token per
four characters; the Anthropic provider reports the API's real usage.) The same loop from
Python, with a tool of your own:

```python
from agentloop import Agent, Budget, RuleBasedFakeProvider, tool
from agentloop.builtin import calculator, make_fs_tools


@tool
def word_count(text: str) -> int:
    """Count the words in a text.

    Args:
        text: The text to count.
    """
    return len(text.split())


agent = Agent(
    RuleBasedFakeProvider(),  # or AnthropicProvider(); reads ANTHROPIC_API_KEY
    [calculator, word_count, *make_fs_tools("./sandbox")],
    system="You are a careful assistant. Use the tools when they help.",
    budget=Budget(max_steps=10, max_cost_usd=0.25),
)
result = agent.run("Read notes.md and compute 12 * 4.")
print(result.output)  # Done: 12 * 4 = 48; read notes.md (58 chars).
print(result.steps, result.usage.total_tokens, f"${result.cost_usd:.4f}")
```

For a live model: copy `.env.example` to `.env`, export `ANTHROPIC_API_KEY`, then
`uv run agentloop run "..." --provider anthropic` or `uv run python examples/anthropic_research.py`.
The default model is `claude-opus-5-5` (override with `--model` or `AGENTLOOP_MODEL`). The live
path is not exercised by the tests or CI.

## How it works

| Module | Responsibility |
|---|---|
| `providers/base.py` | `Message`, `ToolCall`, `ToolResult`, `ModelResponse`, `Usage`, the `Provider` protocol |
| `providers/anthropic.py` | Messages API wire mapping (`tool_use`/`tool_result` blocks, usage, stop reason) |
| `providers/fake.py` | `FakeProvider` (replays a script, records requests) and `RuleBasedFakeProvider` |
| `schema.py`, `tools.py` | signature -> pydantic model -> JSON schema; `@tool`, `Tool`, `ToolRegistry` |
| `agent.py` | `Budget`, `Agent`, `RunResult`, `BudgetExceeded`, the loop, structured output |
| `memory.py` | message history, windowed provider view, `summarize` hook |
| `tracing.py`, `pricing.py` | JSONL `Tracer`, `read_trace`, rich renderer; dated USD price table |
| `builtin/` | `calculator`, `fs` (sandboxed), `http` (allowlisted), `python_exec` (opt-in) |
| `cli.py` | `agentloop run`, `agentloop trace show` |

One loop iteration:

```text
 Agent.arun(task)          Provider                ToolRegistry            Tracer
 ────────────────          ────────                ────────────            ──────
   memory.messages() ──complete(msgs, tools, sys)──>
                     <── ModelResponse(text, tool_calls[], usage, stop) ──
   usage += ; cost += price_table(model)                                    step {latency, usage,
   memory.add(assistant turn)                                                     cost, messages_delta}
   if tool_calls:
     asyncio.gather( ──execute(call_1)──> validate -> run (thread if sync) -> stringify
                     ──execute(call_2)──> ... any failure -> ToolResult(is_error=True)
                   ) <── results, in call order                               tool_call x N {latency,
     memory.add(user turn with all results)                                              is_error}
     budget check (tokens, cost) -> BudgetExceeded(partial RunResult)
     loop (steps / seconds checked before the next call)
   else:
     return RunResult(output, parsed, steps, usage, cost, messages, trace_path)   run_end
```

**What a framework actually is.** Strip the branding and an agent framework is a provider
adapter, a tool registry, a loop with a stopping rule, memory, and tracing. Keeping each in its
own small module (the table above) is the main design decision: the loop never sees the wire
format, the providers never see tools' Python functions, and the fake providers exercise every
line of the loop so the Anthropic adapter is the only untested-against-the-network piece, and
its request/response mapping is unit-tested against a stand-in client.

**Why tools return errors to the model instead of raising.** The model is the only component
that can decide what to do about a failed call: retry with different arguments, try another
tool, or explain. Raising would unwind the loop and discard the context it has built up.
`ToolRegistry.execute` therefore never raises; it returns `ToolResult(is_error=True)` with the
exception text, which maps directly onto the Messages API's `tool_result.is_error`. The demo
shows it working: `read_file` fails, the model lists the directory and recovers. Programmer
errors (a tool without type hints, duplicate names) still raise at definition time, because
the model cannot fix those.

**Why budgets are first class.** The loop is `while True` driven by a stochastic process.
`max_steps` is always on; tokens, cost and wall-clock are optional. Checks happen before each
provider call (steps, seconds) and after it (tokens, cost), and a limit never aborts a run that
has just finished. `BudgetExceeded.result` carries the partial `RunResult` so a caller can still
use what was produced. Cost comes from `pricing.py`, a table copied from the current model
list with its date in the file; models not in the table are flagged `price_known: false` in the
trace rather than silently priced at zero.

**Why structured output goes through a tool.** A hidden `final_answer` tool with the pydantic
model's schema gives the model a contract it can see, and a validation failure becomes an
ordinary error tool result, so retries reuse the loop rather than a second parsing path.
Current Claude models reject forced `tool_choice`, so the agent instructs the model to call
`final_answer` in the system prompt and nudges once more if it answers in plain text; after
`max_output_retries` it raises `StructuredOutputError`. `examples/structured_extraction.py`
shows a rejected call (`plan: Input should be 'free', 'pro' or 'enterprise'; interests: Input
should be a valid list`) followed by an accepted one.

**Concurrency model.** Everything is `asyncio`. All tool calls in one assistant turn run under
`asyncio.gather`; sync tools run in `asyncio.to_thread` so they overlap too; each call has a
timeout via `asyncio.wait_for`. Results are returned to the model in one user message, in call
order, which is what the API expects for parallel tool use. `Agent.run` is `asyncio.run(arun)`.

### Trace

Every run writes one JSON object per line. `uv run agentloop trace show traces/offline_demo.jsonl`
renders the demo above (a copy of this trace is committed at `examples/traces/offline_demo.jsonl`):

```text
╭────────────────────────────────────────── run 0863907d ──────────────────────────────────────────╮
│ task: Compute 17 * 23 + 5, read notes.txt, and write the result and the notes to result.txt.     │
│ provider: fake-rules (fake-rules)   status: ok   steps: 5   tokens: 1018   cost: $0.0000 (model  │
│ not in price table)   elapsed: 0.004s                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
┏━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━┓
┃ step ┃ event ┃ detail                                                                       ┃ ms ┃
┡━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━┩
│    1 │ model │ -> calculator(expression='17 * 23 + 5'), read_file(path='notes.txt')  (45    │  0 │
│      │       │ tok, tool_use)                                                               │    │
│    1 │ tool  │ calculator ok: 396                                                           │  2 │
│    1 │ tool  │ read_file error: FileNotFoundError: no such file: notes.txt                  │  0 │
│    2 │ model │ -> list_dir(path='.')  (133 tok, tool_use)                                   │  0 │
│    2 │ tool  │ list_dir ok: notes.md                                                        │  0 │
│    3 │ model │ -> read_file(path='notes.md')  (186 tok, tool_use)                           │  0 │
│    3 │ tool  │ read_file ok: Q3 planning notes\n- ship agentloop 0.1\n- write the README\n  │  0 │
│    4 │ model │ -> write_file(path='result.txt', content='17 * 23 + 5 = 396\n\n--- notes.md  │  0 │
│      │       │ ---\nQ3 plannin...)  (282 tok, tool_use)                                     │    │
│    4 │ tool  │ write_file ok: wrote 94 bytes to result.txt                                  │  0 │
│    5 │ model │ text: Done: 17 * 23 + 5 = 396; read notes.md (58 chars); wrote result.txt;   │  0 │
│      │       │ recovered from missing notes.txt by listing the directory and reading        │    │
│      │       │ notes.md.  (372 tok, end_turn)                                               │    │
└──────┴───────┴──────────────────────────────────────────────────────────────────────────────┴────┘
```

A raw `tool_call` line from the same file:

```json
{"event": "tool_call", "ts": 1792609628.744, "run_id": "0863907d", "step": 1, "id": "call_2", "name": "read_file", "arguments": {"path": "notes.txt"}, "result": "FileNotFoundError: no such file: notes.txt", "is_error": true, "latency_ms": 0.2}
```

## Results

Runtime overhead only (the fake provider answers instantly, so no model latency is included).
Command: `uv run python examples/bench_overhead.py`. Machine: Apple M3, 8 GB RAM, macOS 27.0,
Python 3.12.12, CPU only. Numbers from one run; five other runs in the same session gave
7,300-8,200 steps/s untraced, 5,900-6,600 traced, 610-750 us per tool build, and the two
sleep-based rows within 2 ms.

| measurement | result |
|---|---|
| loop steps/s: fake provider + 1 tool call per step, no tracing | 7,633 |
| loop steps/s: same, with JSONL tracing on | 6,640 |
| 8 async tools x 50 ms, concurrent (`registry.execute_all`) | 51 ms |
| 8 async tools x 50 ms, one at a time | 413 ms |
| build a 5-parameter tool (schema derivation), mean of 200 | 716 us |

So the loop adds roughly 0.13 ms per step, tracing about 0.02 ms more, and a turn with eight
parallel 50 ms tools costs one tool's latency, not eight.

## Testing

`uv run pytest -q` runs 153 tests in about 5 s with no network and no API key (also passes on
Python 3.11). Coverage by area:

- schema generation for 10+ signatures (defaults, `list[T]`, `Optional`, `Literal`, nested
  pydantic, no-arg, async, locally defined types) and the signature errors;
- registry: duplicate and unknown tools, argument validation, exceptions, async and sync
  timeouts, concurrent execution (wall time asserted below the sum of sleeps);
- loop: ends on text, feeds results back, continues after `is_error`, parallel calls, hooks,
  usage accumulation, memory across runs and windowing;
- budgets: `max_steps`, `max_tokens`, `max_cost_usd`, `max_seconds`, each with the partial
  result, and the "do not abort a finished run" rule;
- structured output: success, validation-retry, plain-text nudge, retry limit, mixed turns;
- `AnthropicProvider`: request and response mapping (parallel `tool_use`, `is_error`, usage,
  thinking blocks ignored, refusal pass-through) against a stand-in client, SDK error
  wrapping, and an end-to-end agent run through the mock;
- tracing: file written and parseable, cost and `price_known`, statuses, renderer;
- built-ins: calculator rejects 17 non-arithmetic inputs; path traversal and symlink escapes;
  `http_get` allowlist, size cap, redirects via `httpx.MockTransport`; `python_exec` timeout;
- CLI via `CliRunner`; the example scripts run as subprocesses.

`uv run ruff check .` is clean; CI runs both on pushes and pull requests.

## Roadmap / Limitations

- Sync tools that time out keep running in their worker thread; the loop moves on but the
  thread is not killed. Use async tools or `python_exec` for anything that must be stoppable.
- `python_exec` is a convenience, not a sandbox: it runs with the agent's OS permissions.
- The Anthropic provider is non-streaming and does not pass thinking blocks back; adding
  streaming and `thinking`/`effort` presets per model is the next provider feature.
- `RuleBasedFakeProvider` only understands "compute / read / write" tasks; it exists to make
  the demo and tests deterministic, not to stand in for a model.
- The price table is a snapshot (date in `pricing.py`); refresh it when models change.
- No retries at the loop level for provider errors (the SDK retries 429/5xx itself) and no
  prompt caching hints yet.

## License

MIT. Copyright (c) 2026 Rohan Reddy Jakkam.
