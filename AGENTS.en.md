# AGENTS.md — dlqq-llm-train engineering conventions

> For AI assistants (DeepSeek Harness / Qwen Code / Claude Code / Cursor …): hard rules,
> directory layout, startup chain, API contract and runtime behaviour.
> User guide: [README.md](README.md) (Chinese) / [README.en.md](README.en.md).
> This file is the English twin of [AGENTS.md](AGENTS.md); **change both together**.
>
> Companion repositories:
> - `dlqq-llm-model`: publishes standard HF model directories (config.json + model.safetensors + chars.json), no scripts;
> - `dlqq-llm-dataset`: private corpus backup, unrelated to the code path.
>
> For what the local AIRES folder is for, its layout and usage rules, see AIRES.md; this file does
> not describe them.

## 1. Hard rules

1. **Fixed stack**: Python 3.11+ / PyTorch 2.x / transformers. The backend may use `torch`, `numpy`
   and `transformers`; everything else is stdlib. The frontend may use these local libraries under
   `web/vendor/`: `xterm.js` (MIT), `marked` (MIT), `DOMPurify` (Apache-2.0 / MPL-2.0), and
   `KaTeX` (MIT code, OFL-1.1 fonts). **No other frontend frameworks and no CDN** (must work offline).
2. **Use only the algorithm/framework — never any existing model weights**: this is a character-level
   model; the vocabulary is built by `code/train.py` itself. The network uses
   `transformers.GPT2LMHeadModel`, but it must start from random initialisation via
   `GPT2Config(...)`. **Forbidden**:
   - `from_pretrained("gpt2")` / `AutoModel...from_pretrained(third-party)` for external weights;
   - loading any third-party pretrained `.bin` / `.safetensors` / `.pt` weights;
   - HuggingFace tokenizers, pretrained vocabularies or embeddings from an existing model;
   - using any existing model's weights as initialisation, a fine-tuning start, or a "base".

   **Allowed**: `from_pretrained(model\<name>\)` to load a standard HF directory that **this project
   just trained itself**. It only reads our own output; it must never be used as a pretrained
   starting point, and must never load a model outside `model\`.

   **Exception (inference proxy only)**: `ollama_client.py` may forward HTTP requests to a
   user-managed Ollama instance. It never reads, stores, or trains third-party weights and is not
   part of the training pipeline.
3. **Three-tier git permissions; never widen them**: write operations (`commit` / `push` / `add` /
   `reset` / `checkout`, etc.) are **absolutely forbidden even if the user authorizes them**; the
   `.git` directory must **never be read or written, even if the user authorizes it** — no
   standard editing tool (read/write/edit) and no script or command may touch any file under
   `.git` (reading is forbidden too — a read command can be mistyped into a write/delete).
   **Single exception**: when the user asks for a commit message, run one read-only git command to
   read the diff first (see rule 20); otherwise run no git commands at all, including
   `status` / `diff` / `log`.
4. **No scripted bulk rewrites of source files**: edit file by file, confirm each change;
   for large refactors, list the plan first.
5. **AI must not run side-effecting actions**: training, corpus download/cleaning, installing,
   deleting, unpacking, bulk cleanup, starting the WebUI or the bridge server, or building the
   runtime. Read-only checks, syntax/static checks and doc edits are fine.
6. **`.bat` files must stay pure ASCII**; `.py` / `.properties` / `.md` / `.html` / `.js` / `.css`
   are UTF-8 without BOM; text files use CRLF on Windows.
7. **Fixed entry/build script locations**: keep them under `.runtime/`; do not scatter
   `.bat` / `.ps1` / `.py` files into the project root or other directories.
8. **No hard-coded local paths** in docs or comments; examples must stay generic.
9. **Weights and corpora stay out of the repo**: `model/*` and `res/*` are ignored. Weights are
   published in `dlqq-llm-model` as `model\<name>\` standard HF directories plus weight-file
   `.sha256` signatures; corpora are backed up in `dlqq-llm-dataset`.
10. **README / AGENTS must not describe local scratch folders in detail**: the rules are not
    expanded here; see the note at the top.
11. **Look before you create**: before adding any file or directory, list the current directory
    first (confirm the path, the naming and whether it already exists); never invent a path or
    create a folder blindly.
12. **Answer the question, don't act**: when the user is only asking, confirming, consulting or
    discussing something, **answer only** — do not change code, files or system settings, do not
    run scripts or commands, do not start processes, and do not run tests. State the conclusion
    from the information already available; act only when the user explicitly asks to change,
    fix or run something. If unsure whether it is "just a question", treat it as one.
13. **Match the user's language**: think and reply in the language the user asks in; default to
    Simplified Chinese when no language is specified.
14. **Keep Chinese and English docs in sync**: when changing `AGENTS.md` / `AGENTS.en.md` or
    `README.md` / `README.en.md`, update both language versions together — never just one.
15. **Keep README in sync with code changes**: whenever code, config keys, defaults, UI behaviour
    or startup behaviour changes, update `README.md` / `README.en.md` in the same change; never
    leave code and docs describing different behaviour.
16. **Write files only with the standard edit tools**: create or modify text, code and docs with
    the built-in `write` / `edit` tools; never use pwsh/cmd, PowerShell (`Set-Content`, `>`
    redirection, `[IO.File]::WriteAllText`) or MCP commands as a substitute. Files produced by
    build or render programs (images, artifacts) are exempt.
17. **`.bat` / `.ps1` must pause and support nopause**: pause before a normal exit and on failure so
    a double-click shows the error; `.bat` recognises `nopause`, `-nopause` and `/nopause`,
    `.ps1` recognises `-NoPause` (all case-insensitive). AI calls must pass it so the command never
    blocks until timeout.
18. **Test through the real entry point**: when checking a `.bat` / `.ps1`, run the entry script as
    a user would; never run its inner `.py` directly to bypass the entry. Add `nopause` for
    automation.
19. **Check after code changes**: after editing code, at least run syntax checks
    (`py_compile` / `ast.parse`) and config checks (`--check-config`); use build/test scripts when
    present, but never run side effects such as training or starting the WebUI.
20. **Read the git diff before writing a commit message**: when the user asks for a commit message,
    never write it from memory, an earlier conclusion, or an old message. First run one read-only
    git command (recommended combination: `git status --short --untracked-files=all`,
    `git diff --stat`, `git diff --name-status`; read-only `git diff -- <file>` when needed) to
    confirm the actual changes, then write the message from that diff. Write the message to
    `.ai_res/git_log/<YYYY-MM-DD_HH-MM-SS>.txt`. Before naming the file, query the actual current
    time and name it from that result; never guess or estimate the filename. The user performs the
    commit and AI never runs `git add`, `commit`, or `push`; describe only this repository's changes,
    and do not put content unrelated to this repository or ignored by git into the commit message.

## 2. Layout (current architecture)

```
Everything except the .runtime\ launcher and bundled interpreter lives under code\ and web\.
.runtime/               local runtime and launcher (python/ is gitignored)
  start.bat             double-click entry: always .runtime\python\python.exe → runs code\webui.py
  make_python_env.bat   rebuilds .runtime\python; prompts for a system Python path
  make_python_env.py
  logo.ico              shortcut icon
  python/               project Python: stdlib + ensurepip (build output, not in git)
code/                   all Python
  webui.py              local web service: static files + JSON API + SSE + child processes
  env_install.py        install training / chat deps into the project Python
  train.py              training (data pipeline + model + loop + sampling)
  chat.py               chat core: ChatSession (load / reply_stream / set_params)
  chat_worker.py        chat child: keeps ChatSession alive, JSON lines over stdin/stdout
  http_server.py        OpenAI-compatible bridge (ThreadingHTTPServer, lazy multi-model)
  ollama_client.py      Ollama HTTP adapter (/api/tags, /v1/chat/completions, SSE)
  mcp_client.py         MCP client (STDIO / Streamable HTTP, tool schemas, tools/call)
  common.py             find_root / order-preserving properties I/O / paths / watchdog / hide dir
  model.py              GPT-2 wrapper (transformers)
  model_signature.py    weight SHA-256 signatures (compute / verify, stdlib only)
web/                    frontend, served by webui.py (no separate web server)
  index.html            nine tabs: train / chat / Agent / server / environment / config / signature / MCP / credits
  chat.html             LAN read-only chat page (served by LANHandler)
  vendor/               local third-party assets: xterm.js / marked / DOMPurify / KaTeX
properties/             four training configs (open-sourced, comments written for outsiders)
model/                  local weights (`model\<name>\` standard HF directory + `checkpoint-<step>\`)
res/                    corpora (not in git)
.cache/                 training cache/logs/config backups/MCP configs and status (contents ignored; hidden on Windows)
THIRD-PARTY-NOTICES.txt  license inventory for Python / dependencies / frontend libraries (manual)
```

- The project root is found by walking up to the directory containing `code\train.py`
  (`common.find_root`), so moving scripts around does not break anything.
- Model id = directory name (`model\chat_demo\` → `chat_demo`).
- Resume directory `model\<name>\checkpoint-<step>\`: `optimizer.pt` / `scheduler.pt` /
  `trainer_state.json` / `rng_state.pth`; the root holds only the best model.

## 3. Startup chain and process model

```
.runtime\start.bat
  └─ .runtime\python\python.exe code\webui.py    ← foreground; stdlib + ensurepip, never imports torch
       ├─ child .runtime\python\python.exe code\train.py --config <cfg> --parent-pid <webui_pid>
       ├─ child .runtime\python\python.exe code\chat_worker.py --config <cfg> --parent-pid <webui_pid>
       ├─ child .runtime\python\python.exe code\http_server.py --port <port> --events --parent-pid <webui_pid>
       ├─ child .runtime\python\python.exe code\env_install.py ... (Environment-tab install)
       └─ child .runtime\python\python.exe code\model_signature.py --mode compute|verify --parent-pid <webui_pid>
```

- **No user-facing CLI**: docs must not advertise command-line usage. `--config` / `--parent-pid` /
  `--events` are an **internal protocol** between the WebUI and its children; no compatibility promise.
- **Single interpreter**: the WebUI, training, chat, bridge, signature and environment installer all
  use `.runtime\python\python.exe`; `common.resolve_python()` no longer reads `properties\python=`
  and never falls back to PATH. If missing, it tells the user to run `make_python_env.bat`.
- **Dependency install**: the bundled Python ships only the standard library + `ensurepip`; the
  Environment tab runs `code\env_install.py` to install training / chat dependencies into the same
  project Python. Users are not told to install a separate system Python. `torch_type` may be
  The form has three rows: PyTorch / Numpy / transformers. The PyTorch chip-type dropdown lists the
  actually supported hardware backends (CPU / GPU NVIDIA / AMD / Apple / XPU); the driver-version
  input takes a numeric version (CUDA 12.6 / ROCm 6.2), and the Torch library version is separate.
  `cu126` and similar tags are internal mappings only. Detect environment handles the chip type/driver
  version and reads installed versions into the form; installation skips already-installed
  libraries with the same version. The WebUI checks Python and Ollama once at startup. Training,
  chat, and server start do not auto-detect again; only the manual Detect environment button in the
  Environment tab refreshes them. The Ollama URL is editable and persisted.
- **Lifecycle**: closing the cmd window of `start.bat` means exit — children are `terminate()`d,
  and `taskkill /F /T` is the fallback after 5 seconds. **Closing the browser tab is not an exit**;
  training keeps running and you can simply reopen the URL.
- **Watchdog**: children poll `--parent-pid` every 3 s via `ctypes` `WaitForSingleObject` on the
  parent handle and call `os._exit(1)` when the parent is gone (protects against a hard-killed webui).
- **Job Object**: children also join a `KILL_ON_JOB_CLOSE` job, so when the webui process dies the
  kernel kills every process in the job — even if the cmd window is force-closed mid-run.
- **Ports**: the console defaults to `127.0.0.1:49134`; if busy it does **not** switch ports
  silently — it asks for a new port, and an empty Enter exits (`--port` behaves the same way).
  The LAN read-only chat page defaults to `0.0.0.0:49135` (`--lan-port` changes it, `--no-lan`
  disables it; if busy it only warns and continues).
- **Security**: the console binds `127.0.0.1` only, generates a one-shot token per start and checks
  it on every `/api/*` call. The LAN port has no token but only serves the chat page,
  `/api/models` and `POST /api/chat`.

## 4. WebUI API contract (changing it requires updating `web/index.html` too)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/env` | inspect/check the project Python (`check=1` probes; returns torch/numpy/transformers/pip/CUDA plus `system.torch_recommend` GPU/driver recommendation) |
| GET | `/api/env/latest` | query PyPI / PyTorch for the latest torch, numpy and transformers versions plus the actually available PyTorch build list; falls back to installed/recommended versions when the network is down |
| GET | `/api/env/ollama` | detect the Ollama URL, version and model counts (chat / embedding) |
| GET | `/api/configs` | config list with parsed key/values |
| GET | `/api/models` | standard model directories under `model\` (with signed flag) |
| GET | `/api/chat/configs` | chat dropdown: local configs + chat-only Ollama models (with backend tag) |
| GET | `/api/service/configs` | service-page dropdown: local configs + all Ollama models (chat / embedding / other) |
| GET | `/api/mcp/configs` | MCP server definitions (STDIO / HTTP) + last detection result |
| POST | `/api/mcp/configs` | save MCP server definitions to `.cache/mcp_servers.json` |
| POST | `/api/mcp/test` | connect to an MCP server and return `tools/list` |
| POST | `/api/mcp/detect` | detect all MCP servers at startup/manually; result saved to `.cache/mcp_status.json` |
| GET | `/api/agent/configs` | tool-capable models, MCP servers and detection results for Agent |
| GET | `/api/agent/tools?server=<id>` | list a server's tools for Agent multi-select |
| POST | `/api/agent/chat` | Agent tool loop; SSE: `meta` / `reasoning` / `tool_pending` / `tool` / `tool_result` / `token` / `done` / `error` |
| GET | `/api/corpora` | corpus directories under `res\` (file count, size) |
| GET | `/api/train/status` `train/log` | training state and incremental log |
| GET | `/api/server/status` | bridge state and port |
| GET | `/api/events` | SSE: `train-raw` / `server-log` / `server-chat` / `server-embedding` / `env-log`; `train-raw` keeps `\r` for xterm |
| POST | `/api/config/save` | line-level value update (keeps comments/order), backs up first |
| POST | `/api/env/install` | install torch / numpy / transformers into the project Python; `torch_type=cpu|gpu|xpu|auto`; `torch_version` / `numpy_version` / `transformers_version` may be empty; legacy `torch_index=cpu|cu126|cu128|xpu|auto` remains supported; output via `env-log` |
| POST | `/api/train/start` `train/stop` | start / stop the training child |
| POST | `/api/chat` | chat, returns SSE: `meta` / `reasoning` / `token` / `done` / `error` |
| POST | `/api/server/start` `server/stop` | start / stop the bridge child |
| POST | `/api/sign` | `mode=compute|verify`, runs the signature tool |

- The bridge prints one `@@EV@@ {json}` line per completed chat (`--events`); the webui reader
  turns it into an SSE `server-chat` event, which the "server" tab renders as a Q&A list.
  Successful embeddings become `server-embedding` and are summarized in the Vectors tab.
- SSE frames are `data: <json>\n\n`; one thread per connection, so new long-lived endpoints
  must consider the thread budget.
- **LAN port** (`LANHandler`) only allows `GET /`, `/chat.html`, `/logo.ico`, `/vendor/*`,
  `/api/models`, `/api/chat/configs` and `POST /api/chat`; everything else returns 403.
  `POST /api/chat` only accepts a `config` that appears in the config list. Changing LAN endpoints
  means updating `web/chat.html` too.

## 5. Config (properties) conventions

- **Syntax**: one `key=value` per line; `#` starts a whole-line comment.
- **No trailing comments**: the parser uses `split("=", 1)`, so `key=1  # note` breaks the value.
- **Keep the four configs isomorphic**: `chat_demo` / `dlqq_exam` / `dlqq_llm` / `dlqq_poem`
  must share the same key set, order and sections.
- **Only keys that exist in `CONFIG`**: `code/train.py`'s `CONFIG` is the single source of truth.
- **Saving rules**: always go through `common.save_properties` (value-only edits, comments and
  order preserved) and back up per the scratch-folder note at the top.
- **No `python=` key**: the interpreter is fixed to `.runtime\python\python.exe`; a legacy
  `python=` is ignored and never resolved.
- **Resume is a UI action**: the "Resume training" button sends `resume=true`; the config files no
  longer carry a `resume` key.

| Key | Meaning |
|---|---|
| `data_path` | corpus **directory** (relative to the project root) |
| `model` | model name → standard HF directory `model\<name>\` |
| `cache_dir` | tokenisation cache dir (`.cache` by default) |
| `chat_mode` / `chat_temperature` / `chat_top_k` / `chat_max_new` | chat-time settings |
| `n_layer` / `n_head` / `n_embd` / `block_size` / `batch_size` | architecture and tokens per step |
| `max_iters` / `eval_interval` / `eval_iters` | step count and eval cadence |
| `lr` / `warmup_iters` / `lr_decay` / `min_lr_frac` / `weight_decay` / `grad_clip` / `dropout` / `seed` | optimisation |
| `grad_accum` / `grad_checkpoint` | effective batch / memory saving |
| `save_every` / `keep_best_checkpoints` / `keep_forced_checkpoints` / `save_dtype` | periodic checkpoint / best checkpoint cap / forced checkpoint cap / dtype |
| `max_vocab` / `val_frac` | vocabulary cap / tail validation split |

## 6. Runtime behaviour (understand before editing)

**Data pipeline**: read in 4M-char chunks → `np.bincount` frequency count → keep top `max_vocab`
chars → `searchsorted` encode to uint16 cache → train via `np.memmap` random reads. Memory does not
grow with corpus size. Cache fingerprint = source absolute path + per-file (rel path, size, mtime,
format, Q/A fields) + `max_vocab` + `val_frac`; **changing any of these invalidates the cache**
(≈5 minutes to re-tokenise 15 GB).

**Validation** is split from the **tail** of the corpus; shuffle category-sorted corpora first.

**Context auto-shrink**: `_safe_T = min(train tokens, val tokens) - 2`; a larger `block_size`
is reduced automatically with a warning.

**Checkpoint policy**: when validation loss improves, update the best model in
`model\<name>\` and write a `checkpoint-<step>\`; `save_every > 0` writes extra checkpoints.
Checkpoints always keep `optimizer.pt` / `scheduler.pt` / `rng_state.pth` for resuming;
`keep_best_checkpoints` / `keep_forced_checkpoints` independently cap the latest N best/forced
checkpoints (0 = keep none, -1 = unlimited). The root directory
always contains only `config.json` / `model.safetensors` / `chars.json` / `meta.json`, never an
optimizer, so it can be published directly.

**Chat parameter precedence**: explicit args > `properties\<config>.properties` > values stored in
the model directory (`meta.json`) > defaults (used by both the web chat and the bridge; implemented
twice in `chat.py` and `http_server.py` — keep them in sync).

**Ollama proxy**: `ollama_client.py` discovers models from `OLLAMA_URL` (default
`http://127.0.0.1:11434`) and prefixes their ids with `ollama/`. The bridge and the WebUI chat page
only forward requests to `/v1/chat/completions`; GGUF is never loaded and training is unaffected.
Reasoning is extracted from `message.thinking` / `delta.reasoning_content` and used for display only,
never added to history. Local HTTP `stream=true` now generates and sends character by character; the
Ollama proxy passes the upstream SSE stream through. The Ollama chat list filters embedding and other
non-chat models using `/api/show` capabilities; `/v1/embeddings` calls Ollama's native `/api/embed`
and converts the result to the OpenAI response shape.

**MCP / Agent**: `mcp_client.py` uses only the standard library and supports STDIO (one JSON per
line) and Streamable HTTP (JSON-RPC POST + Mcp-Session-Id). Configured MCP clients are created at
WebUI startup and stay resident; a STDIO process lives as long as the WebUI, is restarted only when
its config changes, and is closed when the WebUI exits. Agent accepts only Ollama models with the
`tools` capability and loops over `tool_calls → tools/call → tool result` for at most 8 rounds;
each round uses Ollama SSE, the final answer streams token by token; a turn header shows status and
elapsed time, and clicking it collapses only reasoning/tool-call rows while answers stay visible;
reasoning, tool calls and answers appear as flat timeline rows in the model's actual output order.
Agent requests may include a `tools` name list for multi-select filtering;
`thinking=true` forwards `think` / `reasoning_effort` and shows the reasoning, while local
character-level models do not support that toggle. After tool results, an internal continuation user
turn is added so Ollama/Qwen does not reject a pure tool turn with
"no user query found in messages". Server definitions live in `.cache/mcp_servers.json`; all
servers are detected at startup and the result is written to `.cache/mcp_status.json`.

**Stopping**: prompts use `用户：` / `助手：`; generation stops at the next `用户：`. If the vocabulary
lacks those markers, the session falls back to plain continuation mode.

**Chatting while training**: only a **different** model is allowed; the model named by the running
training config is rejected (it would slow training and may read a half-written file).

## 7. Code style

- **File header**: `# -*- coding: utf-8 -*-` plus an `r"""..."""` block (purpose, usage, caveats).
- **Encoding guard**: call `common.setup_stdio()` (or reconfigure stdout/stderr to UTF-8 with
  `errors="replace"`) in every entry point.
- **Comments in Chinese**, explaining *why*, not restating the code; record the measurements that
  justify defaults.
- **Fail fast**: invalid args/env/data → `sys.exit(1)` with an actionable hint; no guessing fallbacks.
- **Never swallow exceptions**: catch concrete types; print or re-raise.
- **Print format**: `[tag] message` with `[数据]` / `[训练]` / `[设备]` / `[错误]` / `[警告]`;
  use `flush=True` inside training loops.
- **Naming**: `snake_case` functions/variables, `UPPER_SNAKE_CASE` constants, `CamelCase` classes;
  config keys match `CONFIG`.
- **No type annotations**: stay consistent with the existing code.
- **No logging framework**: `print(..., flush=True)` or the webui log queue.
- **Single model definition**: `code/model.py` wraps `transformers.GPT2LMHeadModel`; `train.py` /
  `chat.py` / `http_server.py` all import it and no longer keep inline copies. Training must start
  from random initialisation; loading any third-party pretrained weights is forbidden.

## 8. Known limitations

- **Old weights are incompatible**: after switching to GPT-2, the old hand-written GPT `.pt`
  files use different parameter names and cannot be loaded; retrain on this project's corpus.
- **Character level + short context**: with `block_size` 256 only the tail of a long message is seen.
- **Small models memorise**: they replay training answers; do not benchmark on the training set.
- **The bridge is not an agent**: no tool calling, no planning; it only validates the pipeline.
- **Too many Agent tools**: selecting 100+ MCP tools consumes a lot of context and makes the model
  more likely to invent a tool name or emit a truncated tool_call; for large files, select only the
  relevant file tools and let the model write in chunks.
- **The WebUI console is local-only**: it binds `127.0.0.1` with a one-shot token; the LAN side
  exposes only the read-only chat page (`0.0.0.0:49135`, no auth — anyone on the LAN can chat).
- **CPU training** is very slow and has no bf16; use it only to smoke-test tiny models.

## 9. References

- Weights: [ModelScope · donglouqiaoque/dlqq-llm-model](https://www.modelscope.cn/donglouqiaoque/dlqq-llm-model)
- Source mirrors: [GitHub](https://github.com/donglouqiaoque/dlqq-llm) ｜ [Gitee](https://gitee.com/donglouqiaoque/dlqq-llm)
- PyTorch docs: <https://pytorch.org/docs/stable/index.html>
- nanoGPT: <https://github.com/karpathy/nanoGPT>
