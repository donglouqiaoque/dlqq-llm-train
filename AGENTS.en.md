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
   and `transformers`; everything else is stdlib. The frontend is plain HTML/CSS/JS with one
   exception: local `web/vendor/xterm.js` (MIT) for the training-log terminal. **No other frontend
   framework and no CDN** (must work offline).
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
3. **Three-tier git permissions; never widen them**: write operations (`commit` / `push` / `add` /
   `reset` / `checkout`, etc.) are **absolutely forbidden even if the user authorizes them**; the
   `.git` directory must **never be read or written, even if the user authorizes it** — no
   standard editing tool (read/write/edit) and no script or command may touch any file under
   `.git` (reading is forbidden too — a read command can be mistyped into a write/delete); only
   read-only commands (`status` / `diff` / `log`, etc.) may be allowed once when the user
   explicitly asks, and must stop immediately afterwards — no further use unless the user asks
   again. There is no "the user allowed it implicitly" or "the user said it before" excuse.
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
  common.py             find_root / order-preserving properties I/O / paths / watchdog / hide dir
  model.py              GPT-2 wrapper (transformers)
  model_signature.py    weight SHA-256 signatures (compute / verify, stdlib only)
web/                    frontend, served by webui.py (no separate web server)
  index.html            six tabs: train / chat / server / environment / config / signature
  chat.html             LAN read-only chat page (served by LANHandler)
  vendor/               local third-party assets: xterm.js / xterm.css / LICENSE.xterm
properties/             four training configs (open-sourced, comments written for outsiders)
model/                  local weights (`model\<name>\` standard HF directory + `checkpoint-<step>\`)
res/                    corpora (not in git)
.cache/                 training cache/logs/config backups (directory tracked, contents ignored)
THIRD-PARTY-NOTICES.txt  license inventory for Python / dependencies / xterm (maintained manually)
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
  project Python. Users are not told to install a separate system Python. `torch_version` may be
  `2.14.1+cu126` / `2.14.1+cpu`; auto detection fills the recommended value into the text box and
  the user may override the version/source suffix.
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
| GET | `/api/configs` | config list with parsed key/values |
| GET | `/api/models` | standard model directories under `model\` (with signed flag) |
| GET | `/api/chat/configs` | configs that map to an existing weight (name + model) |
| GET | `/api/corpora` | corpus directories under `res\` (file count, size) |
| GET | `/api/train/status` `train/log` | training state and incremental log |
| GET | `/api/server/status` | bridge state and port |
| GET | `/api/events` | SSE: `train-raw` / `server-log` / `server-chat` / `env-log`; `train-raw` keeps `\r` for xterm |
| POST | `/api/config/save` | line-level value update (keeps comments/order), backs up first |
| POST | `/api/env/install` | install torch / numpy / transformers into the project Python; `torch_version` may be `2.14.1+cu126` / `2.14.1+cpu`; `torch_index=cpu|cu126|cu128|auto` remains supported; output via `env-log` |
| POST | `/api/train/start` `train/stop` | start / stop the training child |
| POST | `/api/chat` | chat, returns SSE: `meta` / `token` / `done` / `error` |
| POST | `/api/server/start` `server/stop` | start / stop the bridge child |
| POST | `/api/sign` | `mode=compute|verify`, runs the signature tool |

- The bridge prints one `@@EV@@ {json}` line per completed chat (`--events`); the webui reader
  turns it into an SSE `server-chat` event, which the "server" tab renders as a Q&A list.
- SSE frames are `data: <json>\n\n`; one thread per connection, so new long-lived endpoints
  must consider the thread budget.
- **LAN port** (`LANHandler`) only allows `GET /`, `/chat.html`, `/logo.ico`, `/api/models`,
  `/api/chat/configs` and `POST /api/chat`; everything else returns 403. `POST /api/chat` only
  accepts a `config` that appears in the config list. Changing LAN endpoints means updating
  `web/chat.html` too.

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
| `save_every` / `save_dtype` / `save_optimizer` | periodic checkpoint / dtype / keep optimizer after finish |
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
`save_optimizer=true` (default) keeps `optimizer.pt` / `scheduler.pt` / `rng_state.pth` in
checkpoints for resuming; `false` removes the optimizer from every checkpoint after training,
and the root directory keeps release weights only. The root directory always contains only
`config.json` / `model.safetensors` / `chars.json` / `meta.json`.

**Chat parameter precedence**: explicit args > `properties\<config>.properties` > values stored in
the model directory (`meta.json`) > defaults (used by both the web chat and the bridge; implemented
twice in `chat.py` and `http_server.py` — keep them in sync).

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
- **The WebUI console is local-only**: it binds `127.0.0.1` with a one-shot token; the LAN side
  exposes only the read-only chat page (`0.0.0.0:49135`, no auth — anyone on the LAN can chat).
- **CPU training** is very slow and has no bf16; use it only to smoke-test tiny models.

## 9. References

- Weights: [ModelScope · donglouqiaoque/dlqq-llm-model](https://www.modelscope.cn/donglouqiaoque/dlqq-llm-model)
- Source mirrors: [GitHub](https://github.com/donglouqiaoque/dlqq-llm) ｜ [Gitee](https://gitee.com/donglouqiaoque/dlqq-llm)
- PyTorch docs: <https://pytorch.org/docs/stable/index.html>
- nanoGPT: <https://github.com/karpathy/nanoGPT>
