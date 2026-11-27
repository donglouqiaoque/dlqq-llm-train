# dlqq-llm-train · Character-level GPT: training and chat

[中文](README.md) ｜ **English**

Weights trained by this project ｜ [ModelScope model hub](https://www.modelscope.cn/donglouqiaoque/dlqq-llm-model)

A minimal language-model toolchain built on transformers: character-level vocabulary, GPT-2
architecture, trained from scratch — **no pretrained weights, no HuggingFace tokenizer**.
Corpus in, model out — and you can chat with it when training finishes.

**Project history**: This project was migrated here from the `dlqq-llm` repository and is now named **`dlqq-llm-train`**. The old repository was deleted and rebuilt; this is a new repository with a fresh commit history; the open-source license was changed from Apache 2.0 to MIT.

```
corpus  →  character-level vocab  →  GPT-2 (from scratch)  →  AdamW training  →  sampling
```

The network is `GPT2LMHeadModel` from transformers; training starts from random initialisation
via `GPT2Config(...)` and uses only this project's corpus. Inference only calls `from_pretrained`
on a standard directory this project saved itself — **no third-party pretrained weights**.
The default tier has ~0.8 M parameters; an 8 GB laptop GPU trains it in minutes.

> The old hand-written GPT `.pt` files use different parameter names and **cannot be loaded
> directly**; retrain after switching.

---

## Contents

- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Training a model](#training-a-model)
- [Chatting with a model](#chatting-with-a-model)
- [Using your own corpus](#using-your-own-corpus)
- [Tiers and parameters](#tiers-and-parameters)
- [Runtime modes and VRAM](#runtime-modes-and-vram)
- [Directory layout](#directory-layout)
- [FAQ](#faq)
- [Serving the model to DSH](#serving-the-model-to-dsh)

---

## Quick start
Double-click `.runtime\start.bat`; the browser opens the local console at an address like
`http://127.0.0.1:49134/?t=...`. If the port is busy it does not switch silently: it asks for a
new port, and an empty Enter exits. Ten tabs:

| Tab | What it does |
|---|---|
| Train | pick a config and corpus, press Start; logs stream live and land in `.cache\train-logs\` |
| Chat | pick a config and type; the reply streams character by character and can be stopped |
| Agent | pick a tool-capable API model; run with MCP tools or chat without tools |
| API | connect any OpenAI-compatible service; only the API Key environment variable name is stored |
| Server | run a local OpenAI-compatible bridge; every request shows up on the page in real time |
| Environment | inspect the project Python, install / update torch / numpy / transformers |
| Config | edit `properties\*.properties` in a form; comments and key order are preserved |
| Signature | compute or verify SHA-256 for the standard weight file in every `model\<name>\` |
| MCP | configure STDIO / HTTP MCP servers and inspect their tool lists |
| Credits | list frontend third-party components, versions, licenses and local license files |

> **Closing the cmd window of `start.bat` means exit**: training, the bridge and any signature job
> stop with it. Closing the browser tab is not an exit — training keeps running; reopen the URL.

Phones and other machines on the same LAN can open `http://<host-LAN-IP>:49135/` — a **read-only
chat page**: no token, chat only, and the train/config/signature tabs are unreachable. The first
launch may prompt Windows Firewall; allow Python on **private networks**. Pass `start.bat --no-lan`
to skip the LAN page.

The command line is no longer a user entry point; `--config` / `--parent-pid` are an internal
protocol between the WebUI and its child processes.

## Requirements

Every Python entry point — WebUI, training, chat, bridge and signature — uses the project runtime:

```text
.runtime\python\python.exe
```

| Item | Requirement |
|---|---|
| WebUI | no system Python required; `start.bat` runs the project runtime directly |
| Training / chat / server | also run on the same project Python; a user Python is no longer consulted |
| Build time | only `make_python_env.bat` needs a full CPython 3.10+ installation |
| torch | 2.x; install from the WebUI Environment tab (PyPI / CUDA 12.6 / CUDA 12.8) |
| transformers | 5.x; installed from the Environment tab together with torch and numpy |
| GPU | 8 GB VRAM runs the default tier; CPU works but is much slower |

### First-time training dependencies

1. Double-click `.runtime\start.bat`;
2. Open the **Environment** tab;
3. Press **Detect environment** next to Project Python: it detects the chip type/driver version and fills the
   currently installed Torch / Numpy / transformers versions into the form;
4. Fill the install form row by row:
   - **PyTorch**: choose the chip type; enter the numeric driver version
     (e.g. CUDA `12.6`); the Torch library version may be empty (= latest) or pinned;
   - **Numpy**: empty means latest, or enter `2.5.4`;
   - **transformers**: empty means latest, or enter `5.22.2`;
5. Press **Install / update training dependencies** and wait for pip to finish;
   already-installed libraries with the same version are skipped;
6. Press **Detect environment** again to confirm the chip type/driver version and installed versions.

Detect environment handles the chip type/driver version and reads installed versions; empty version
fields mean latest, or pin them manually. `2.14.1` is only an example.

Ollama is no longer detected on the Environment tab; it is an OpenAI-compatible API config on the
**API** tab. The WebUI detects it at startup together with Python / MCP / other APIs, then does not
auto-detect again before training, chat, or server start.

Dependencies are installed into `.runtime\python\Lib\site-packages`, the same project Python that
runs the WebUI. The WebUI itself imports only the standard library and never imports torch, so the
UI stays usable during installation.

`make_python_env` only writes the CPython and `ensurepip` license texts to
`.runtime\python\licenses\`. torch, numpy and transformers are installed by the user into the project
Python and are not part of the distributed package, so their licenses are not collected. Files
outside the Python environment are maintained manually; make does not create or overwrite them.

### Rebuilding the project Python

If the runtime is missing or broken, double-click:

```text
.runtime\make_python_env.bat
```

Enter a full CPython installation directory (the one containing `python.exe`). The script copies only:

- the interpreter;
- the standard library;
- `ensurepip` (used by the Environment tab to bootstrap pip);
- the Windows runtime DLLs.

For automation, add:

```text
make_python_env.bat <python-dir> autoyes nopause
```

`autoyes` confirms overwrite automatically; `nopause` skips the final key prompt.

It **does not copy** `site-packages`, torch, numpy, transformers, tkinter or Tcl/Tk.

Training dependencies are lost after a rebuild; reinstall them from the Environment tab.
`make_python_env` only builds the minimal WebUI runtime, not the training environment.

> An older `.runtime\python` may not contain `ensurepip`; after upgrading, run
> `make_python_env.bat` once before using the Environment tab, otherwise pip cannot be bootstrapped.

### Which Python is used

There is exactly one source:

```text
<project root>\.runtime\python\python.exe
```

- `python=` in `properties\*.properties` is no longer read;
- PATH is no longer searched for `python` / `python3` / `py`;
- if the project runtime is missing, the WebUI tells the user to run `make_python_env.bat`;
  it never falls back to another interpreter;
- training, chat, server and signature all use the same project Python;
- restarting the WebUI clears environment-check results, so run the check again.

### Environment checks

- The **Environment** tab is the single source of truth; the shortcut check buttons on Train / Chat /
  Server all check the same project Python;
- The check reports Python version, architecture, torch / numpy / transformers, pip / ensurepip and CUDA;
- If training dependencies are missing, Train / Chat / Server refuse to start until they are installed
  from the Environment tab;
- The LAN chat page has no Environment tab; it checks the project Python automatically on send and
  returns a clear error if the check fails;
- There is no longer a “different Python per config” case, so multiple-interpreter conflicts are gone.

If the check has not run or fails, training / chat / server do not start; there is no PATH fallback.
CPU is selected automatically when CUDA is unavailable, at the cost of speed.

---

## Training a model

### Train from the Train tab (the only entry point)

1. Double-click `.runtime\start.bat`; the page opens on the **Train** tab;
2. pick a `properties\*.properties` in **Config** (the dropdown shows each file's header comment);
3. **Corpus** lists every directory under `res\` with file count and size — informational only,
   the directory actually used is `data_path` in the config;
4. press **Start**. The log streams live and is also written to
   `.cache\train-logs\<config>-<timestamp>.log`;
5. press **Stop** to stop. To continue from the last checkpoint, press **Resume training**
   (requires `checkpoint-<step>/optimizer.pt`).

### Where the configs are and how to edit them

All under **`properties\`**, one `key=value` per line, `#` starts a comment:

| File | Corpus | Tier | Notes |
|---|---|---|---|
| `chat_demo.properties` | `res\chat_demo\` | small (~0.9 M params) | 30 hand-written Q&A pairs, seconds |
| `dlqq_exam.properties` | `res\dlqq_exam\` | medium (12.55 M params) | 4,562 exam questions, a few minutes |
| `dlqq_llm.properties` | `res\dlqq_llm\` | large (31.44 M params) | GB-scale corpus |
| `dlqq_poem.properties` | `res\dlqq_poem\` | small (13.40 M params) | poetry (classical + Song ci + modern) |

**Every parameter lives in the file.** You can edit it on the **Config** tab: saving only rewrites
the values, keeping comments, blank lines and key order, and backs the old file up under
`.cache\cfg-backup\`. Changes apply to the next training run.

| Key | Meaning |
|---|---|
| `data_path` | Corpus **directory** (relative to the repo root; any number of `txt` / `jsonl` files) |
| `model` | Model name (no suffix; saved as the HF directory `model\<name>\`) |
| `cache_dir` | Where tokenised caches go (`.cache` by default) |
| `n_layer` / `n_head` / `n_embd` | Model shape |
| `block_size` / `batch_size` | Context length / samples per batch |
| `max_iters` / `eval_interval` | Training steps / evaluate every N steps |
| `lr` / `dropout` / `weight_decay` / `grad_clip` / `seed` | Optimisation |
| `max_vocab` / `val_frac` | Vocabulary cap / validation split |

Copy an existing file to add a config — the dropdown is scanned from `properties\` on each request,
so the new file shows up immediately.

### What happens without a config

If `data_path` is empty, training stops with a clear error asking for a corpus directory;
it never downloads data.

Every evaluation point prints the validation loss, and **only the best validation checkpoint is
saved**. After training a sample is generated so you can see where the model stands.

---

## Chatting with a model

Use the **Chat** tab:

1. **Model** lists every standard model directory under `model\` with size, mtime and signature status;
2. adjust temperature / top_k / length (defaults come from `properties\<model>.properties`);
3. type and press Enter — the reply streams character by character, and **Stop** interrupts it.

- Switching models does not need a page reload: the old model is released first (VRAM included);
- while training is running you may only pick a **different** model; the model named by the running
  config is rejected, so training is not slowed down and you never read a half-written file;
- the conversation is truncated to the model's `block_size`;
- assistant replies are rendered as Markdown, including `$...$` and `$$...$$` math via local KaTeX;
  user messages stay plain text;
- API models (including Ollama) show a collapsible thinking block and a thinking-level selector
  based on detected metadata; local models do not show the thinking area.

### Two modes

- **Dialogue mode**: your text becomes `用户：xxx` + `助手：`, and the model continues as the assistant;
- **Continuation mode**: your text is treated as ordinary prose and the model writes on.

If the model's vocabulary has no `用户：` / `助手：` characters it was not trained on dialogue data,
and the session **switches to continuation mode automatically**.

---

## Using your own corpus

**All corpora live in one directory** (`data_path` accepts directories only, not single files).

Create a folder under `res\`, e.g. `res\my_corpus\`, and drop files in:

```
res\
  my_corpus\
    part1.jsonl
    part2.jsonl        ← any number of files; merged in filename order
    notes.txt
```

Supported files:

| File | Handling |
|---|---|
| `.jsonl` with Q&A fields (`input`/`target`, …) | Joined as `用户：… / 助手：…` |
| `.jsonl` plain text (`{"text": "..."}`) | Fed as plain text |
| `.txt` | Fed as plain text (UTF-8 / GBK / BIG5 auto-detected) |

For a chat model, write the jsonl like this:

```json
{"input": "什么是过拟合", "target": "模型把训练数据背下来了，换一批新数据就不行了。"}
{"input": "那怎么缓解", "target": "加数据、加正则、减小模型容量，三条路一起走。"}
```

Then train:

```bat
# create a new file under properties\ with these three lines:
#   data_path=res/my_corpus
#   model=mymodel
#   max_iters=5000
```

Rough expectations:

| Corpus size | What you get |
|---|---|
| < 100 KB | Mostly **memorised**; exact questions work, rephrasings break |
| ~1 MB | Sentence patterns start to generalise |
| 10 MB+ | Short sentences are fluent, long ones drift |

Encodings are detected per file (UTF-8 / utf-8-sig / gb18030 / big5), so Notepad's GBK
files work as-is.

### Clean downloaded corpora first

Wikipedia dumps and web crawls cannot be fed directly (`{{templates}}`, `[[links]]`,
mixed languages and ads). Cleaning strips wiki markup, normalises whitespace, filters by
length and Chinese ratio, and de-duplicates lines. The helper scripts that do this are local
tools, not shipped with the repo — usage in AIRES.md.

It strips wiki markup, normalises whitespace, filters by length (`--min-chars`) and
Chinese ratio (`--min-zh`), de-duplicates lines, and writes a `.txt` (blank line between
paragraphs) or `.jsonl` the trainer can read.

---

## Tiers and parameters

| Tier | n_layer | n_head | n_embd | block | batch | Params | VRAM | Notes |
|---|---|---|---|---|---|---|---|---|
| `tiny` | 2 | 2 | 64 | 64 | 16 | ~0.2 M | < 0.5 G | Seconds per run; sanity check |
| `small` (default) | 4 | 4 | 128 | 128 | 32 | ~0.8 M | < 1 G | Visible results |
| `medium` | 6 | 6 | 384 | 256 | 32 | ~10 M | ~2–3 G | Clearly better with enough data |
| `large` | 8 | 8 | 512 | 256 | 32 | ~25 M | ~6–7 G | At the edge of 8 GB; do not go bigger |

Every config key can be edited on the Config tab, or directly in the properties file:

```properties
n_layer=6
n_embd=384
lr=0.001
max_iters=5000
```

Common keys:

| Key | Default | Meaning |
|---|---|---|
| `data_path` | none | Corpus directory; training errors out when it is empty |
| `out_path` | `model/mymodel` | Model output directory (HF standard directory) |
| `max_iters` | 2000 | Training steps — the simplest "how long" knob |
| `batch_size` | 32 | Samples per step; lower it when VRAM is tight |
| `block_size` | 128 | Context length; shrinks automatically on tiny corpora |
| `lr` | 3e-4 | Learning rate |
| `dropout` | 0.1 | Raise it for small corpora |
| `eval_interval` | 200 | Evaluate (and maybe save) every N steps |
| `max_vocab` | 8000 | Vocabulary cap: keep the N most frequent characters |
| `val_frac` | 0.05 | Tail fraction used for validation |
| `cache_dir` | `.cache` | Tokenised binary cache lives here |
| `prompt` | empty | Sampling prompt after training; auto-picked when empty |
| `data_format` | `auto` | `auto` / `txt` / `jsonl`; auto sniffs extension and header |
| `q_field` | empty | jsonl question field; inferred when empty |
| `a_field` | empty | jsonl answer field; inferred when empty |
| `eval_iters` | 50 | Batches per evaluation; 20 saves time on long runs |
| `grad_accum` | 1 | Gradient accumulation to fake a larger batch |
| `warmup_iters` | 0 | LR warmup steps; 1000–2000 for long runs |
| `lr_decay` | `cosine` | `cosine` decay / `none` constant |
| `min_lr_frac` | 0.1 | Decay floor = `lr × this` |
| `grad_checkpoint` | `False` | Checkpointing: less VRAM, ~20–30% slower |
| `save_every` | 0 | Extra forced save every N steps (0 = only keep the best model) |
| `keep_best_checkpoints` | 1 | Keep the latest N validation-best checkpoints; 0 = keep none, -1 = unlimited |
| `keep_forced_checkpoints` | 1 | Keep the latest N forced checkpoints; 0 = keep none, -1 = unlimited |
| `chat_mode` | `True` | `False` = pure continuation model (poetry), no `用户：/助手：` |
| `chat_temperature` | 0.8 | Sampling temperature; higher is wilder |
| `chat_top_k` | 40 | Sample from the top K; 0 = no truncation |
| `chat_max_new` | 180 | Max characters generated per turn |

`chat_*` keys are **dialogue parameters** stored in the checkpoint. Both the web chat and the
bridge resolve them by this priority (editing properties takes effect without retraining):

```
explicit parameter  >  properties\<config>.properties  >  value in the checkpoint  >  default
```

---

## Training `dlqq_llm` longer (optional)

`dlqq_llm` trains directly from `properties\dlqq_llm.properties`; to train longer, edit
that file — no extra config needed:

| Parameter | Effect |
|---|---|
| `max_iters` | Steps; multiply by `batch_size × block_size` to estimate tokens seen |
| `grad_accum` | Accumulate gradients to fake a bigger batch |
| `warmup_iters` / `lr_decay` / `min_lr_frac` | Warmup + cosine decay; a must for long runs |
| `grad_checkpoint` | Checkpointing: less VRAM, 20–30% slower |
| `save_every` | Extra forced checkpoint every N steps (0 = only keep the best model) |
| `keep_best_checkpoints` | Keep the latest N best checkpoints; 0 = none, -1 = unlimited |
| `keep_forced_checkpoints` | Keep the latest N forced checkpoints; 0 = none, -1 = unlimited |

Example: set `max_iters=60000`, run `.runtime\start.bat`, pick `dlqq_llm` on the Train tab and press
**Resume training** (the checkpoint must contain optimizer state).

---

## A tiny exam demo (`dlqq_exam`)

Repo size is limited (Gitee caps a single file at 50 MB). `chat_demo` is 30 hand-written
Q&A pairs (13 KB) — enough to smoke-test the pipeline, but thin to play with.
`dlqq_exam` uses an **exam question bank** instead: far more information per byte.

Two steps:

```bat
REM 1) Prepare the exam corpus: merge C-Eval / gaokao / biology into res\dlqq_exam\dlqq_exam.jsonl
REM    (the corpus is not in the repo — download and merge it yourself)

REM 2) Train and chat
REM    Double-click .runtime\start.bat, pick dlqq_exam on the Train tab.
REM    Then pick dlqq_exam on the Chat tab to chat.
```

Staying under 50 MB comes down to these lines in `properties\dlqq_exam.properties`:

```properties
# half-precision weights, half the size
save_dtype=fp16
# checkpoints always keep the optimizer; cap best and forced snapshots separately
keep_best_checkpoints=1
keep_forced_checkpoints=1
```

`model\dlqq_exam\` only holds the best weights and never contains an optimizer; checkpoints are for
resuming training.

Measured: 4,562 records tokenise to ~2.7 M tokens; 12.55 M parameters at fp16 give a release weight
of about **27 MB**. `chat_demo` is smaller still: 0.9 M parameters ≈ **1.9 MB**.

> **⚠ Attribute the source**: `dlqq_exam` trains on the questions themselves, so the model
> **memorises them**. Credit the datasets (ModelScope: C-Eval / gaokao-benchmark / …) and
> **do not** report benchmark scores on the same questions — that is self-grading, not ability.
> C-Eval is licensed CC BY-NC-SA 4.0 (non-commercial); do not redistribute this corpus.

### Where corpora come from (ModelScope only, direct in mainland China)

| Dataset (ModelScope repo) | Size | Use |
|---|---|---|
| `philexohf/LLM-Pretrain-Data` | ~24.6 GB / 4.96 B tokens | **Preferred pretraining**: web 55% + news 27% + wiki 12% + encyclopedia 6%, cleaned and deduped, one document per line |
| `AI-ModelScope/chinese-fineweb-edu` | ~290 GB | Subset when you only want web text |
| `OmniData/C-Eval` | 1.5 MB | 52-subject multiple choice |
| `opencompass/gaokao-benchmark` | 5.7 MB | Gaokao multiple choice |
| `pandayi/Biology_subject_of_college_entrance_examination` | 0.2 MB | Gaokao biology (with explanations) |
| `SHPDataGR/medical-exam-question-bank-json` | 35 MB | Medical exam bank |
| `tongyi_dianjin/CFLUE` | 66 MB | Finance qualification questions |

Downloads:

```bat
REM Exam data: download C-Eval / gaokao-benchmark / gaokao biology from ModelScope and merge
REM            (ModelScope only — no GitHub, no HuggingFace, no login)

REM Pretraining text: download, clean into jsonl, drop it into the corpus folder the config points at
```

Downloading and cleaning are done by local helpers (not shipped with the repo) — usage in
AIRES.md. The resulting jsonl goes straight into the corpus folder the config points at, so
download and train.

### How big is the pretraining corpus, and why 2 GB by default?

The helper can list the corpora and their sizes — usage in AIRES.md:

```
mix      philexohf/LLM-Pretrain-Data
         1 file available, 12.72 GB total
fineweb  AI-ModelScope/chinese-fineweb-edu
         302 files available, 311.44 GB total
```

| Corpus | Total | Notes |
|---|---|---|
| `mix` (train.txt) | **12.72 GB** ≈ 4.5 B tokens | web + news + wiki + encyclopedia, one doc per line |
| `fineweb` | **311 GB** (302 parquet shards) | Education-oriented web text, shard by shard |

2 GB is chosen from the model size:

- `dlqq_llm` is 31.44 M parameters; at "20 tokens per parameter" it needs ≈ **630 M tokens**;
- 2 GB of Chinese text ≈ 700 M tokens, plus the existing `firefly-train-1.1M.jsonl`
  (1.1 GB ≈ 400 M tokens) gives ~1.1 B tokens — already more than enough;
- Downloading all 12.7 GB buys a 29.4 M model nothing but 1–2 hours and 13 GB of disk.

For the full set pass `--max-mb 0` (download until done), or `--max-mb 8000`.

---

## Training on GB-scale corpora

1 GB of Chinese is roughly 350 M characters. Reading a whole file into memory and
encoding it into a Python list would take tens of GB — so the pipeline is **streaming**:

```
read in chunks → numpy counts characters → keep top max_vocab → encode to uint16 → write a cache file
training uses np.memmap and reads a few KB at random
```

**Memory use is independent of corpus size**; only `block_size` / `batch_size` matter.

### Feed JSONL directly, no conversion to txt

Point `data_path` at a directory containing `.jsonl` and the script finds the Q&A fields
and joins them into `用户：… / 助手：…` (**feeding raw jsonl teaches the model braces and
quotes, so this step matters**).

Field pairs tried in order:

```
input/target · q/a · question/answer · src/tgt · instruction/output
prompt/response · query/response · ask/answer · title/content
```

If none match, specify them: `q_field=... a_field=...`.

### Measured (1.12 GB in `res\dlqq_llm\`, 1,649,399 Q&A records)

| Metric | Value |
|---|---|
| Tokenisation time | **30 s** (including startup; reused afterwards) |
| Tokens produced | 425,220,455 |
| Vocabulary | 8000 (capped); only 210 k characters (0.05%) dropped |
| `data.bin` | 850 MB |
| Peak memory | ~1.5 GB (independent of corpus size) |
| Second run | 4.9 s (cache reused) |

Key settings to put into `properties\dlqq_llm.properties`:

```properties
data_path=res/dlqq_llm
model=dlqq_llm
max_vocab=8000
n_layer=8
n_head=8
n_embd=512
block_size=256
batch_size=32
max_iters=60000
lr=0.0003
eval_interval=2000
```

Measured speed on this corpus (29.4 M parameters, fits in 8 GB):

| Tier | Config | Speed | 60,000 steps |
|---|---|---|---|
| small | 4 layers / 128 dim / block 128 / batch 32 | 143 steps/s | ~7 min |
| large | 8 layers / 512 dim / block 256 / batch 32 | ~5 steps/s | ~3.3 h |

> The small tier (0.8 M parameters) badly underfits 1 GB — **use large for GB-scale data**.
> In the smoke test, large pushed validation loss from 9.06 to 5.72 within 30 steps,
> so the data path is sound.

Two caveats:

- Each corpus (together with `max_vocab` / `val_frac` / format / field settings) gets its
  own cache file; changing any of them writes a new one. So the demo `chat_demo\` and the
  GB-scale `dlqq_llm\` can **coexist** — switching back and forth never re-tokenises.
  Delete the cache folder to start over;
- The validation split is taken from the **tail** of the corpus. If the data is ordered by
  category (news first, dialogue later), shuffle the lines first — otherwise validation
  loss will not reflect real quality.

### Adding data: drop files into the directory

`data_path` accepts a directory; `dlqq_llm.properties` points at one:

```properties
data_path=res/dlqq_llm
```

The directory can hold any number of files; they are read in filename order and
concatenated into one stream:

| What you put in | Handling |
|---|---|
| `*.jsonl` with Q&A fields | Joined as `用户：… / 助手：…` |
| `*.jsonl` with only `{"text": "..."}` | Fed as plain text (pretraining data looks like this) |
| `*.txt` | Fed as plain text |
| Subdirectories | Scanned too (hidden ones skipped) |

Adding, removing or editing files changes the cache fingerprint, so **the next run
re-tokenises automatically** — no manual cache cleanup.

To grow the corpus, download from ModelScope (no GitHub / HuggingFace / login); downloading
and cleaning are done by local helpers — usage in AIRES.md.

It filters by length and Chinese ratio, de-duplicates lines, and writes
`pretrain-mix-000.jsonl`, `pretrain-mix-001.jsonl`, … at ~200 MB each. **Existing files are
never overwritten** (numbering continues). A progress bar runs during download and it prints
"corpus directory now has N files / X GB" at the end. Then double-click `.runtime\start.bat` and
pick `dlqq_llm` on the Train tab.

---

## Training a poetry model (`dlqq_poem`)

A small model dedicated to classical Chinese poetry: 6 layers × 384 dim (13.40 M parameters),
trained on **Tang/Song/Yuan/Ming/Qing poems + Song ci + modern poetry**; about 30 minutes on
8 GB VRAM.

The corpus holds **poem text only — no `诗：` / `词：` labels**, so the model writes poems
instead of repeating labels.

### 1. Download the corpus

The three sources (poems 43 MB, ci 19 MB, modern 57 MB) are downloaded and cleaned by local
helpers — usage in AIRES.md.

| Source | Repo | Size | Content |
|---|---|---|---|
| `shici` | `modelscope/chinese-poetry-collection` | 43.4 MB | Tang to Qing poems |
| `songci` | `peaceminuczy/songci` | 18.7 MB | Song ci |
| `modern` | `Iess/chinese_modern_poetry` (hf-mirror.com) | 56.6 MB | Modern poetry (Chinese poets + translations) |

Everything lands in `res\dlqq_poem\` as `shici-000.jsonl`, `songci-000.jsonl`,
`modern-000.jsonl`. Existing files are kept, a progress bar runs, and lines break at
punctuation.

> Older corpora carried `诗：` / `词：` prefixes. The script detects such files and **stops,
> telling you to delete them** (`del res\dlqq_poem\*.jsonl`); otherwise old and new files
> mix the labels into the text.

### 2. Train

```bat
REM Double-click .runtime\start.bat, then pick dlqq_poem on the Train tab.
```

`properties\dlqq_poem.properties` is tuned to the actual corpus:

| Item | Value | Rationale |
|---|---|---|
| Shape | 6 layers, 384 dim (13.40 M) | ~40 M characters / 13.4 M params ≈ 2.9 tokens per parameter |
| `block_size` | 256 | Modern poems average 210 chars, 90% under 136 — 256 fits a whole poem |
| `batch_size` | 32 | 32 × 256 = 8,192 tokens/step (same steps as 64×128, double the context) |
| `max_iters` | 15000 | ≈ 123 M tokens ≈ 3.1 passes over the 39.2 M-token corpus |
| `max_vocab` | 7000 | Covers >99.9% of characters |
| `chat_mode` | `false` | Tells `chat.py` this is a continuation model — no `用户：/助手：` |

Measured ~70 k tokens/s, **about 29 minutes** end to end; only the best validation checkpoint
is saved (fp16).

### 3. Let it write

The corpus is bare text, so "chatting" means **give it a start and it continues**:

```bat
REM Double-click .runtime\start.bat, then pick dlqq_poem on the Chat tab.
```

```
你  > 春
它  > 风暖鸟声碎，日高花影重。
      ...
```

- One or two characters → it starts a line from there;
- A line of poetry → it continues;
- A modern-poetry sentence → it continues in that style.

Because `chat_mode=false` is stored in the checkpoint, `chat.py` uses continuation mode
and never prepends `用户：/助手：`.

Sampling parameters also live in `properties\dlqq_poem.properties` and take effect
**without retraining**:

```properties
chat_mode=false
# 0.8 makes a small model loop ("他的女人，他的女人")
chat_temperature=1.0
# 0 = sample from the full vocabulary
chat_top_k=0
chat_max_new=200
```

The Chat and Server tabs both read these; to change them temporarily, edit the same keys in the
properties file and restart that tab/service.

---

## Runtime modes and VRAM

The training log and the **Environment** tab report the runtime environment:

```
[模式] GPU —— 混合精度 bf16（推荐：显存占用更低、张量核更快）
[设备] NVIDIA GeForce RTX 4060 Laptop GPU | sm_89
[显存] 总 8.0 GB，已用 1.1 GB（可用 6.9 GB）
```

The three modes:

| Mode | Trigger | Notes |
|---|---|---|
| **GPU · bf16 mixed precision** | CUDA device present | Forward/backward in bf16; less VRAM, faster tensor cores |
| **GPU · fp32** | CUDA present but AMP off | More precise, more VRAM |
| **CPU · fp32** | No usable CUDA device | Also reports **RAM** usage; bf16 on CPU is slower, so it is off |

When VRAM is short: lower `batch_size` → lower `block_size` → drop to a smaller tier.

---

## Directory layout
```
.runtime\                local runtime and launcher (python/ not in git)
  start.bat             double-click entry: always .runtime\python\python.exe → runs code\webui.py
  make_python_env.bat   rebuilds .runtime\python; prompts for a system Python path
  make_python_env.py
  logo.ico              shortcut icon
  python\               project Python: stdlib + ensurepip (training deps installed from Environment tab)
code\                   all Python
  webui.py              local web service: static files + JSON API + SSE + child processes
  env_install.py        install training / chat deps into the project Python
  train.py              training (data pipeline + model + loop + sampling)
  chat.py               chat core: ChatSession (load / streaming reply / set params)
  chat_worker.py        chat child: keeps ChatSession alive, JSON lines over stdin/stdout
  http_server.py        OpenAI-compatible bridge (local models + API proxy)
  ollama_client.py      Ollama native metadata helper (for API-config capability enrichment)
  openai_client.py      OpenAI-compatible API client (Base URL / env-var Key / SSE / embeddings)
  mcp_client.py         MCP client (STDIO / Streamable HTTP / tool schemas)
  common.py             find_root / order-preserving properties I/O / watchdog / hide dir
  model.py              GPT-2 wrapper (transformers)
  model_signature.py    weight SHA-256 signatures (stdlib only)
web\                    frontend served by webui.py
  index.html            ten tabs: train / chat / Agent / API / server / environment / config / signature / MCP / credits
  chat.html             LAN read-only chat page
  vendor\               local third-party assets: xterm.js / marked / DOMPurify / KaTeX
properties\             four training configs
res\                    corpora (not in git)
model\                  training output (model\<name>\ standard HF directory + checkpoint-<step>\ + .sha256)
                        config.json / model.safetensors / chars.json / meta.json
.cache\                 training cache/logs/config backups/MCP and API configs/sessions (directory tracked, contents ignored)
.ai_res\                AI scratch area (not in git, never used for training)
THIRD-PARTY-NOTICES.txt  license inventory for Python / dependencies / frontend libraries (manual)
```

- The project root is found by walking up to the directory containing `code\train.py`.
- Model id = directory name: `model\chat_demo\` → `chat_demo`.
- Resume directory: `model\<name>\checkpoint-<step>\` contains `config.json` / `model.safetensors` /
  `optimizer.pt` / `scheduler.pt` / `trainer_state.json` / `rng_state.pth`.

## FAQ

**Q: The window flashes and disappears.**

Every `.bat` ends with `pause`, so a normal run stays open. A real flash usually means
`python.exe` was not found — run it from a terminal to see the full error:

```bat
.runtime\start.bat
```

**Q: "Missing dependency" at startup.**

Open the WebUI **Environment** tab, press **Check environment**, then **Install / update training
dependencies**. Dependencies go into the bundled project Python; do not install into a different
Python on PATH.

**Q: Training reports missing dependencies after rebuilding `.runtime\python`.**

`make_python_env` only rebuilds the minimal WebUI runtime without torch / numpy / transformers.
Reinstall them from the Environment tab after a rebuild.

**Q: Running from an IDE's Run button fails with "cannot read keyboard input".**

That mode has no interactive terminal, so `input()` gets EOF and the number prompt cannot
work. Two options:

1. **Double-click** `.runtime\start.bat` (a real console);
2. If you run `code\webui.py` directly and the port is busy, there is no terminal to enter a new
   port; use `.runtime\start.bat` or free the port first.

**Q: Will retraining with a different corpus collide with the model name?**

No. If the name exists you are asked whether to overwrite; answer `n` and pick another name.

**Q: I want to train but not with `chat_demo`.**

Create a corpus folder under `res\`, then copy a config and change two lines (the new
config is listed automatically):

```
# properties\my_corpus.properties
data_path=res\my_corpus      ← a directory holding jsonl / txt
model=mymodel
...（copy the rest of chat_demo.properties; adjust n_layer / max_iters as needed）
```

**`data_path` accepts directories only**; `txt` and `jsonl` are both supported and the
format is inferred from extension and header. For jsonl the Q&A fields are auto-detected
(single-field `{"text": ...}` is treated as plain text).

**Q: I want to change one parameter without editing the config.**

Call the training script directly; command-line values override the config:

```bat
# change in the config: data_path=res/chat_demo, model=chat_demo, max_iters=500
```

**Q: The model still parrots after a long training run.**

The corpus is too small. A character-level model under 100 KB is mostly reciting;
aim for 1 MB+ to see generalisation.

**Q: Can I train on CPU?**

Yes, but it is slow, and bf16 is disabled on CPU (it is slower there). Start with the
`tiny` tier to validate the pipeline.

**Q: How do I know a weight file was not altered or corrupted?**

Double-click `Verify on the Signature tab`: it walks every standard model directory under `model\`,
recomputes the weight file's SHA-256 and compares it with the adjacent `<weight-file>.sha256`; every
file must print PASS. Run it after training, or before uploading weights to the model hub.
Re-generate signatures with `Compute on the Signature tab` (mandatory once the weights change).

---

## Serving the model to DSH
On the **Server** tab, first confirm in the **Environment** tab that the project Python passes the
check. The server no longer checks a Python per config and has no “all configs / multiple Python”
conflict. With "check on server start" ticked, a missing check is done automatically before launch.
Set the port (default 49128) and press Start. The page shows `http://127.0.0.1:49128/v1`, which you put
into the DSH provider settings. Every request then appears in the session list below (fed by the
`@@EV@@` lines the service prints to stdout), so you debug from the page instead of the console.
Press Stop to shut it down — closing the `start.bat` window also stops it.

The service is OpenAI-compatible: `/v1/models`, `/v1/chat/completions` (with `stream`) and
`/v1/embeddings` (API embedding models). Local models generate and send token by token when
`stream=true`; API models proxy the upstream SSE and embeddings. The service page distinguishes
chat / embedding from detection results, and embedding requests write the **Vectors** log with
model, count, dimensions, latency and the complete first vector in a collapsible detail; full
high-dimensional vectors are not placed in chat bubbles. Models come from the `properties\`
configs whose `model=` points at an existing standard model directory (no more directory
scanning), and chat parameters are read from the same config. Models load only on first use.

### Ollama as an API config

Ollama is merged into the OpenAI-compatible API and no longer uses separate `ollama/` model ids:

```text
name             Ollama
Base URL         http://127.0.0.1:11434/v1
Key env var      (leave empty)
```

At startup the WebUI requests `http://127.0.0.1:11434/v1/models`, then uses Ollama's `/api/show`
to enrich type, tool and thinking metadata. Models appear as:

```text
Ollama/qwen3.5:4b (API)
Ollama/bge-m3:latest (API · embedding)
```

Embedding models only appear on the Service page, never in Chat / Agent.

### MCP / Agent (optional)

The **MCP** tab stores multiple MCP server definitions and supports two transports:

```text
STDIO: command + args + cwd
HTTP : http://127.0.0.1:<port>/mcp
```

STDIO example:

```text
java -Dfile.encoding=UTF-8 -jar <MCP-SERVER-JAR>
```

HTTP example:

```text
http://127.0.0.1:<port>/mcp
```

Definitions are stored in `.cache\mcp_servers.json`. After saving, press **Connect and list tools**
to verify the connection and inspect tool names. The WebUI also detects saved MCP servers at
startup and stores the result in `.cache\mcp_status.json`; **Detect all** in the MCP tab re-runs it.
Configured STDIO MCP processes start with the WebUI and stay resident instead of exiting after each
conversation; a config change restarts that process, and WebUI exit closes them all.
MCP connections recover automatically: after a STDIO process exit or an HTTP connection loss,
the WebUI reconnects with a 500 ms initial delay and 30 s maximum exponential backoff, at most
10 consecutive attempts. During an outage tools stay listed but calls return a clear error;
after reconnecting the tool list is fetched again. After the attempt limit the connection is
marked failed and automatic retry stops; use **Reconnect** on the MCP tab to retry manually.
Non-idempotent tool calls that were already dispatched are never replayed automatically.

The **Agent** tab only lists API models whose tool capability is not explicitly unsupported, plus
saved MCP servers. Model, MCP server, the **Tools** multi-select and
**Thinking** level live above the composer. The tool panel can search by tool name or
description; Select all / Clear apply to the current filter results. With no tools selected,
Agent behaves as plain chat: no tool list is sent and no tool call is executed; running MCP
servers stay resident. For each message:

```text
user message → model tool_calls → MCP tools/call → tool result → model continues
```

Execution is shown as a flat Agent timeline rather than nested bubbles: the turn header shows
status and elapsed time, reasoning / tool-call rows are collapsed by default while answers
remain visible; clicking the header expands the history, and each arguments/result block expands
on its own. Reasoning / tool call / answer rows follow in the model's actual output order. As
soon as the tool name is known, the row shows "preparing / calling"; it changes to "done" or
"failed" after execution, and arguments/results are shown in full and sent back to the model in
full, with no hard truncation. Only tools selected for this request are executed; if the model
calls an unselected or unknown tool, the result is "tool does not exist" and MCP is not called.
With a thinking level selected, reasoning and tool calls are interleaved above the answer in
their actual order. Embedding models
and local character-level models are not listed in Agent; local character-level models do not
support the thinking level. The bridge continues to forward client-provided `tools` to the model
as before; it does not execute MCP tools inside the bridge.

For large files, select only the relevant file tools (for example `dlqq_create_file` /
`dlqq_append_file` / `dlqq_overwrite_file`) and do not ask the model to emit the whole file in one
go. A very large tool list consumes context and makes the model more likely to invent a tool name
or produce a truncated `tool_call`.

### Session persistence and images

Chat / Agent have a session selector at the top. Sessions are saved, switched, and deleted there;
each turn auto-saves to `.cache/session-<id>.json`, with workspace metadata in
`.cache/workspace.json`. Restoring a session restores its history, model, MCP server, selected
tools, and thinking level; interrupted model/tool turns are not replayed automatically.

When a model declares `input_modalities` containing `image`, the composer shows an **Images**
button for up to four images of 6 MB each. Images are sent as OpenAI-compatible `image_url`
content blocks; local character-level models reject images. Old images are represented by
placeholders or summaries during context compaction and are not charged again as base64 text.

### Connect an OpenAI-compatible API (optional)

The **API** tab stores multiple OpenAI-compatible service definitions:

```text
name
Base URL (for example https://api.openai.com/v1)
API Key environment variable name
```

Model names are no longer entered manually; capabilities are detected from `/models` first.
Unknown context, output, and tool support are shown as **unknown**, not as fixed values. Click a
model row on the API tab to override its context, maximum output, tool capability, input types,
and reasoning levels; overrides are stored in `.cache/openai_model_overrides.json` and are not
overwritten by later detection. `GET /models` does not consume completion tokens, but it is still
an HTTP request and may count toward rate limits.

Definitions live in `.cache\openai_apis.json`; detection results live in
`.cache\openai_status.json`. **Only the environment variable name is stored, not the Key.** Set the
actual Key in the process environment before starting the WebUI, for example. Local services such
as Ollama may leave the env-var name empty, in which case no `Authorization` header is sent.

```bat
set OPENAI_API_KEY=sk-...
start.bat
```

Configured services appear in Chat as `DeepSeek/deepseek-flash (API)`; models whose tool capability
is supported or unknown appear in Agent, while explicitly unsupported models do not. The Service
page lists every API model (chat / embedding), and the bridge proxies `/v1/chat/completions` and
`/v1/embeddings`; embedding requests are written to the Service page's Vectors log.
**Detect models** in the API tab only re-requests the selected API's `/models`; startup still
detects every saved API. The model list shows type, tool support, input types, reasoning levels,
and the context limit. Ollama chat requests use a 600-second timeout while other providers use
300 seconds. Context is read from `/models` fields such as `context_length`, `context_window`,
`max_model_len` and `top_provider.context_length`; when unavailable the UI shows
`unknown (provider default)` while a conservative internal budget is used. Request history is
compacted against the context after reserving output space: old history is summarized first and
falls back to deterministic trimming, keeping `tool_calls` / `tool_result` paired. Note that
Agent's "max_tokens output limit" error is the per-response output cap, not the context window.
Chat and Agent show an estimated context occupancy and percentage below the composer
(character-based approximation; images use a fixed visual-token estimate), together with the
output cap.

---

This project is released under the **MIT** license. See [LICENSE](LICENSE).

