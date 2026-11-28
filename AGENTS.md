# AGENTS.md — dlqq-llm-train 工程约定

> 本文给 AI 助手（DeepSeek Harness / Qwen Code / Claude Code / Cursor 等）阅读：
> 硬性规则、目录结构、启动链路、接口约定与运行时行为。
> 使用说明见 [README.md](README.md)（英文版 [README.en.md](README.en.md)）；
> 本文英文版见 [AGENTS.en.md](AGENTS.en.md)。
> README 与 AGENTS 是**中英文双份**的唯一说明文档，改一份必须同步改另一份。
>
> 配套仓库：
> - `dlqq-llm-model`：发布 HF 标准模型目录（config.json + model.safetensors + chars.json），不含脚本；
> - `dlqq-llm-dataset`：语料备份（私有，不公开），与代码链路无关。
>
> 本地临时目录（AIRES）的具体作用、目录结构和使用约定请查看 AIRES.md，此处不进行说明。

## 一、硬性规则（必须遵守）

1. **技术栈固定**：Python 3.11+ / PyTorch 2.x / transformers。后端允许 `torch`、`numpy`、
   `transformers` 三个第三方库，其余一律标准库；前端以原生 HTML/CSS/JS 为主，
   **前端允许的本地第三方库**：`web/vendor/` 中的 `xterm.js`（MIT）、`marked`（MIT）、
   `DOMPurify`（Apache-2.0 / MPL-2.0）和 `KaTeX`（代码 MIT，字体 OFL-1.1）；
   **不引其他框架、不引 CDN**（断网也要能用）。
2. **只使用算法和框架，禁止任何已有模型权重**：字符级模型，词表由 `code/train.py` 自己统计；
   网络使用 `transformers.GPT2LMHeadModel`，但必须从 `GPT2Config(...)` 随机初始化开始训练。
   **禁止**：
   - 用 `from_pretrained("gpt2")` / `AutoModel...from_pretrained(第三方)` 加载外部预训练权重；
   - 加载任何他人预训练的 `.bin` / `.safetensors` / `.pt` 权重；
   - 使用 HuggingFace tokenizer、预训练词表或现成模型的 embedding；
   - 把任何已有模型的权重当作初始化、微调起点或“底座”。

   **允许**用 `from_pretrained(model\<名字>\)` 读取**本项目自己刚训练出的标准 HF 目录**；
   它只是加载自己的训练结果，不能作为预训练起点，也不能加载 `model\` 之外的模型。

   **例外（仅推理代理）**：`ollama_client.py` 可以把 HTTP 请求转发给用户自己启动的 Ollama，
   不读取、不保存、不训练任何第三方权重；它只作为可选外部推理后端，不参与本项目训练链路。
3. **git 权限分三层，禁止放宽**：写操作（`commit` / `push` / `add` / `reset` / `checkout` 等）
   **无论用户是否授权都绝对禁止**；`.git` 目录**无论用户是否授权都绝对禁止读写**，无论是标准
   编辑工具（读/写/编辑）还是执行脚本或命令，一律禁止读写 `.git` 目录下的任何文件（读也不行，
   防止把只读命令写成写/删命令）。**唯一例外**：用户要求写提交信息时，先执行一次只读 git
   命令读取差异（见第 20 条）；除此之外不执行任何 git 命令，包括 `status` / `diff` / `log`。
4. **禁止用脚本批量改写源码**：逐文件、逐处确认后再改；大范围调整先出清单。
5. **禁止 AI 执行有副作用的操作**：训练、下载/清洗语料、安装、删除、解包、批量清理、
   启动 WebUI 或桥接服务、构建运行环境，一律交给用户。AI 只做只读检查、语法与静态检查、
   文档改动。
6. **`.bat` 只能出现 ASCII**；`.py` / `.properties` / `.md` / `.html` / `.js` / `.css` 用 UTF-8 无 BOM；
   Windows 下文本文件统一 CRLF。
7. **入口/构建脚本位置固定**：统一放 `.runtime/`；不要把 `.bat` / `.ps1` / `.py` 散落到
   工程根目录或其它目录。
8. **禁止硬编码本机路径**：文档与注释里的示例只能是通用形式。
9. **权重与语料不进版本库**：`model/*`、`res/*` 均被忽略；权重发布在 `dlqq-llm-model`
   （`model\<名字>\` 标准 HF 目录与权重文件同名 `.sha256` 成对），语料备份在 `dlqq-llm-dataset`。
10. **README / AGENTS 不写本机临时目录的细节**：具体约定不在本文展开，只看文首说明。
11. **先看目录再创建**：新建任何文件或目录之前，先列一下当前目录（确认路径、命名、
    是否已存在），禁止凭空拼路径、造目录。
12. **先回答问题，别动手**：用户只是提问、确认、咨询或讨论时，**只回答**——不改代码、不改文件、
    不执行脚本或命令、不启动进程、不改系统设置、不跑测试。先用现有信息把结论说清楚；只有用户
    明确要求「改」「修」「执行」时才动手。拿不准是不是「只是问问」时，按「只是问问」处理。
13. **跟随用户语言**：用户用哪种语言提问，就用哪种语言思考和回复；没有明确语言时默认用
    简体中文思考和回复。
14. **中英文文档必须同步**：修改 `AGENTS.md` / `AGENTS.en.md`、`README.md` / `README.en.md` 时，
    必须中英文两份一起改，不能只改其中一个。
15. **代码改动必须同步 README**：改代码、配置键、默认值、界面行为或启动方式时，
    `README.md` / `README.en.md` 的说明必须一起改；不能留下代码和文档不一致的描述。
16. **写文件只用标准编辑工具**：创建或修改文本、代码、文档必须用内置的 `write` / `edit` 工具；
    禁止用 pwsh / cmd / PowerShell（包括 `Set-Content`、`>` 重定向、`[IO.File]::WriteAllText`）
    或 MCP 命令代替写文件。构建、渲染图片等由程序产出的文件除外。
17. **`.bat` / `.ps1` 必须 pause 且支持 nopause**：正常结束和失败退出前都要停住，方便双击
    查看报错；`.bat` 识别 `nopause`、`-nopause`、`/nopause`，`.ps1` 识别 `-NoPause`
    （均不区分大小写）。AI 调用时必须传入，避免被阻塞到超时。
18. **测试必须走真实入口**：验证 `.bat` / `.ps1` 时必须像用户一样实际运行入口脚本，
    不能直接运行它内部的 `.py` 绕过入口；自动化调用加 `nopause`。
19. **改完代码必须检查**：修改代码后至少做语法检查（`py_compile` / `ast.parse`）和配置检查
    （`--check-config`）；有构建/测试脚本的按脚本走，但不执行训练、启动 WebUI 等有副作用的操作。
20. **写 git 提交信息前必须先读取 git 差异**：用户要求写提交信息时，禁止凭记忆、上一轮结论
    或旧提交信息直接编写；必须先执行一次只读 git 命令（建议组合
    `git status --short --untracked-files=all`、`git diff --stat`、`git diff --name-status`，
    必要时只读 `git diff -- <文件>`）确认本次实际改动，再按真实差异撰写。提交信息写入
    `.ai_res/git_log/<YYYY-MM-DD_HH-MM-SS>.txt`。命名前必须先查询当前实际时间，并按查询结果
    命名，禁止凭直觉或估算写文件名。提交信息由用户提交；AI 不执行 `git add` / `commit` /
    `push`；只写本仓库的改动，和本仓库无关或 git 忽略的内容不要写进 git 提交信息。
21. **git 提交信息只写本次改动本身**：不得加入“未执行 git add / commit / push”
    “提交信息由用户提交”、验证或测试结论、工具执行过程等与改动无关的说明文字；
    提交信息只描述代码、配置和文档的实际变化。
22. **第三方版本与许可证声明必须跟着实际文件走**：修改 `.runtime/python`、`web/vendor/`、
    `THIRD-PARTY-NOTICES.txt` 或任何随包第三方组件的版本时，必须同时核对并更新版本号、
    许可证标识（如 MIT / Apache-2.0 / MPL-2.0 / OFL-1.1 / PSF-2.0）、本地许可证文件路径和
    版权声明，并同步 `web/index.html` 的「版权鸣谢」表与 README / AGENTS 中的相关说明。
    Python 版本以 `.runtime/python/VERSION.txt` 和 `python --version` 为准，
    前端版本以 bundle header 或官方包元数据为准，版权文字以随包 LICENSE 原文为准；
    禁止凭记忆写版本，禁止保留过期版本号或错误版权。训练依赖由用户安装且不随包分发，
    不写入随包声明；新增随包第三方组件必须同步补通知。

## 二、目录结构（当前架构）

```
除 .runtime\ 的启动器和随包解释器外，其余都在 code\ 与 web\ 里。
.runtime/               本地运行时与启动器（python/ 被 gitignore）
  start.bat             双击入口：固定用 .runtime\python\python.exe 运行 code\webui.py
  make_python_env.bat   重建 .runtime\python；可输入系统 Python 路径
  make_python_env.py
  logo.ico              快捷方式图标
  python/               项目 Python：标准库 + ensurepip（构建产物，不进版本库）
code/                   所有 Python
  webui.py              本地 Web 服务：静态页 + JSON API + SSE + 子进程管理
  env_install.py        用项目 Python 安装训练 / 对话依赖
  train.py              训练（数据管线 + 模型 + 训练循环 + 采样）
  chat.py               对话核心：ChatSession（load / reply_stream / set_params）
  chat_worker.py        对话子进程：常驻 ChatSession，stdin/stdout 走 JSON 行
  http_server.py        OpenAI 兼容桥接服务（本地模型 + API 代理）
  ollama_client.py      Ollama 原生元数据辅助（只用于 API 配置的能力补齐）
  openai_client.py      OpenAI 兼容客户端（Base URL、Key、SSE、tools、embeddings）
  mcp_client.py         MCP 客户端（STDIO / Streamable HTTP、工具 schema、tools/call）
  common.py             find_root / properties 保序读写 / 路径解析 / 父进程看门狗 / 隐藏目录
  model.py              GPT-2 封装（transformers）
  model_signature.py    权重 SHA-256 签名（compute / verify，纯标准库）
web/                    前端（被 webui.py 托管，不需要单独服务器）
  index.html            十个页签：训练 / 对话 / Agent / API / 服务 / 环境 / 配置 / 签名 / MCP / 版权鸣谢
  chat.html             局域网只读对话页（LANHandler 托管）
  vendor/               本地第三方前端资源：xterm.js / marked / DOMPurify / KaTeX
properties/             四个训练配置（要开源，注释写给外人看）
model/                  本地权重（model\<名字>\ 标准 HF 目录 + checkpoint-<step>\）
res/                    语料目录（不进版本库）
.cache/                 训练缓存/日志/配置备份/MCP 与 API 配置、检测结果、工作区注册表、会话、输入区附件（内容忽略；Windows 下自动隐藏）
THIRD-PARTY-NOTICES.txt  Python / 依赖 / 前端第三方库的许可证清单（人工维护）
```

- 工程根目录靠「向上找到含 `code\train.py` 的那一级」定位（`common.find_root`），脚本搬家不失效。
- 模型 id = 目录名（`model\chat_demo\` → `chat_demo`）。
- 续训目录 `model\<名字>\checkpoint-<step>\`：`optimizer.pt` / `scheduler.pt` /
  `trainer_state.json` / `rng_state.pth`，root 只放最佳模型。

## 三、启动链路与进程模型

```
.runtime\start.bat
  └─ .runtime\python\python.exe code\webui.py    ← 前台运行；标准库 + ensurepip，不 import torch
       ├─ 子进程 .runtime\python\python.exe code\train.py --config <cfg> --parent-pid <webui_pid>
       ├─ 子进程 .runtime\python\python.exe code\chat_worker.py --config <cfg> --parent-pid <webui_pid>
       ├─ 子进程 .runtime\python\python.exe code\http_server.py --port <port> --events --parent-pid <webui_pid>
       ├─ 子进程 .runtime\python\python.exe code\env_install.py ...（环境页安装依赖）
       └─ 子进程 .runtime\python\python.exe code\model_signature.py --mode compute|verify --parent-pid <webui_pid>
```

- **没有面向用户的 CLI**：文档不介绍命令行用法；`--config` / `--parent-pid` / `--events` 等
  参数是 WebUI 与子进程之间的**内部协议**，不承诺兼容。
- **解释器统一**：WebUI、训练、对话、桥接服务、签名和环境安装全部使用
  `.runtime\python\python.exe`；`common.resolve_python()` 不再读取 `properties\python=`，
  也不回退 PATH，缺失时直接提示运行 `make_python_env.bat`。
- **依赖安装**：随包 Python 只带标准库 + `ensurepip`；训练 / 对话依赖由「环境」页调用
  `code\env_install.py` 安装到同一个项目 Python，不引导用户另装系统 Python。
  安装表单按 PyTorch / Numpy / transformers 三行填写；PyTorch 的芯片类型下拉框来自实际可用的
  硬件后端（CPU / GPU NVIDIA / AMD / Apple / XPU 等），驱动版本输入框填数字版本
  （CUDA 12.6 / ROCm 6.2），Torch 库版本单独填写。`cu126` 等只做内部映射。
  检测环境只负责芯片类型/驱动版本；同时读取当前已安装版本填入版本框，
  安装时跳过已安装且版本相同的库。版本输入框留空即安装最新版。
  WebUI 启动时检测一次 Python 和 API 模型；之后训练、对话、服务启动前不重复自动检测，
  只有手动【检测模型】或环境页【检测环境】才重新检测。Ollama 作为 API 配置放在 API 页。
- **生命周期**：关掉 `start.bat` 的 cmd 窗口 = 退出（先 `terminate()` 子进程，5 秒不退再
  `taskkill /F /T`）；**关浏览器标签不算退出**，训练照常跑，重新访问控制台地址即可。
- **看门狗**：子进程按 `--parent-pid` 每 3 秒用 `ctypes` `WaitForSingleObject` 检查父进程句柄，
  父进程一消失就 `os._exit(1)`（防 webui 被强杀留下孤儿）。
- **Job Object**：子进程还会加入一个 `KILL_ON_JOB_CLOSE` 的 Job，WebUI 进程一消失，内核
  直接杀掉 job 里的全部进程；cmd 窗口被强制关闭时也能兜住，不依赖看门狗轮询。
- **端口**：控制台默认 `127.0.0.1:49134`；被占用时**不自动换端口**，提示输入新端口，
  直接回车退出（`--port` 指定时同样处理）；局域网只读对话页默认 `0.0.0.0:49135`
  （`--lan-port` 改端口，`--no-lan` 关掉；占用时仅警告继续）。
- **安全**：控制台只绑 `127.0.0.1`，每次启动生成一次性 token，`/api/*` 全部校验；
  局域网端口不校验 token，但只放行对话页 + `/api/models` + `POST /api/chat`。

## 四、WebUI 接口约定（改接口必须同步 `web/index.html`）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/env` | 查询/检测项目 Python（`check=1` 才 probe；返回 torch/numpy/transformers/pip/CUDA，以及 `system.torch_recommend` 显卡/驱动推荐） |
| GET | `/api/env/latest` | 查询 PyPI / PyTorch 源的最新 torch、numpy、transformers 版本，并返回实际可用的 PyTorch 构建列表；网络失败时回退到已安装/推荐版本 |
| GET | `/api/env/ollama` | 兼容旧接口：检测 Ollama 地址、版本和模型数量（主流程已改用 API 配置） |
| GET | `/api/configs` | 配置列表 + 解析后的键值 |
| GET | `/api/models` | `model\` 下的标准模型目录清单（含是否已签名） |
| GET | `/api/chat/configs` | 对话下拉用：本地配置 + OpenAI 兼容 API 的对话模型（带 backend 标记） |
| GET | `/api/service/configs` | 服务页下拉用：本地配置 + OpenAI 兼容 API 的对话 / 嵌入模型 |
| GET | `/api/mcp/configs` | MCP 服务器配置列表（STDIO / HTTP）+ 上次检测结果 |
| POST | `/api/mcp/configs` | 保存 MCP 服务器配置到 `.cache/mcp_servers.json` |
| POST | `/api/mcp/test` | 连接 MCP 服务器并返回 tools/list |
| POST | `/api/mcp/detect` | 启动时/手动检测全部 MCP 服务器，结果写入 `.cache/mcp_status.json` |
| GET | `/api/openai/configs` | OpenAI 兼容 API 配置 + 模型、工具、思考能力检测结果（不返回 Key） |
| POST | `/api/openai/configs` | 保存 API 配置到 `.cache/openai_apis.json`，并立即检测模型 |
| POST | `/api/openai/detect` | 启动时检测全部已保存 API 的 `/models`，结果写入 `.cache/openai_status.json` |
| POST | `/api/openai/test` | 只检测当前选中的 API（草稿也可）；已保存配置同时更新状态并返回模型/思考能力 |
| POST | `/api/openai/model` | 保存单个模型的手工能力覆盖（上下文、输出、工具、输入类型、思考档位） |
| GET | `/api/sessions` | 会话列表；`?id=` 返回单个会话；`?id=&history_page=1&history_turns=5&history_before=<n>` 只返回最近 5 个用户轮次及 `history_total` / `history_start` / `history_has_more` / 用户轮次 `history_outline`（也兼容旧的 `history_limit=<消息条数>`）；`?kind=chat\|agent&workspace=` 过滤 |
| POST | `/api/sessions` | 保存会话（历史、模型、MCP 总开关与多服务器选择、工具、思考等级、Agent 工作区）；带 `history_start` 时与旧历史前缀合并，支持懒加载分页；`mcp_servers` 是启用服务器 id 列表，`mcp_tools` 按服务器保存工具勾选；assistant 历史项可带仅 UI 使用的 `steps`（上下文注入/思考/工具/回答分段），发送模型前会被剥离 |
| POST | `/api/sessions/delete` | 删除会话（同时清理该会话的附件缓存目录） |
| POST | `/api/sessions/rename` | 会话改名 |
| POST | `/api/sessions/move` | 把 Agent 会话移动到另一个工作区 |
| POST | `/api/attachments` | 输入区粘贴/选择的 `.txt` / `.md` 文本写入 `.cache/attachments/<会话 ID>/<uuid>.<ext>`，返回缓存文件的绝对路径、名称和大小；单文件上限 1MB |
| GET | `/api/workspaces` | Agent 工作区注册表（路径、标题、默认工作区） |
| POST | `/api/workspaces` | `action=add` 添加绝对路径工作区；`rename` 改名；`delete` 删除 |
| POST | `/api/workspaces/pick` | 弹出本机原生目录选择窗口，返回绝对路径 |
| GET | `/api/workspace` | 兼容旧接口：默认工作区 ID、标题和最近会话 |
| POST | `/api/mcp/restart` | 手动重连指定 MCP 服务器并刷新工具列表 |
| GET | `/api/agent/configs` | Agent 可用的工具模型 + MCP 服务器 + 检测结果 |
| GET | `/api/agent/tools?server=<id>` | 获取某个 MCP 服务器的工具（供 Agent 多选） |
| POST | `/api/agent/chat` | Agent 工具循环，SSE：`meta` / `inject` / `context` / `reasoning` / `tool_pending` / `tool` / `tool_result` / `token` / `done` / `error`；API 模型的 `done.usage`/`done.session_usage` 带本轮和会话累计统计，`compact=true` 时手动压缩；`servers` 为本次启用的多台 MCP 服务器 id 列表（旧前端只传 `server`），`mcp_enabled=false` 时不发送任何服务器和 `tools`；`meta.servers` / `meta.tool_errors` 返回服务器名和加载失败信息；`inject.items` 只在本轮真正追加 AGENTS.md / runtime-context / 时钟快照时发送，描述新增的上下文内容和“首次注入/已更新/新快照”标签 |
| GET | `/api/corpora` | `res\` 下的语料目录（文件数、体积） |
| GET | `/api/train/status` `train/log` | 训练状态与增量日志 |
| GET | `/api/server/status` | 桥接服务状态与端口 |
| GET | `/api/events` | SSE：`train-raw` / `server-log` / `server-chat` / `server-embedding` / `env-log`；`train-raw` 保留 `\r`，由 xterm 渲染 |
| POST | `/api/config/save` | 逐行改 value 保存（保注释与键顺序），先备份 |
| POST | `/api/env/install` | 用项目 Python 安装 torch / numpy / transformers；`torch_type=cpu|gpu|xpu|auto`，`torch_version` / `numpy_version` / `transformers_version` 可留空；兼容旧 `torch_index=cpu|cu126|cu128|xpu|auto`；输出走 `env-log` |
| POST | `/api/train/start` `train/stop` | 启动 / 停止训练子进程 |
| POST | `/api/chat` | 对话，返回 SSE：`context` / `meta` / `reasoning` / `token` / `done` / `error`；API 模型的 `done.usage`/`done.session_usage` 带本轮和会话累计统计 |
| POST | `/api/server/start` `server/stop` | 启停桥接服务 |
| POST | `/api/sign` | `mode=compute|verify`，跑签名工具 |

- 桥接服务每次完成对话会往 stdout 打一行 `@@EV@@ {json}`（`--events`），
  webui 读线程按前缀识别后转成 SSE `server-chat`，页面「服务」页据此显示问答；
  embedding 成功时转成 `server-embedding`，在「向量」子页显示摘要。
- SSE 用 `data: <json>\n\n`，一条连接一个线程；新增长连接接口要考虑线程上限。
- **局域网端口**（`LANHandler`）只放行 `GET /`、`/chat.html`、`/logo.ico`、
  `/vendor/*`、`/api/models`、`/api/chat/configs` 与 `POST /api/chat`，其余路径一律 403；
  `POST /api/chat` 的 `config` 必须是配置清单里的名字。改局域网接口时同步 `web/chat.html`。

## 五、配置文件（properties）约定

- **语法**：一行一个 `key=value`，`#` 开头是整行注释。
- **禁止行尾注释**：解析器按 `split("=", 1)` 取值，行尾注释会被当成值。
- **四个配置保持同构**：`chat_demo` / `dlqq_exam` / `dlqq_llm` / `dlqq_poem` 键集合、顺序、分节一致。
- **只能写 `CONFIG` 里已有的键**：`code/train.py` 的 `CONFIG` 是唯一来源。
- **保存规则**：一律走 `common.save_properties`（只改 value，保留注释/空行/顺序），
  保存前按文首说明的临时目录约定备份。
- **不再有 `python=` 键**：解释器固定为 `.runtime\python\python.exe`，旧配置里遗留的
  `python=` 只会被忽略，不参与任何解析。
- **续训由界面决定**：训练页【继续训练】按钮会传 `resume=true`，配置里不再有 `resume` 键。

| 键 | 含义 |
|---|---|
| `data_path` | 语料**目录**（相对工程根） |
| `model` | 模型名，自动落到 `model\<名字>\` 标准 HF 目录 |
| `cache_dir` | 切词缓存目录（默认 `.cache`） |
| `chat_mode` / `chat_temperature` / `chat_top_k` / `chat_max_new` | 对话参数，加载时读取 |
| `n_layer` / `n_head` / `n_embd` / `block_size` / `batch_size` | 结构与每步 token 数 |
| `max_iters` / `eval_interval` / `eval_iters` | 步数与评估节奏 |
| `lr` / `warmup_iters` / `lr_decay` / `min_lr_frac` / `weight_decay` / `grad_clip` / `dropout` / `seed` | 优化 |
| `grad_accum` / `grad_checkpoint` | 等效 batch / 省显存 |
| `save_every` / `keep_best_checkpoints` / `keep_forced_checkpoints` / `save_dtype` | 定时存档 / 最优存档上限 / 强制存档上限 / 存档精度 |
| `max_vocab` / `val_frac` | 词表上限 / 尾部验证集比例 |

## 六、运行时行为（改代码前先理解）

**数据管线**：分块读（4M 字）→ `np.bincount` 统计 → 取前 `max_vocab` → `searchsorted` 编码 uint16
写缓存 → 训练用 `np.memmap` 随机读，内存与语料大小无关。
缓存指纹 = 源绝对路径 + 每个文件的（相对路径、大小、mtime、格式、问答字段）+ `max_vocab` + `val_frac`；
**改这些会让缓存失效**（15 GB 语料约 5 分钟重切）。

**验证集**从语料**尾部**切；按类别排序的语料要先打乱。

**上下文自动缩水**：`_safe_T = min(训练 token, 验证 token) - 2`，`block_size` 更大时会被自动调小。

**保存策略**：验证损失变好时更新 `model\<名字>\` 最佳模型，并写一个
`checkpoint-<step>\`；`save_every > 0` 时额外写 checkpoint。
checkpoint 始终保留 `optimizer.pt` / `scheduler.pt` / `rng_state.pth`，可续训；
`keep_best_checkpoints` / `keep_forced_checkpoints` 分别限制最近 N 个最优/强制
checkpoint（0 = 不保留，-1 = 不限制）。
root 目录始终只有 `config.json` / `model.safetensors` / `chars.json` / `meta.json`，
不含优化器，可直接发布。

**对话参数优先级**：`显式参数` > `properties\<配置名>.properties` > `模型文件里存的` > `默认值`
（网页对话与桥接服务都按这个顺序取，`chat.py` 与 `http_server.py` 各实现一份，改一处要同步另一处）。

**API（OpenAI 兼容，含 Ollama）**：`openai_client.py` 只用标准库实现 chat completions、models 和
embeddings 请求。配置保存在 `.cache/openai_apis.json`，只保存 `name` / `base_url` / `api_key_env`；
`api_key_env` 可为空，空值时不发送 Authorization，供 Ollama 等本地服务使用。
模型不由配置指定：WebUI 启动时和 API 页手动检测都会请求 `/models`（不消耗生成 Token），
Ollama 配置还会用 `/api/show` 补齐类型、工具、思考和上下文。每个模型用
`api:<配置id>/<模型>` 作为 id，类型分为 `chat` / `embedding`；Chat/Agent 只用 chat，
服务页两者都列。API 模型由桥接服务代理 `/v1/chat/completions` 和 `/v1/embeddings`。
Ollama 配置的聊天超时为 600 秒，其他供应商为 300 秒；Ollama 长时间思考不应被 300 秒误断开。

模型能力采用“自动检测 + 手工覆盖”：

- 上下文、最大输出：检测不到时界面显示“未知（供应商默认）”，内部使用保守预算；
- 工具能力：`tool_use` 为 `native` / `none` / `unknown`，不再按模型名猜；
  Agent 允许 `native` 和 `unknown`，只隐藏明确 `none`；
- 输入类型：`input_modalities` 至少 `text`，可增加 `image`；
- 思考：`reasoning_efforts` 是“档位 → API 参数值”的映射，`off` 为 `null`；
  界面另提供 `default`（Default，不传任何思考参数），与显式 `off` 区分；
  未知模型不再伪造四档。

API 页点击模型行可手工覆盖上述字段，保存在 `.cache/openai_model_overrides.json`，
自动检测不会覆盖手工值。上下文上限从 `/models` 的 `context_length` / `context_window` /
`max_model_len` 等字段读取。API 请求会按上下文预留输出空间后压缩历史：使用 DSH 风格
80% 阈值和 16% 近期保留预算，把旧历史替换为检查点摘要；摘要请求复用原 system prompt、
tools 和旧消息前缀，压缩指令追加在末尾，尽量复用 provider 热前缀。压缩前会修剪超大工具
结果；摘要失败时退回按工具调用组裁剪，`tool_calls` / `tool_result` 必须成对保留。上下文
溢出错误会强制压缩并重试一次。Agent 输入框支持 `/compact`，也可从输入框【＋】→「压缩上下文」
执行同样的手动压缩。
Agent 的 `max_tokens` 默认 256K（有模型输出字段时用模型值，并受上下文限制），它不是上下文上限；
单轮输出上限和上下文窗口分开计算。Chat / Agent 输入框下方的状态栏显示按字符数估算的上下文
占用率圆环、缓存命中率，以及本轮和会话的输入/输出 token；点击任一项弹出详情，输出上限不再
默认平铺。图片按固定视觉 token 估算，不按 base64 字符数计费。流式回答按约 100ms 节流解析
Markdown（先不跑 KaTeX，公式显示为 `$...$` 原文），结束后再做一次完整渲染。API 模型完成一轮后
显示 provider 上报的
本轮和会话累计 token（输入、输出、缓存命中/未命中）及命中率，公式为
缓存读 /（未缓存输入 + 缓存读 + 缓存写）；兼容 DeepSeek
`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 和 OpenAI
`prompt_tokens_details.cached_tokens`，Agent 多步按 token 加权；无缓存字段显示“—”，
本地模型和没有缓存统计的 Ollama 不显示。会话累计写入会话 JSON 的 `usage_totals`，重新
打开会话后仍保留。流式请求对非 Ollama API 尝试
`stream_options.include_usage`，代理不支持时自动去掉重试一次。

**MCP / Agent**：`mcp_client.py` 只用标准库实现 STDIO（一行一个 JSON）和 Streamable HTTP
（JSON-RPC POST + Mcp-Session-Id）。已配置的 MCP 客户端在 WebUI 启动时创建并常驻，STDIO 进程
生命周期跟随 WebUI；修改配置时只重启对应服务器，WebUI 退出时统一关闭。后台监控每 2 秒检查
STDIO 进程，HTTP 每 30 秒做一次 tools/list 健康检查；断连后按 500ms → 30s 指数退避，
最多连续 10 次，稳定连接后重置计数。重连成功后重新拉取工具列表，超过次数标记 `failed`
并停止自动重连，可在 MCP 页手动【重连】。已发出的非幂等工具不会自动重放。
Agent 允许 `tool_use` 为 `native` 或 `unknown` 的 API 模型，只隐藏明确 `none` 的模型；
MCP 是多服务器模型：会话里 `mcp_enabled` 是总开关，`mcp_servers` 保存启用服务器 id 列表，
`mcp_tools` 按服务器保存工具勾选；前端浮层可同时启用 A、关闭 B，并用「工具来源」下拉
选择当前编辑哪台服务器的工具。发送时把启用服务器列表和合并后的工具名发给 `/api/agent/chat`，
后端从各服务器拉取工具并在调用时按工具名路由回对应服务器；多台服务器工具重名时保留
先启用的那台。总开关关闭或某台服务器关闭时，其工具不发送、不执行，MCP 进程仍保持常驻；
开启但一个工具都没勾选时同样不发送 `tools`，两种状态分开保存。切换服务器/工具后立即保存；
即使会话还没有消息，也会保存一个空会话（标题“新会话”）用于刷新后恢复 MCP 设置。
首次进入 Agent 时先建立草稿会话，并按已加载的 MCP 配置同步总开关和工具数量；
MCP 按钮必须一开始就显示正确状态，不能依赖点开 MCP 浮层后才刷新。
带工具时按
`tool_calls → tools/call → tool result` 循环，直到模型不再返回工具调用；
不再使用固定 8 轮上限，用户可随时停止；
每轮使用 SSE，最终回答逐 token 推给前端；顶部显示运行状态和用时，当前步骤状态
（如“正在调用 …”“… 已完成，继续生成…”）紧跟“运行中 · N 秒”后面；完成/停止/失败后清空，
输入区不再单独显示“完成”状态。默认折叠上下文注入/思考/工具调用历史，回答和会导致会话中断的
报错始终显示；点击顶部展开历史，单条参数/结果再自行展开。
上下文注入、思考、工具调用/结果和回答分段写入会话 `history` 条目的 UI-only `steps` 字段，
刷新或重开会话时由前端重建；发送模型前 `_strip_ui_messages` 会剥离 `steps`，模型上下文仍只
使用 `model_history`，不做任何改变。
只有本轮真正追加了 `AGENTS.md` / runtime-context / 时钟快照时，才发送 `inject` 事件并在
时间线显示「上下文注入」行：首次请求显示 baseline，后续只在内容变化/时钟刷新时显示新快照；
没有变化时完全不显示。`system` 提示词和 `tools` 是每次请求都携带的请求参数，不作为
“重新注入”展示。展开可看实际追加内容，标签区分“首次注入 / 已更新 / 新快照”。
思考、工具调用、回答按模型实际输出顺序作为平级时间线行展示。
思考行折叠时在“思考”后显示内容的最后一句话，展开后隐藏该摘要并显示完整思考。
工具结果按 `dlqq-mcp-server` 的返回类型美化：文件读取为行号代码、目录树、搜索结果卡片、
文本编辑 diff、JSON/键值、命令输出等；每张工具卡片仍保留原始返回。工具参数和结果完整回填
给模型，不做硬截断；只执行本次勾选的工具，模型返回未勾选或不存在的工具时回填
“工具不存在”，不调用 MCP。每条思考、回答和工具调用均显示浏览器所在系统本地时区的
`YYYY-MM-DD HH:mm:ss.SSS` 时间戳。
`thinking` 支持 `default/off/low/medium/high` 等档位，按 `reasoning_efforts` 映射到实际 API
参数值；`default` 表示不发送任何思考参数，`off` 才是显式关闭；未知模型不伪造档位，
本地字符级模型不支持该控件。DeepSeek 默认开启思考。对 DeepSeek（按 API base_url/name 识别）
选择 `off` 时必须显式发送 `thinking: {"type": "disabled"}`，选择非 off/default 档位时发送
`thinking: enabled` 加映射后的 `reasoning_effort`；`default` 不发送 thinking，
让 DeepSeek 使用自身默认（开启）思考。不能只省略 `reasoning_effort` 来表示关闭。
工具结果之后会补一条内部续跑 user 提示，避免 Ollama/Qwen 把纯 tool 轮次判成
“no user query found in messages”。服务器定义保存在 `.cache/mcp_servers.json`；启动时检测
全部服务器并写入 `.cache/mcp_status.json`。
每次 `tools/call` 会把会话 ID、Agent ID 和**会话所属工作区**的绝对路径写入 MCP
`params._meta`，键名分别为 `dsh/sessionId`、`dsh/agentId`、`dsh/workspace`；
服务端可据此按会话隔离工作区。`mcp_client.MCPClient.call_tool(name, args, meta=...)`
负责把 `_meta` 放进请求参数。

**会话 / 工作区 / 输入区**：Session 存在 `.cache/session-<id>.json`；Agent 工作区注册表存在
`.cache/workspaces.json`，默认工作区是 `<工程根目录>\.runtime\default_workspace`，可添加其他
绝对路径；Agent 左侧是「工作区 → 会话」树，每个工作区下只显示自己的会话，一个工作区可保存
多个会话。新建 Agent 会话默认使用当前工作区，未指定时使用默认工作区。会话保存时记录
`workspace_id`；同时保存完整的 `model_history`（含 system、tool_calls、tool 结果和最终回答），
下一轮请求以它为前缀追加新用户消息，避免丢失工具消息导致 prompt cache 中断。首次请求注入
工作区指令 baseline（根目录 `AGENTS.md` / `CLAUDE.md`，有上限）和 runtime-context 快照
（工作区绝对路径、进程 CWD、时区时钟）；后续按 DSH 方式比较后追加：指令内容或 runtime 事实
变化时才追加一条替换旧快照的 user 消息，未变化不重复注入；工具读取/修改文件后重新发现
工作区根到该文件之间的嵌套 `AGENTS.md` / `CLAUDE.md`。时钟快照按 step 检查，每 10 分钟最多
追加一次（工具续跑步骤也可以追加）。会话 JSON 记录 `context_snapshot`、
`instruction_snapshot` 和 `instruction_paths`，前端保存会话时不能覆盖这些字段。
system prompt 包含按名称稳定排序的工具指导。
Agent 调用 MCP 工具时以**会话所属工作区**的绝对路径优先，而不是页面临时选择。
工作区路径选择通过 `/api/workspaces/pick` 调用本机原生目录窗口。会话支持改名、删除和
移动工作区（`/api/sessions/move`），移动后每次 MCP 调用仍读取会话当前 `workspace_id`；
每轮结束自动保存；恢复时恢复历史、模型、MCP、工具和思考选择。Chat / Agent 侧边栏支持收起，
并可拖动右边缘调整宽度，状态保存在浏览器 localStorage。未完成的模型/工具轮次不自动重放。
Chat / Agent 的多个会话可并行执行：前端按会话保存独立的 DOM、历史、usage、上下文和
AbortController，切换会话不中止后台流；同一会话只允许一个生成任务，停止按钮只停止当前会话。
会话打开时默认滚到最新消息，只加载最近 5 个**用户轮次**（不是 20 条消息）；只有用户主动
向上滚动且 `scrollTop` 变小并接近顶部时才按 `history_before` 补更早一页。程序化滚动
（打开会话、补页后保持位置、轮次跳转）用 `programmaticScroll` 锁住，不会自动补页；
初始页不足一屏时保留【加载更早消息】按钮。保存时用 `history_start` 合并旧前缀。
右侧是 DSH 风格居中紧凑轮次刻度：固定间距、内部滚动、当前轮高亮，可跳转，目标未加载时先补页。
用户消息在发送时写入 `ts`（本地完整时间），会话 JSON 持久化；发给模型前由后端剥离界面字段。
输入区不设“新对话”按钮，统一由侧边栏或收起后的侧栏【＋】新建会话。输入框左侧【＋】菜单提供
图片、`.txt`/`.md` 文本文件和指令；模型与思考等级合并为一个按钮；MCP 只占一个按钮和浮层。
文本文件先经 `POST /api/attachments` 写入 `.cache/attachments/<会话 ID>/`，发送时把缓存绝对路径
以 `[附件] ...` 行追加到用户消息末尾，会话删除时清理对应目录，并在上传时顺带清理超过 7 天的
残留目录。附件路径只是字符串：Chat API 没有文件工具，Agent 的文件工具也可能因 `.cache` 不在
工作区内而拒绝访问。输入框下方状态栏显示上下文圆环、缓存命中率和本轮/会话 token。
模型 `input_modalities` 含 `image` 时【＋】菜单可选图片，最多 4 张、单张 6MB；也支持在
Chat / Agent 输入框直接粘贴剪贴板图片（`paste` 事件的 `clipboardData.items` / `files`），
与文件选择共用 `addImages` 校验、预览和 DataURL 发送链路。图片以 OpenAI
兼容 `image_url` 内容块发送，本地字符级模型拒绝。图片 token 按固定视觉 token 估算，不按
base64 字符数计费。聊天消息里的图片和输入框待发送图片预览都支持点击放大：事件委托监听
`.msgs img, .imagepreviews img`，打开 `#imageViewer` 遮罩层，点击遮罩、关闭按钮或按 Esc 关闭；
用户图片缩略图、Markdown 图片和待发送预览都适用。
输入区主按钮同一位置两态：空闲显示 `↑` 发送，生成中显示 `■` 停止；
图片/文件读取中或提交中禁用。

**生成停止**：提示词用 `用户：` / `助手：`；生成到下一个 `用户：` 直接截断；词表缺这两个标记时
自动切「续写模式」。

**训练中对话**：只允许换**不同**的模型；与训练配置同名的模型会被拒绝（拖慢且可能读到半成品）。

## 七、代码风格

- **文件头**：`# -*- coding: utf-8 -*-` + `r"""..."""`（用途、用法、注意事项）。
- **编码兜底**：所有入口先 `sys.stdout/stderr.reconfigure(encoding="utf-8", errors="replace")`
  或调 `common.setup_stdio()`。
- **注释用中文**，解释「为什么」，不复述代码；参数取值的依据写进注释。
- **失败即退出**：参数/环境/数据不合法时 `sys.exit(1)`，并给可执行的修复提示；不做兜底猜测。
- **不吞异常**：`except` 写具体类型；要么打印要么重抛。
- **打印格式**：`[标签] 内容`，标签用 `[数据]` / `[训练]` / `[设备]` / `[错误]` / `[警告]`；
  训练循环里带 `flush=True`。
- **命名**：函数/变量 `snake_case`，常量 `UPPER_SNAKE_CASE`，类 `CamelCase`；配置键与 `CONFIG` 一致。
- **不用类型注解**：与现有代码保持一致。
- **不引入日志框架**：`print(..., flush=True)`，或交给 webui 的日志队列。
- **模型唯一定义**：`code/model.py` 封装 `transformers.GPT2LMHeadModel`；`train.py` / `chat.py` /
  `http_server.py` 都从这里 import，不再各自复制网络定义。训练必须随机初始化，禁止加载任何第三方预训练权重。

## 八、已知限制

- **旧权重不兼容**：换成 GPT-2 后，旧版手写 GPT 的 `.pt` 参数名不同，不能直接加载；
  需要用本项目语料重新训练。
- **字符级 + 短上下文**：`block_size` 256 时一条长消息只能看到最后一段。
- **小模型会背语料**：同题复现能力强，换问法就露馅；不要拿训练题报分数。
- **桥接服务不是 agent**：不调用工具、不规划，只适合验证链路。
- **Agent 工具过多**：一次勾选 100+ 个 MCP 工具会占用大量上下文，并增加模型编造工具名或
  生成被截断 tool_call 的概率；创建大文件时应只选相关文件工具，并让模型分块写入。
- **WebUI 控制台只对本机**：控制台绑 `127.0.0.1` + 一次性 token；局域网只暴露只读对话页
  （`0.0.0.0:49135`，无鉴权，同一内网的人都能聊）。
- **CPU 训练**很慢且不启用 bf16，只建议小模型跑通流程。

## 九、参考资料

- 权重仓库：[ModelScope · donglouqiaoque/dlqq-llm-model](https://www.modelscope.cn/donglouqiaoque/dlqq-llm-model)
- 源码镜像：[GitHub](https://github.com/donglouqiaoque/dlqq-llm) ｜ [Gitee](https://gitee.com/donglouqiaoque/dlqq-llm)
- PyTorch 文档：<https://pytorch.org/docs/stable/index.html>
- nanoGPT：<https://github.com/karpathy/nanoGPT>
