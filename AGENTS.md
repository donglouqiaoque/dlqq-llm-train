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
   **唯一允许的第三方前端库是本地 `web/vendor/xterm.js`（MIT）**，用于训练日志终端；
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
3. **git 权限分三层，禁止放宽**：写操作（`commit` / `push` / `add` / `reset` / `checkout` 等）
   **无论用户是否授权都绝对禁止**；`.git` 目录**无论用户是否授权都绝对禁止读写**，无论是标准
   编辑工具（读/写/编辑）还是执行脚本或命令，一律禁止读写 `.git` 目录下的任何文件（读也不行，
   防止把只读命令写成写/删命令）；只有 `status` / `diff` / `log` 等只读命令允许在用户**主动说明**
   时授权一次，用完立即停止，后续禁止继续使用，除非用户再次主动说明，不存在“用户默许”“用户之前
   说过”之类的理由。
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
  http_server.py        OpenAI 兼容桥接服务（ThreadingHTTPServer，多模型懒加载）
  common.py             find_root / properties 保序读写 / 路径解析 / 父进程看门狗 / 隐藏目录
  model.py              GPT-2 封装（transformers）
  model_signature.py    权重 SHA-256 签名（compute / verify，纯标准库）
web/                    前端（被 webui.py 托管，不需要单独服务器）
  index.html            六个页签：训练 / 对话 / 服务 / 环境 / 配置 / 签名
  chat.html             局域网只读对话页（LANHandler 托管）
  vendor/               本地第三方前端资源：xterm.js / xterm.css / LICENSE.xterm
properties/             四个训练配置（要开源，注释写给外人看）
model/                  本地权重（model\<名字>\ 标准 HF 目录 + checkpoint-<step>\）
res/                    语料目录（不进版本库）
.cache/                 训练缓存/日志/配置备份（目录进版本库，内容忽略）
THIRD-PARTY-NOTICES.txt  Python / 依赖 / xterm 的许可证清单（人工维护）
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
  `torch_version` 可写 `2.14.1+cu126` / `2.14.1+cpu`，自动检测会把推荐值填入文本框；
  用户可手动修改版本和源后缀。
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
| GET | `/api/configs` | 配置列表 + 解析后的键值 |
| GET | `/api/models` | `model\` 下的标准模型目录清单（含是否已签名） |
| GET | `/api/chat/configs` | 对话下拉用：能对上权重的配置名 + 模型名 |
| GET | `/api/corpora` | `res\` 下的语料目录（文件数、体积） |
| GET | `/api/train/status` `train/log` | 训练状态与增量日志 |
| GET | `/api/server/status` | 桥接服务状态与端口 |
| GET | `/api/events` | SSE：`train-raw` / `server-log` / `server-chat` / `env-log`；`train-raw` 保留 `\r`，由 xterm 渲染 |
| POST | `/api/config/save` | 逐行改 value 保存（保注释与键顺序），先备份 |
| POST | `/api/env/install` | 用项目 Python 安装 torch / numpy / transformers；`torch_version` 可写 `2.14.1+cu126` / `2.14.1+cpu`；`torch_index=cpu|cu126|cu128|auto` 兼容旧调用；输出走 `env-log` |
| POST | `/api/train/start` `train/stop` | 启动 / 停止训练子进程 |
| POST | `/api/chat` | 对话，返回 SSE：`meta` / `token` / `done` / `error` |
| POST | `/api/server/start` `server/stop` | 启停桥接服务 |
| POST | `/api/sign` | `mode=compute|verify`，跑签名工具 |

- 桥接服务每次完成对话会往 stdout 打一行 `@@EV@@ {json}`（`--events`），
  webui 读线程按前缀识别后转成 SSE `server-chat`，页面「服务」页据此显示问答。
- SSE 用 `data: <json>\n\n`，一条连接一个线程；新增长连接接口要考虑线程上限。
- **局域网端口**（`LANHandler`）只放行 `GET /`、`/chat.html`、`/logo.ico`、`/api/models`、
  `/api/chat/configs` 与 `POST /api/chat`，其余路径一律 403；`POST /api/chat` 的 `config`
  必须是配置清单里的名字。改局域网接口时同步 `web/chat.html`。

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
| `save_every` / `save_dtype` / `save_optimizer` | 定时存档 / 存档精度 / 训练完成后是否保留优化器 |
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
`save_optimizer=true`（默认）让 checkpoint 保留 `optimizer.pt` / `scheduler.pt` / `rng_state.pth`，
可续训；`false` 时训练结束后从所有 checkpoint 删除优化器，root 只保留发布权重。
root 目录始终只有 `config.json` / `model.safetensors` / `chars.json` / `meta.json`。

**对话参数优先级**：`显式参数` > `properties\<配置名>.properties` > `模型文件里存的` > `默认值`
（网页对话与桥接服务都按这个顺序取，`chat.py` 与 `http_server.py` 各实现一份，改一处要同步另一处）。

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
- **WebUI 控制台只对本机**：控制台绑 `127.0.0.1` + 一次性 token；局域网只暴露只读对话页
  （`0.0.0.0:49135`，无鉴权，同一内网的人都能聊）。
- **CPU 训练**很慢且不启用 bf16，只建议小模型跑通流程。

## 九、参考资料

- 权重仓库：[ModelScope · donglouqiaoque/dlqq-llm-model](https://www.modelscope.cn/donglouqiaoque/dlqq-llm-model)
- 源码镜像：[GitHub](https://github.com/donglouqiaoque/dlqq-llm) ｜ [Gitee](https://gitee.com/donglouqiaoque/dlqq-llm)
- PyTorch 文档：<https://pytorch.org/docs/stable/index.html>
- nanoGPT：<https://github.com/karpathy/nanoGPT>
