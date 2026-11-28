# dlqq-llm-train · 字符级 GPT 训练与对话

**中文** ｜ [English](README.en.md)

本项目训练出的权重 ｜ [ModelScope 模型仓库](https://www.modelscope.cn/donglouqiaoque/dlqq-llm-model)

一套基于 transformers 的极小语言模型工具链：字符级词表、GPT-2 网络结构、从零训练，
**不加载任何已有模型权重**，也不使用 HuggingFace tokenizer。

**项目沿革**：本项目由 `dlqq-llm` 仓库迁移至此，现在名为 **`dlqq-llm-train`**；旧仓库已删除并重建，当前仓库为新仓库，提交历史从零开始；开源许可证由 Apache 2.0 调整为 MIT。

```
语料 txt  →  字符级词表  →  GPT-2（从零训练）  →  AdamW 训练  →  采样生成
```

网络结构使用 transformers 提供的 `GPT2LMHeadModel`，训练时从 `GPT2Config(...)`
随机初始化开始，只用本项目语料训练；加载推理时只 `from_pretrained` 本项目自己保存的标准目录，
**不加载任何第三方预训练权重**。
默认档位约 80 万参数，8 GB 显存的笔记本显卡几分钟就能跑出结果。

> 旧版手写 GPT 的 `.pt` 与 GPT-2 参数名不同，**不能直接加载**；切换后需要重新训练。

---

## 目录

- [快速开始](#快速开始)
- [环境准备](#环境准备)
- [训练模型](#训练模型)
- [跟模型对话](#跟模型对话)
- [用自己的语料](#用自己的语料)
- [参数与档位](#参数与档位)
- [运行模式与显存](#运行模式与显存)
- [目录结构](#目录结构)
- [常见问题](#常见问题)
- [把模型接进 DSH](#把模型接进-dsh)

---

## 快速开始
双击 `.runtime\start.bat`，浏览器会自动打开本地控制台（地址形如
`http://127.0.0.1:49134/?t=...`）。端口被占用时不会自动换端口，会提示输入新端口，
直接回车则退出。页面有十个标签：

| 标签 | 用途 |
|---|---|
| 训练 | 选配置和语料，点【开始训练】；日志实时滚动，同时落盘到 `.cache\train-logs\` |
| 对话 | 选配置，直接打字；回复逐字出现，可随时【停止】 |
| Agent | 选择支持工具调用的 API 模型；可带 MCP 工具，也可不选工具直接对话 |
| API | 接入任意 OpenAI 兼容服务，只保存 API Key 的环境变量名 |
| 服务 | 起一个 OpenAI 兼容的本地桥接服务，页面上能实时看到每次问答 |
| 环境 | 检测项目 Python、安装 / 更新 torch / numpy / transformers |
| 配置 | 在表单里改 `properties\*.properties`，保存时保留注释与键顺序 |
| 签名 | 给 `model\<名字>\` 下的标准权重文件计算或校验 SHA-256 |
| MCP | 配置 STDIO / HTTP 两种 MCP 服务器，并测试工具列表 |
| 版权鸣谢 | 列出前端第三方组件、版本、许可证和本地许可证文件 |

> **关掉 `start.bat` 的 cmd 窗口 = 退出**：训练、桥接服务、签名任务都会一起结束。
> 关掉浏览器标签**不算**退出，训练照跑，重新访问控制台地址即可。
> 每个模型目录里的权重文件带同名 `.sha256`，可在「签名」页核对后再加载。

局域网里的手机 / 其它电脑可以打开 `http://<本机IP>:49135/`：那是**只读对话页**，
不需要 token，只能选配置聊天，训练、配置、签名都看不到。第一次启动可能要在
Windows 防火墙里允许 Python 访问**专用网络**；不想开局域网页面就 `start.bat --no-lan`。

命令行不再是使用入口：`--config` / `--parent-pid` 这些参数只由 WebUI 内部调用。

## 环境准备

所有 Python 入口——WebUI、训练、对话、桥接服务、签名——统一使用项目内置运行时：

```text
.runtime\python\python.exe
```

| 项 | 说明 |
|---|---|
| WebUI | 不需要系统 Python；`start.bat` 直接用项目运行时启动 |
| 训练 / 对话 / 服务 | 也使用同一个项目 Python，不再读取用户自己的 Python |
| 构建期 | 只有运行 `make_python_env.bat` 时需要一份完整 CPython 3.10+（带 tkinter 不是必需的） |
| torch | 2.x；在 WebUI「环境」页安装，可选 PyPI / CUDA 12.6 / CUDA 12.8 |
| transformers | 5.x；在 WebUI「环境」页和 torch、numpy 一起安装 |
| 显卡 | 8 GB 显存即可跑默认档位；没有显卡时使用 CPU（会慢很多） |

### 第一次安装训练依赖

1. 双击 `.runtime\start.bat` 打开 WebUI；
2. 切到「环境」页；
3. 点项目 Python 旁边的【检测环境】：检测芯片类型/驱动版本，并把当前已安装的 Torch / Numpy / transformers 版本填入表单；
4. 安装表单按行填写：
   - **PyTorch**：芯片类型下拉框选硬件后端；驱动版本输入框填数字版本（如 CUDA `12.6`）；
     Torch 库版本可留空 = 最新，也可以手动指定；
   - **Numpy**：留空 = 最新版，也可以填 `2.5.4`；
   - **transformers**：留空 = 最新版，也可以填 `5.22.2`；
5. 点【安装 / 更新训练依赖】，等待 pip 日志结束；已安装且版本相同的库会自动跳过；
6. 再点一次【检测环境】，确认芯片类型/驱动版本和已安装版本。

检测环境只负责芯片类型/驱动版本和读取当前已安装版本；版本输入框默认留空即安装最新版，
也可以手动指定。`2.14.1` 只是示例。

Ollama 不再在环境页单独检测，而是作为一条 OpenAI 兼容 API 配置放在「API」页；WebUI 启动时
和 Python / MCP / 其他 API 一起检测，之后训练、对话、服务启动前不会重复自动检测。

依赖会安装到 `.runtime\python\Lib\site-packages`，也就是 WebUI 自己使用的项目 Python。WebUI 进程本身只 import 标准库，不 import torch，因此安装期间界面仍可用。

`make_python_env` 只会在 `.runtime\python\licenses\` 生成 CPython 与 `ensurepip` 的许可证。torch、numpy、transformers 是用户自行安装到项目 Python 的依赖，不属于随包分发内容，不收集其许可证；Python 环境目录之外的声明文件由人工维护，make 脚本不创建、不覆盖。

### 重建项目 Python

运行时缺失或损坏时，双击：

```text
.runtime\make_python_env.bat
```

按提示输入一份完整 CPython 安装目录（包含 `python.exe` 的那一级）。脚本只复制：

- 解释器本体；
- 标准库；
- `ensurepip`（供「环境」页安装训练依赖）；
- Windows 运行库。

自动化调用可加：

```text
make_python_env.bat <Python目录> autoyes nopause
```

`autoyes` 会自动确认覆盖，`nopause` 不等待按键。

**不会复制** `site-packages`、torch、numpy、transformers，也不会复制 tkinter/Tcl-Tk。

重建后训练依赖会丢失，需要重新到「环境」页安装。`make_python_env` 本身只负责 WebUI 最小运行时，不负责训练环境。

> 如果你之前使用的是旧版 `.runtime\python`，它可能没有 `ensurepip`；升级后先运行一次
> `make_python_env.bat`，否则「环境」页无法初始化 pip。

### 用哪个 Python

现在只有一种来源：

```text
<工程根>\.runtime\python\python.exe
```

- 不再读取 `properties\*.properties` 的 `python=`；
- 不再查找 PATH 上的 `python` / `python3` / `py`；
- 找不到项目运行时，WebUI 直接提示运行 `make_python_env.bat`，不会自动换解释器；
- 训练 / 对话 / 服务 / 签名全部使用同一个项目 Python；
- 重启 WebUI 后环境检测结果会清空，需要重新检测。

### 环境检测

- 统一在「环境」页检测，训练 / 对话 / 服务页的快捷检测按钮也只是检测同一个项目 Python；
- 检测内容包括：Python 版本、位数、torch / numpy / transformers、pip / ensurepip、CUDA；
- 缺少训练依赖时，训练 / 对话 / 服务按钮不会启动，先到「环境」页安装；
- 局域网对话页没有环境页，发送时会自动检测项目 Python，失败会返回明确错误；
- 不再存在“每个配置对应不同 Python”的情况，因此也没有多解释器冲突。

### 环境不对时会怎样

未检测或检测不通过时，不会启动训练 / 对话 / 服务，也不会去 PATH 里另找解释器。  
没有显卡时自动回退 CPU（只是慢），不会因为找不到 CUDA 而报错。

---

## 训练模型
### 在「训练」页训练（唯一入口）

1. 双击 `.runtime\start.bat`，页面默认就在「训练」标签；
2. **配置**下拉里选一个 `properties\*.properties`（下拉项后面会显示该文件头部的说明）；
3. **语料**下拉显示 `res\` 下每个语料目录的文件数与体积，仅供核对——实际读哪个目录由配置里的
   `data_path` 决定；
4. 点【开始训练】。日志会实时滚动，同时写入 `.cache\train-logs\<配置名>-<时间>.log`；
5. 想停就点【停止】；想接着上次的存档训，点【继续训练】（需要 `checkpoint-<step>/optimizer.pt`）。

页面上看不到参数？参数都在配置文件里：`properties\` 下四个文件（`chat_demo` / `dlqq_exam` /
`dlqq_llm` / `dlqq_poem`），键和含义见下文的参数表，也可以在「配置」页直接改。

### 配置文件在哪、怎么改

- 文件：`properties\<名字>.properties`，一行一个 `key=value`，`#` 开头是整行注释；
- **不要写行尾注释**：解析器按 `split("=", 1)` 取值，`block_size=128  # 说明` 会把注释当成值；
- 四个文件的键集合、顺序、分节保持一致，新增键要同时出现在四个文件里；
- 在「配置」页保存时，只改 value，注释、空行、键顺序都保留，改完下次训练生效。

### 想加一套新配置

复制一份现有的改改名字，它会自动出现在「训练」页的下拉里（下拉内容是实时扫描 `properties\`）。

## 跟模型对话
在「对话」标签里：

1. **模型**下拉会自动列出 `model\` 下的标准模型目录（显示体积、修改时间、是否已签名）；
2. 调好温度 / top_k / 生成长度（默认取 `properties\<模型名>.properties` 里的值）；
3. 输入内容回车发送，回复会**逐字**出现；输入框右下角是同一位置的发送/停止按钮，
   空闲时显示 ↑，生成中变成 ■，点击即可停止当前会话。

- 切换模型不用重启页面：旧模型会被释放（含显存）再加载新的；
- 训练正在跑时，**只能选不同的模型**——与训练配置同名的模型会被拒绝，避免拖慢训练或读到半成品；
- 对话上下文按模型的 `block_size` 截断，拼提示词的规则是 `用户：…` / `助手：…`；
  词表里没有这两个标记的模型会自动切「续写模式」；
- 助手回复会按 Markdown 渲染，支持 `$...$` 和 `$$...$$` 数学公式（本地 KaTeX）；
  用户消息仍按纯文本显示；
- API 模型（包括 Ollama）的模型和思考等级合并到输入框下方的同一个按钮里，点开后先选模型、
  再选推理等级，等级来自检测结果；本地模型不显示思考等级。推理等级第一项 `Default` 表示
  不发送任何思考参数，交给服务端默认行为；`关闭` 才是显式关闭思考。

## 用自己的语料

**语料统一放在一个目录里**（`data_path` 只接受目录，不再接受单个文件）。

先在 `res\` 下建一个目录，例如 `res\my_corpus\`，把语料文件丢进去：

```
res\
  my_corpus\
    part1.jsonl
    part2.jsonl        ← 可以多个文件，训练时按文件名排序合并
    notes.txt
```

支持的文件：

| 文件 | 怎么处理 |
|---|---|
| `.jsonl` 问答（`input`/`target` 等） | 拼成「用户：…／助手：…」 |
| `.jsonl` 纯文本（`{"text": "..."}`） | 原样当纯文本 |
| `.txt` | 原样当纯文本（UTF-8 / GBK / BIG5 都能自动识别） |

想让它会对话，jsonl 这样写：

```json
{"input": "什么是过拟合", "target": "模型把训练数据背下来了，换一批新数据就不行了。"}
{"input": "那怎么缓解", "target": "加数据、加正则、减小模型容量，三条路一起走。"}
```

然后训练：

```bat
# 在 properties\ 里新建一份配置，改这三行：
#   data_path=res/my_corpus
#   model=我的模型
#   max_iters=5000
```

几个经验值：

| 语料规模 | 大致效果 |
|---|---|
| < 100 KB | 基本是**背下来**的，问答能对上，换个问法就崩 |
| 1 MB 左右 | 开始能看出「学到了句式」，泛化明显变好 |
| 10 MB 以上 | 短句通顺，长句仍会飘 |

目录里的文件按 UTF-8 / utf-8-sig / gb18030 / big5 自动识别，记事本存的 GBK 文件也能直接读。

### 下载来的语料要先清洗

从维基、fineweb 这类源下回来的东西不能直接喂（维基是 `{{模板}}` `[[链接]]`，网页语料混着英文和广告）。
清洗要做的是：剥维基标记、统一空白、长度过滤、中文占比过滤、按行去重，
最后输出训练脚本能直接吃的 `.txt`（段落间空行）或 `.jsonl`。
本机有现成的清洗脚本（不随仓库发布），用法见 AIRES.md。

---

## 参数与档位

| 档位 | n_layer | n_head | n_embd | block | batch | 参数量 | 显存 | 说明 |
|---|---|---|---|---|---|---|---|---|
| `tiny` | 2 | 2 | 64 | 64 | 16 | ~0.2 M | < 0.5 G | 几秒一轮，用来验证流程 |
| `small`（默认） | 4 | 4 | 128 | 128 | 32 | ~0.8 M | < 1 G | 能看出效果 |
| `medium` | 6 | 6 | 384 | 256 | 32 | ~10 M | ~2–3 G | 语料够大时明显更好 |
| `large` | 8 | 8 | 512 | 256 | 32 | ~25 M | ~6–7 G | 8 G 显存的上限附近，别再往上 |

所有配置项都能在「配置」页直接修改，也可以在 properties 文件里改：

```properties
n_layer=6
n_embd=384
lr=0.001
max_iters=5000
```

常用的几个键：

| 键 | 默认值 | 含义 |
|---|---|---|
| `data_path` | 无 | 语料目录；必须指定，未配置时训练直接报错 |
| `out_path` | `model/mymodel` | 模型保存目录（HF 标准目录） |
| `max_iters` | 2000 | 训练步数，最直观的「训练多久」 |
| `batch_size` | 32 | 每步几条样本，显存不够就调小 |
| `block_size` | 128 | 上下文长度；语料太短会被自动调小 |
| `lr` | 3e-4 | 学习率 |
| `dropout` | 0.1 | 语料小的时候可以调大一点 |
| `eval_interval` | 200 | 多少步评估一次、顺便决定要不要保存 |
| `max_vocab` | 8000 | 词表上限，只保留最常出现的前 N 个字 |
| `val_frac` | 0.05 | 从语料尾部切这个比例做验证集 |
| `cache_dir` | `.cache` | 切好词的二进制缓存放这里 |
| `prompt` | 空 | 训练结束采样用的开头；留空自动挑 |
| `data_format` | `auto` | `auto` / `txt` / `jsonl`；auto 看扩展名和文件开头 |
| `q_field` | 空 | jsonl 里「问」的字段名；留空自动推断 |
| `a_field` | 空 | jsonl 里「答」的字段名；留空自动推断 |
| `eval_iters` | 50 | 每次评估跑几批；长训练建议调到 20 省时间 |
| `grad_accum` | 1 | 梯度累积：显存不够时用它凑等效 batch |
| `warmup_iters` | 0 | 学习率预热步数，长训练建议 1000–2000 |
| `lr_decay` | `cosine` | `cosine` 余弦退火 / `none` 恒定 |
| `min_lr_frac` | 0.1 | 退火终点 = `lr × 这个比例` |
| `grad_checkpoint` | `False` | 梯度检查点：省显存，慢约 20–30% |
| `save_every` | 0 | 0 = 只存最优；>0 = 额外每 N 步强制存档 |
| `keep_best_checkpoints` | 1 | 保留最近 N 个“验证变好”的 checkpoint；0 = 不保留，-1 = 不限制 |
| `keep_forced_checkpoints` | 1 | 保留最近 N 个强制保存的 checkpoint；0 = 不保留，-1 = 不限制 |
| `chat_mode` | `True` | `False` = 纯续写模型（诗词），提示词不加「用户：／助手：」 |
| `chat_temperature` | 0.8 | 对话采样温度，越大越发散 |
| `chat_top_k` | 40 | 只从概率最高的 K 个里采样；0 = 不截断 |
| `chat_max_new` | 180 | 一次最多生成多少字 |

`chat_*` 这几个是**对话参数**，训练时写进模型文件；网页对话和桥接服务加载模型时都按下面的
优先级取用（改 properties 不用重训就生效）：

```
显式参数  >  properties\<配置名>.properties  >  模型文件里存的  >  默认值
```

---

## 让 dlqq_llm 训得更久（可选）

`dlqq_llm` 现在直接在 `properties\dlqq_llm.properties` 上训练，想训得更久就改这个文件里的参数，
不需要另建配置：

| 参数 | 作用 |
|---|---|
| `max_iters` | 训练步数；配合 `batch_size × block_size` 估算见过的 token 数 |
| `grad_accum` | 显存不够时用梯度累积凑等效 batch |
| `warmup_iters` / `lr_decay` / `min_lr_frac` | 学习率预热 + 余弦退火，长训练必开 |
| `grad_checkpoint` | 梯度检查点，省显存（慢 20–30%） |
| `save_every` | 0 = 只存最优；>0 = 额外每 N 步强制存档，防止「验证损失一直没变好 → 一直不存盘」 |
| `keep_best_checkpoints` | 保留最近 N 个最优 checkpoint；0 = 不保留，-1 = 不限制 |
| `keep_forced_checkpoints` | 保留最近 N 个强制存档；0 = 不保留，-1 = 不限制 |

例：把 `max_iters` 调到 60000，双击 `.runtime\start.bat`，在「训练」页选 `dlqq_llm`、点【继续训练】
即可接着上次的存档（存档里要有优化器状态）。

---

## 做一个小体积的考题 demo（dlqq_exam）

仓库体积有限（Gitee 单文件 50 MB）。`chat_demo` 是 30 条手写问答（13 KB），
跑通流程够用，但互动性弱；`dlqq_exam` 改用**考题库**当语料，同样体积下信息密度高得多。

两步：

```bat
REM 1) 准备考题语料：把 C-Eval / 高考 / 生物 合并成 res\dlqq_exam\dlqq_exam.jsonl
REM    （语料不进仓库，需自行下载并合并）

REM 2) 训练 + 聊天
REM    双击 .runtime\start.bat，在「训练」页选 dlqq_exam 开始训练
REM    在「对话」页选 dlqq_exam 进行对话
```

体积控制的关键是 `properties\dlqq_exam.properties` 里这几行：

```properties
# 权重存半精度，体积减半
save_dtype=fp16
# checkpoint 始终带优化器；分别限制最优存档和强制存档
keep_best_checkpoints=1
keep_forced_checkpoints=1
```

根目录 `model\dlqq_exam\` 只保留最佳权重，不含优化器；checkpoint 才用于续训。

实测：4,562 条语料切出约 270 万 token，12.55 M 参数存成 fp16 的发布权重约 **27 MB**；
`chat_demo` 更小，0.9 M 参数约 **1.9 MB**。

> **⚠ 数据来源要写明**：`dlqq_exam` 把题库全部用于训练，模型是**记住了这些题**，
> 所以要标注语料来源（魔搭上的 C-Eval / gaokao-benchmark 等），
> 并且**不要**把同一批题当 benchmark 报分数 —— 那是自问自答，不是能力。
> 另外 C-Eval 的许可是 CC BY-NC-SA 4.0（非商业），这份语料不要随仓库公开分发。

### 语料从哪来（全部走魔搭，国内直连）

| 数据集（魔搭仓库） | 规模 | 用途 |
|---|---|---|
| `philexohf/LLM-Pretrain-Data` | 约 24.6 GB / 49.6 亿 token | **预训练首选**：网页 55% + 新闻 27% + 维基 12% + 百科 6%，已清洗去重，txt 每行一篇 |
| `AI-ModelScope/chinese-fineweb-edu` | 约 290 GB | 只要网页语料时取子集 |
| `OmniData/C-Eval` | 1.5 MB | 52 学科选择题 |
| `opencompass/gaokao-benchmark` | 5.7 MB | 高考选择题 |
| `pandayi/Biology_subject_of_college_entrance_examination` | 0.2 MB | 高考生物（含解析） |
| `SHPDataGR/medical-exam-question-bank-json` | 35 MB | 医学考试题库 |
| `tongyi_dianjin/CFLUE` | 66 MB | 金融资格考试选择题 |

下载方式：

```bat
REM 考题：从魔搭自行下载 C-Eval / gaokao-benchmark / 高考生物，合并成一个 jsonl
REM       （直连魔搭，不需要 GitHub / HuggingFace / 登录）

REM 预训练文本：下载后清洗成 jsonl，放进训练配置指向的语料目录
```

下载和清洗都用本机的辅助脚本完成（不随仓库发布），用法见 AIRES.md。
产出的 jsonl 直接进语料目录，训练配置指向的就是那里，下完直接训即可。

### 预训练语料有多大？为什么默认只下 2 GB？

脚本支持 `--list` 查看各语料的总量（用法见 AIRES.md）：

```
mix      philexohf/LLM-Pretrain-Data
         可用 1 个文件，合计 12.72 GB
fineweb  AI-ModelScope/chinese-fineweb-edu
         可用 302 个文件，合计 311.44 GB
```

| 语料 | 总量 | 说明 |
|---|---|---|
| `mix`（train.txt） | **12.72 GB** ≈ 45 亿 token | 网页+新闻+维基+百科，一行一篇 |
| `fineweb` | **311 GB**（302 个 parquet 分片） | 教育向网页，按片下载 |

默认 2 GB 是按模型规模算的：

- `dlqq_llm` 是 31.44 M 参数，按「每参数 20 token」≈ **6.3 亿 token** 就够；
- 2 GB 中文文本 ≈ 7 亿 token，加上原来的 `firefly-train-1.1M.jsonl`（1.1 GB ≈ 4 亿 token），
  总量约 11 亿 token，已经超过所需；
- 全下 12.7 GB 对 29.4 M 模型没有收益，只是多花 1–2 小时下载和 13 GB 硬盘。

真要全量就传 `--max-mb 0`（一直下到结束），或者用 `--max-mb 8000` 取 8 GB。

---

## 训练 GB 级大语料

1 GB 中文约 3.5 亿字。如果先把整个文件读进内存、再转成 Python 列表，光这一步就要
吃掉十几 GB —— 所以数据管线是**流式**的：

```
分块读 → numpy 统计字频 → 取前 max_vocab 个字 → 分块编码成 uint16 → 写缓存文件
训练时 np.memmap 内存映射，随机读几 KB
```

**内存占用与语料大小无关**，只跟 `block_size` / `batch_size` 有关。

### 直接喂 JSONL，不用先转 txt

`data_path` 指到 `.jsonl` 就行，脚本会自动找问答字段并拼成 `用户：…／助手：…`
（**直接把 jsonl 喂进去会把括号引号也学进文法，所以这一步必须做**）。

自动识别的字段名，按顺序尝试：

```
input/target · q/a · question/answer · src/tgt · instruction/output
prompt/response · query/response · ask/answer · title/content
```

都对不上就显式指定：`q_field=问  a_field=答`。

### 实测（1.12 GB 的 `res\dlqq_llm\`，1,649,399 条问答）

| 指标 | 数值 |
|---|---|
| 切词耗时 | **30 秒**（含启动，建一次后续复用） |
| 产出 token | 425,220,455 |
| 词表 | 8000（上限截断），词表外只丢弃 21 万字（0.05%） |
| `data.bin` | 850 MB |
| 峰值内存 | 约 1.5 GB（与语料大小无关） |
| 第二次运行 | 4.9 秒（直接复用缓存） |

配置要点（写进 `properties\dlqq_llm.properties`）：

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

这份语料的实测训练速度（29.43 M 参数，8 GB 显存跑得下）：

| 档位 | 配置 | 速度 | 60000 步 |
|---|---|---|---|
| small | 4 层 / 128 维 / block 128 / batch 32 | 143 步/秒 | 约 7 分钟 |
| large | 8 层 / 512 维 / block 256 / batch 32 | 约 5 步/秒 | 约 3.3 小时 |

> small 档只有 0.8 M 参数，喂 1 GB 会严重欠拟合；**GB 级语料请用 large**。
> 冒烟测试里 large 档 30 步就把验证损失从 9.06 压到 5.72，说明数据是通的。

两点注意：

- 每份语料（连同 `max_vocab` / `val_frac` / 格式 / 字段这些设置）对应一个独立的缓存文件，
  改任何一项都会自动写一份新的。所以 demo 的 `chat_demo\` 和 GB 级的 `dlqq_llm\`
  可以**同时留着**，来回切换不用重新切词；想全部重来就删掉缓存目录；
- 验证集是从**语料尾部**切的。语料若按类别顺序排列（比如前半新闻、后半对话），
  建议先打乱行序，否则验证损失反映不了真实水平。

### 加语料：往目录里丢文件就行

`data_path` 只接受**目录**（传文件会直接报错并打印迁移命令）。`dlqq_llm.properties` 指向：

```properties
data_path=res/dlqq_llm
```

这个目录里可以放任意多个文件，训练时按文件名排序依次读、拼成一条流：

| 放进去的东西 | 怎么处理 |
|---|---|
| `*.jsonl`（`input`/`target` 等问答字段） | 拼成 `用户：…／助手：…` |
| `*.jsonl`（只有 `{"text": "..."}`） | 当纯文本直接喂（预训练语料就长这样） |
| `*.txt` | 原样当纯文本 |
| 子目录 | 也会扫（跳过隐藏目录） |

新增/删除/改动文件后，缓存指纹会变，**下次训练自动重新切词**，不用手动删缓存。

扩充语料从魔搭下载（不需要 GitHub / HuggingFace / 登录），下载与清洗由本机的辅助脚本
完成，用法见 AIRES.md。常用的几种取法：

它会做长度过滤、中文占比过滤、按行去重，输出 `pretrain-mix-000.jsonl`、`pretrain-mix-001.jsonl`……
每个默认 200 MB，**已有文件绝不覆盖**（序号往后顺延）。下载时有进度条，
结束后会打印「语料目录现在共 N 个文件 / X GB」。下完直接双击 `.runtime\start.bat`，在「训练」页选 `dlqq_llm`。

---

## 训练一个写诗模型 dlqq_poem

专门写诗的小模型：6 层 384 维（13.40 M 参数），语料是**古诗 + 宋词 + 近现代诗**，
在 8 GB 显存上半小时左右训完。

语料里**只有诗词正文，没有「诗：」「词：」这类标注**，所以它写出来的就是诗本身，
不会把标注当内容复读。

### 1. 下语料

三个源（诗 43 MB、宋词 19 MB、近现代诗 57 MB）分别下载、清洗，由本机的辅助脚本完成，
用法见 AIRES.md。

| 源 | 仓库 | 大小 | 内容 |
|---|---|---|---|
| `shici` | `modelscope/chinese-poetry-collection` | 43.4 MB | 唐宋元明清诗 |
| `songci` | `peaceminuczy/songci` | 18.7 MB | 宋词 |
| `modern` | `Iess/chinese_modern_poetry`（国内镜像 hf-mirror.com） | 56.6 MB | 近现代诗（中国诗人 + 外国诗中译） |

都会下载到 `res\dlqq_poem\`，输出 `shici-000.jsonl`、`songci-000.jsonl`、`modern-000.jsonl`，
已有文件不覆盖、进度条、按句号断行排版。

> 旧版本生成的语料每首带「诗：」「词：」前缀。脚本会检测这种文件并**停下提醒你删掉**
> （`del res\dlqq_poem\*.jsonl`），否则新旧混在一起会把标注学进去。

### 2. 训练

```bat
REM 双击 .runtime\start.bat，在「训练」页选 dlqq_poem
```

配置在 `properties\dlqq_poem.properties`，是按实际语料调过的：

| 项 | 值 | 依据 |
|---|---|---|
| 结构 | 6 层 384 维（13.40 M） | 约 4000 万字 / 13.4 M 参数 ≈ 2.9 token/参数 |
| `block_size` | 256 | 现代诗平均 210 字、90% 在 136 字内，256 能完整装下一首 |
| `batch_size` | 32 | 32 × 256 = 每步 8192 token（与 64×128 步数相同，上下文翻倍） |
| `max_iters` | 15000 | ≈ 1.23 亿 token ≈ 语料（3920 万 token）3.1 轮 |
| `max_vocab` | 7000 | 覆盖 99.9% 以上的字 |
| `chat_mode` | `false` | 告诉 `chat.py` 这是续写模型，别套「用户：／助手：」 |

实测约 7.0 万 token/秒，**全程约 29 分钟**；只保存验证最优模型（fp16）。

### 3. 让它写

语料是纯正文，所以对话就是**给它一个开头，它往下写**：

```bat
REM 双击 .runtime\start.bat，在「对话」页选 dlqq_poem
```

```
你  > 春
它  > 风暖鸟声碎，日高花影重。
      ...
```

- 输入一两个字 → 以它起句；
- 输入一句诗 → 接着往下续；
- 直接输入一段现代诗的句子 → 按现代诗的断行风格续写。

因为 `chat_mode=false` 存在模型文件里，`chat.py` 会自动用续写模式，不会再加「用户：／助手：」。

采样参数也放在 `properties\dlqq_poem.properties` 里，改完**不用重训**就生效：

```properties
chat_mode=false
# 小模型用 0.8 容易陷入「他的女人，他的女人」循环
chat_temperature=1.0
# 0 = 不截断，从全词表采样
chat_top_k=0
chat_max_new=200
```

`.runtime\start.bat` 的「对话」页和「服务」页加载这个模型时都会读这些值；
临时想改，直接在 properties 里改同名项，重新开始一次对话或重开服务即可生效。

---

## 运行模式与显存

训练日志和「环境」页会报告运行环境：

```
[模式] GPU —— 混合精度 bf16（推荐：显存占用更低、张量核更快）
[设备] NVIDIA GeForce RTX 4060 Laptop GPU | sm_89
[显存] 总 8.0 GB，已用 1.1 GB（可用 6.9 GB）
```

三种模式的含义：

| 模式 | 触发条件 | 说明 |
|---|---|---|
| **GPU · 混合精度 bf16** | 有 CUDA 设备 | 前向与反向走 bf16，显存占用更低、张量核更快 |
| **GPU · 单精度 fp32** | 有 CUDA 设备但关掉 AMP | 精度更高，显存占用更大 |
| **CPU · 单精度 fp32** | 没有可用的 CUDA 设备 | 会额外报**内存**总量与占用；CPU 上开 bf16 反而更慢，所以不启用 |

显存不够时的处理顺序：调小 `batch_size` → 调小 `block_size` → 换更低的档位。

---

## 目录结构
```
.runtime\               本地运行时与启动器（python/ 不进版本库）
  start.bat             双击入口：固定用 .runtime\python\python.exe 起 code\webui.py
  make_python_env.bat   重建 .runtime\python；可输入系统 Python 路径
  make_python_env.py
  logo.ico              快捷方式图标
  python\               项目 Python：标准库 + ensurepip（训练依赖由环境页安装）
code\                   所有 Python
  webui.py              本地 Web 服务：静态页 + JSON API + SSE + 子进程管理
  env_install.py        用项目 Python 安装训练 / 对话依赖
  train.py              训练（数据管线 + 模型 + 训练循环 + 采样）
  chat.py               对话核心：ChatSession（加载 / 流式回复 / 改参数）
  chat_worker.py        对话子进程：常驻 ChatSession，stdin/stdout 走 JSON 行
  http_server.py        OpenAI 兼容桥接服务（本地模型 + API 代理）
  ollama_client.py      Ollama 原生元数据辅助（仅用于 API 配置的模型能力补齐）
  openai_client.py      OpenAI 兼容 API 客户端（Base URL / 环境变量 Key / SSE / embeddings）
  mcp_client.py         MCP 客户端（STDIO / Streamable HTTP / 工具 schema）
  common.py             find_root / properties 保序读写 / 父进程看门狗 / 隐藏目录
  model.py              GPT-2 封装（transformers）
  model_signature.py    权重 SHA-256 签名（只依赖标准库）
web\                    前端页面（由 webui.py 托管）
  index.html            十个标签：训练 / 对话 / Agent / API / 服务 / 环境 / 配置 / 签名 / MCP / 版权鸣谢
  chat.html             局域网只读对话页
  vendor\               本地第三方前端资源：xterm.js / marked / DOMPurify / KaTeX
properties\             四个训练配置
res\                    语料目录（不进版本库）
model\                  训练产物（model\<名字>\ 标准 HF 目录 + checkpoint-<step>\ + .sha256）
                        每个目录含 config.json / model.safetensors / chars.json / meta.json
.cache\                 训练缓存/日志/配置备份/MCP 与 API 配置、工作区注册表、会话、输入区附件（目录纳入版本控制，内容忽略）
.ai_res\                AI 临时目录（不进版本库，不参与训练）
THIRD-PARTY-NOTICES.txt  Python / 依赖 / 前端第三方库的许可证清单（人工维护）
```

- 工程根目录靠「向上找到含 `code\train.py` 的那一级」定位，整个目录搬到哪都能跑；
- 模型 id = 目录名：`model\chat_demo\` 的 id 就是 `chat_demo`。
- 续训目录：`model\<名字>\checkpoint-<step>\` 内含 `config.json` / `model.safetensors` /
  `optimizer.pt` / `scheduler.pt` / `trainer_state.json` / `rng_state.pth`。

## 常见问题

**Q：窗口一闪就没了，看不到结果？**

所有 bat 结尾都有 `pause`，正常情况不会闪退。如果确实闪退，多半是没找到 `python.exe`，
在终端里运行一次就能看到完整报错：

```bat
.runtime\start.bat
```

**Q：提示「运行环境缺少依赖」怎么办？**

到 WebUI 的「环境」页点【检测环境】，再点【安装 / 更新训练依赖】。依赖会安装到项目内置 Python，
不需要也不应该去 PATH 上找另一个 Python。

**Q：重建 `.runtime\python` 后训练报缺少依赖？**

`make_python_env` 只重建 WebUI 最小运行时，不包含 torch / numpy / transformers。重建后回到「环境」页
重新安装训练依赖即可。

**Q：在 IDE 的「运行」按钮里跑，报「读不到键盘输入」？**

那种方式没有可交互的终端，`input()` 直接拿到 EOF，所以没法选编号。两个办法：

1. **双击** `.runtime\start.bat`，在「训练」页选 `dlqq_exam`，正常答题；
2. 在 IDE 里直接运行 `code\webui.py` 时，如果端口被占用，没有终端可输入新端口；
   改用 `.runtime\start.bat` 启动，或先释放端口。

**Q：训练完想换语料重训，模型名会冲突吗？**

不会。输入已存在的名称时会问你是否覆盖，回答 `n` 就可以换个名字。

**Q：想训练但不想用 `chat_demo`？**

先在 `res\` 下建一个语料目录，再复制一份配置改两行（「训练」页的下拉会实时列出新配置）：

```
# properties\my_corpus.properties
data_path=res\my_corpus      ← 目录，里面放 jsonl / txt
model=我的模型
...（其余照抄 chat_demo.properties，按需改 n_layer / max_iters 等）
```

**`data_path` 只接受目录**；目录里 `txt` 和 `jsonl` 都支持，格式按扩展名和文件开头自动判断，
jsonl 会自动找问答字段（`{"text": ...}` 这种单字段的当纯文本用）。

**Q：想临时改一个参数，不想动配置文件？**

在「配置」页直接改并保存即可；保存前会自动备份，训练/对话启动时读取当前 properties。

**Q：模型训练很久还是只会复读？**

语料太小。字符级模型在 100 KB 以下的语料上主要是在背诵，
想看到泛化，把语料做到 1 MB 以上再试。

**Q：能用 CPU 训练吗？**

能，但会很慢，而且 CPU 上不会启用 bf16（反而更慢）。建议至少 `tiny` 档位先跑通流程。

**Q：怎么确认权重文件没被改过、没传坏？**

双击 `WebUI「签名」页的【校验签名】`：它扫描 `model\` 下所有标准模型目录，重算权重文件的 SHA-256，
和旁边的 `<权重文件>.sha256` 比对，四个都 PASS 才算没问题。训练完、或把权重传到模型仓库
之前跑一次。签名用 `WebUI「签名」页的【计算签名】` 重新生成（改过权重后必须重新生成）。

---

## 把模型接进 DSH
在「服务」标签里：

1. 在「环境」页确认项目 Python 检测通过；「服务」页不再按配置区分解释器，也不需要“全部配置”检测；
   勾选“启动服务时检测”时，启动前会自动补一次检测；
2. 填端口（默认 49128），点【启动服务】；
3. 页面显示 `http://127.0.0.1:49128/v1`，把这个地址填进 DSH 的模型提供方设置；
4. 之后每次请求的问答都会实时出现在页面下方的「会话记录」里（来自服务进程输出的
   `@@EV@@` 事件），排错时不用再翻控制台；
5. 点【停止】关闭服务；关掉 `start.bat` 的窗口时它也会一起退出。

服务本身是 OpenAI 兼容的：`/v1/models`、`/v1/chat/completions`（支持 `stream`）
和 `/v1/embeddings`（API 的 embedding 模型）。本地模型在 `stream=true` 时逐字生成并发送；
API 模型直接代理上游 SSE 和 embeddings。服务页的模型清单按 API 检测结果区分
对话 / 嵌入，embedding 请求会写「向量」日志，显示模型、条数、维度、耗时，并把完整向量
放在可折叠详情里（批量请求只展示第 1 条完整向量），不会把完整高维向量塞进会话气泡。
模型清单来自 `properties\` 里能对上标准模型目录的配置（不再扫描旧 `.pt` 文件），谁被调用才加载谁；
对话参数也读同一份配置。

### Ollama 作为 API 配置

Ollama 已合并到 OpenAI 兼容 API，不再使用单独的 `ollama/` 模型 id：

```text
名称        Ollama
Base URL    http://127.0.0.1:11434/v1
Key 环境变量 （留空）
```

WebUI 启动检测时会请求 `http://127.0.0.1:11434/v1/models`，再通过 Ollama 的 `/api/show`
补齐模型类型、工具和思考能力。模型显示成：

```text
Ollama/qwen3.5:4b（API）
Ollama/bge-m3:latest（API · 嵌入）
```

嵌入模型只出现在服务页，不会出现在 Chat / Agent。

### 接入 MCP / Agent（可选）

「MCP」页可以保存多个 MCP 服务器配置，支持两种传输：

```text
STDIO：command + args + cwd
HTTP ：http://127.0.0.1:<端口>/mcp
```

STDIO 示例：

```text
java -Dfile.encoding=UTF-8 -jar <MCP-SERVER-JAR>
```

HTTP 示例：

```text
http://127.0.0.1:<端口>/mcp
```

配置保存在 `.cache\mcp_servers.json`。保存后点【连接并获取工具】可以验证连接并查看工具名称。
WebUI 启动时也会自动检测已保存的 MCP 服务器，结果保存到 `.cache\mcp_status.json`；
MCP 页的【检测全部】可手动重新检测。已配置的 STDIO MCP 进程会随 WebUI 启动并常驻，
不会在每次对话结束时退出；修改对应配置时重启该进程，WebUI 退出时统一销毁。
MCP 连接带自动恢复：STDIO 进程退出或 HTTP 连接失效后按 500ms 起步、最长 30s 指数退避重连，
默认最多连续尝试 10 次；重连期间工具保留但调用返回明确错误，重连成功后重新拉取工具列表；
超过次数后标记失败并停止自动重连，可在 MCP 页手动【重连】。已发出的非幂等工具不会自动重放。

「Agent」页列出工具能力为“支持”或“未知”的 API 模型，以及已保存的 MCP 服务器。模型和思考等级
合并到输入框下方的模型按钮；MCP 只占一个按钮，显示 `MCP 开 · 2/3 · 42`（已启用服务器数 /
服务器总数 · 已选工具数）或 `MCP 关`。首次进入 Agent 时会建立草稿会话并按当前配置同步
MCP 状态，按钮直接显示正确的开关和工具数量，不需要再点开 MCP 浮层刷新。点开 MCP 浮层后可以：

- 用总开关快速关闭/恢复全部 MCP；
- **逐台**启用/禁用服务器，例如开启 A、关闭 B，互不影响；
- 用「工具来源」下拉选择当前要编辑哪台服务器的工具；
- 按工具名或说明搜索工具、全选/清空当前筛选结果、逐个勾选。

每台服务器分别保存自己的启用状态和工具勾选，会话 JSON 里对应 `mcp_servers`、`mcp_tools`
和总开关 `mcp_enabled`；切服务器开关或工具后会立即保存，即使当前还没发第一条消息也会保存一个
空会话，刷新页面后能恢复这些设置。关闭总开关或某台服务器后，该服务器的 `tools` 不会发送给模型；
MCP 进程仍保持常驻，其它会话不受影响。多台服务器工具重名时，后端保留先启用的那台，
避免模型调用歧义。旧会话只有单个 `server` / `tools` 字段时，会自动迁移成一台已启用服务器。
发送任务后：

```text
用户消息 → 模型 tool_calls → MCP tools/call → 工具结果回填 → 模型继续生成
```

调用工具时会把当前 Agent 工作区的**绝对路径**通过 MCP `params._meta["dsh/workspace"]`
一起发送；会话 ID 和 Agent ID 分别放在 `dsh/sessionId`、`dsh/agentId`。MCP 服务端可据此
按会话隔离工作区，首次收到请求后使用该路径初始化会话工作区。

执行过程不使用嵌套气泡，而是平铺的 Agent 时间线：顶部显示运行状态和用时，当前步骤状态
（如“正在调用 …”“… 已完成，继续生成…”）紧跟在“运行中 · N 秒”后面；默认折叠
所有「上下文注入 / 思考 / 工具调用」行，回答内容和会导致会话中断的报错（如模型请求超时、
输出达到上限）始终显示；每条思考、回答和工具调用都带系统本地时区的
`YYYY-MM-DD HH:mm:ss.SSS` 时间戳。点击顶部展开历史，单条参数/结果再自行展开。
上下文注入、思考、工具调用/结果和回答分段也会随会话保存为 UI 时间线；刷新页面或重新打开
会话后仍可展开查看。这些时间线字段只用于界面恢复，不写入 `model_history`，不会改变发给
模型的上下文。完成、停止或失败后，顶部状态文字会清空，不再在输入区单独显示“完成”。
下面按模型实际输出顺序排列「思考 / 工具调用 / 回答」行。工具名一出现就先显示“准备调用 /
正在调用”，执行完成后更新为“完成”或“失败”；工具参数和结果完整回填给模型，不做硬截断。
`dlqq-mcp-server` 的工具返回会按类型美化显示：文件读取显示行号代码，目录树保留树形结构，
搜索结果显示文件路径/行号/上下文卡片，文本编辑显示增删 diff，JSON 和键值返回显示结构化
内容，命令/构建输出显示等宽文本；每张工具卡片仍可展开原始返回。只执行本次勾选的工具；
模型返回未勾选或不存在的工具时，回填“工具不存在”而不调用 MCP。设置思考等级后，思考内容
和工具调用按实际顺序排列在回答上方；思考行折叠时会在“思考”后面显示最后一句话，展开后
显示完整内容。推理等级菜单第一项 `Default` 表示不传任何思考参数，`关闭` 才是显式关闭。
DeepSeek 默认开启思考，选择“关闭”时会显式发送
`thinking: {"type": "disabled"}`；选择其它档位时发送 `thinking: enabled` 和对应的
`reasoning_effort`，避免只省略参数但模型仍然思考。
上下文注入行只在**本轮真正追加了上下文消息**时出现：首次请求的
`AGENTS.md` / `CLAUDE.md` baseline、runtime-context 快照、时钟快照，或后续指令/工作区
变化、时钟刷新时新增的替换快照；没有变化时这一组完全不会出现。`system` 提示词和 `tools`
是每次请求都必须携带的请求参数，不属于“重新注入”，因此不再作为注入行展示。展开注入卡片
可以看到实际追加的内容，标签显示“首次注入 / 已更新 / 新快照”。
embedding 模型和本地字符级模型不会出现在 Agent 模型列表中；本地字符级模型也不支持思考
等级控件。桥接服务仍按原逻辑转发客户端传来的 `tools`，不在桥接进程内自行执行 MCP 工具。

创建大文件时建议只勾选相关的文件工具（如 `dlqq_create_file` / `dlqq_append_file` /
`dlqq_overwrite_file`），不要让模型一次生成过长的内容；工具选得过多会占用上下文，模型也
更容易编造不存在的工具名或生成被截断的 tool_call。

### 工作区、会话与输入区

Chat / Agent 页改为左侧导航、右侧对话内容：

- **Chat** 只有会话列表，不关心工作区；
- **Agent** 是「工作区 → 会话」的树形结构：每个工作区下面列出自己的会话，一个工作区可以保存多个会话；
- 会话行支持改名、删除和**移动工作区**；移动后下一次 MCP 工具调用自动使用新工作区路径；
- 左侧栏可收起，也可拖动右侧边缘调整宽度；宽度和收起状态会保存在浏览器本地；
  窗口宽度不足时会默认收起侧栏，消息区优先占用宽度，Markdown 表格和代码块在窄屏内横向滚动；
- 新建 Agent 会话时使用所属工作区；没有指定时使用默认工作区
  `<工程根目录>\.runtime\default_workspace`。

会话每轮结束后自动保存到 `.cache/session-<id>.json`，会话记录自己的 `workspace_id`；
Agent 的工作区注册表在 `.cache/workspaces.json`。Agent 左侧工作区行悬停可以改名、删除或
直接在该工作区新建会话；添加工作区时既可以直接输入绝对路径，也可以点【浏览…】打开本机
目录选择窗口。

Chat 和 Agent 的多个会话可以并行执行：切换会话不会中止正在流式输出的任务，后台会话会继续
接收 token 并保存自己的历史、统计和模型状态；会话列表中运行中的会话显示闪烁圆点。
同一个会话一次只允许一个生成任务，停止按钮只停止当前会话，不影响其他正在运行的会话。

打开会话时默认滚动到最新消息，并且只加载最近 5 个**用户轮次**（不是 20 条消息）；
只有用户主动向上滚动并接近顶部时才继续补更早的一页，打开会话或程序化滚动不会自动补页。
一整页仍不足一屏时，顶部保留【加载更早消息】按钮，由用户决定是否继续加载。
消息区右侧是 DSH 风格的居中紧凑轮次刻度：固定间距、可内部滚动、当前轮高亮，点击可跳转到
对应位置；如果目标轮次尚未加载，会先自动补加载。用户消息在发送时记录完整本地时间，
并随会话 JSON 持久化。

Agent 每次调用 MCP 工具时，使用**该会话所属工作区**的绝对路径作为
`params._meta["dsh/workspace"]`，而不是页面上临时的选择，避免切换工作区后工具 cwd 带错。
Agent 会话还保存完整的模型消息（包含 tool_calls、tool 结果和上一轮 assistant 回答），
下一轮请求以它为前缀再追加新用户消息，避免因丢失工具消息导致 prompt cache 从中间失效。
首次请求会注入工作区指令 baseline：工作区根目录的 `AGENTS.md` / `CLAUDE.md`（有大小上限）
和一条 runtime-context 快照（工作区绝对路径、进程 CWD、时区时钟）；后续请求按 DSH 的方式
**比较后追加**：指令文件内容变化时追加一条“替换旧指令”的 user 消息，runtime 事实变化时追加
一条新的 `<system-reminder>` 快照并声明取代旧快照；没有变化就不重复注入。工具读取/修改文件
后会重新发现工作区根到该文件之间的嵌套 `AGENTS.md` / `CLAUDE.md`，内容变化才追加更新。
时钟快照按 step 检查，默认每 10 分钟最多追加一次（工具续跑步骤也可以追加）。这些 user 上下文
只追加在历史末尾，不重写已有前缀；system prompt 同时包含按名称排序的可用工具指导，工具集合
不变时保持稳定，避免无谓的缓存失效。
会话行支持改名、删除和移动工作区。恢复时会恢复历史消息、模型选择、MCP 服务器、工具选择和思考等级；
未完成的模型/工具轮次不会自动重放。

输入框左侧的【＋】菜单集中了图片、文本文件和指令：

- **图片**：模型声明 `input_modalities` 包含 `image` 时才可选择多张图片（单张 6MB 以内、
  最多 4 张）；也可以直接在 Chat / Agent 输入框中粘贴剪贴板图片（如系统截图），
  粘贴与文件选择共用同一套校验和预览。图片以 OpenAI 兼容的 `image_url` 内容块发送；
  本地字符级模型不支持图片。聊天消息里的图片缩略图、输入框待发送图片预览都可以点击放大，
  点击遮罩、右上角关闭按钮或按 Esc 关闭。旧图片在上下文压缩时按占位符或摘要处理，不重复按
  base64 字符数计费。
- **文本文件**：只接受 `.txt` / `.md`，单文件 1MB 以内。浏览器拿不到普通文件的原始绝对路径，
  所以选中的文件会先由服务端写到 `.cache\attachments\<会话 ID>\<uuid>.<扩展名>`，
  再把该缓存文件的绝对路径作为 `[附件] ...` 行追加到用户消息末尾；发送时附件 chip 会清空。
  附件目录随会话删除而清理，超过 7 天的残留目录会在后续上传时自动清理。
- **指令**：Agent 可执行「压缩上下文」（同 `/compact`），Chat / Agent 都可以导出当前会话为
  Markdown；也可以从这里再次打开模型与思考等级菜单。

> 附件路径只是字符串：只有配了文件工具的 Agent 才可能真正读取它；Chat API 没有文件工具，
> 只能看到这行路径。`.cache` 不在 Agent 工作区时，MCP 文件工具也可能拒绝访问。

### 接入 OpenAI 兼容 API（可选）

「API」页可以保存多个 OpenAI 兼容服务配置：

```text
名称
Base URL（例如 https://api.openai.com/v1）
API Key 环境变量名
```

不再手工填写模型名；模型能力优先从 `/models` 自动检测。检测不到的上下文、最大输出、
工具能力会显示为“未知”，不会伪装成固定值。点击 API 页的模型行可以手工覆盖上下文、
最大输出、工具能力、输入类型和思考档位；手工值保存在 `.cache\openai_model_overrides.json`，
后续自动检测不会覆盖。`GET /models` 不消耗生成 Token，但会产生一次 HTTP 请求和可能的速率计数。

配置保存在 `.cache\openai_apis.json`；检测结果保存在 `.cache\openai_status.json`。
**只保存环境变量名，不保存 Key 本身**；实际 Key 请在启动 WebUI 前设置到进程环境，例如：
Ollama 等本地服务可以把环境变量名留空，此时不发送 `Authorization`。

```bat
set OPENAI_API_KEY=sk-...
start.bat
```

配置后，Chat 下拉会按 `DeepSeek/deepseek-flash（API）` 的形式显示每个模型；工具能力为
“支持”或“未知”的模型会出现在 Agent，明确“不支持”的模型不会出现。服务页会列出全部 API
模型（对话 / 嵌入），桥接服务会代理
`/v1/chat/completions` 和 `/v1/embeddings`，embedding 请求会写入服务页的「向量」日志。
API 页只有一个【检测模型】按钮，只重新请求当前选中 API 的 `/models`；启动时仍然自动检测
全部已保存 API。模型列表会同时显示类型、工具能力、输入类型、思考档位和上下文上限。
Ollama 配置的聊天请求超时为 600 秒，其他 API 供应商为 300 秒，避免本地大模型思考时间较长
被误判为断线。
上下文优先读取 `/models` 的 `context_length` / `context_window` / `max_model_len` /
`top_provider.context_length` 等字段；读不到时内部按保守 budget 运行，但界面显示“未知（供应商默认）”。
请求历史会按上下文预留输出空间后压缩：按 DSH 的 80% 阈值和 16% 近期保留预算，优先把
旧历史替换为检查点摘要；摘要请求复用原 system prompt、tools 和旧消息前缀，只把压缩指令
追加在末尾，尽量命中原有 prompt cache。压缩前会修剪超大的工具结果；摘要失败时退回确定性
裁剪，并保持 `tool_calls` / `tool_result` 成对。上下文溢出错误会先强制压缩并重试一次。
Agent 输入框支持 `/compact`，也可以从输入框【＋】→「压缩上下文」执行同样的手动压缩。
注意 Agent 的“输出达到 max_tokens 上限”是单轮输出上限，不是上下文上限。
输入区不再有“新对话”按钮，统一从左侧栏或收起后的侧栏【＋】新建会话。
输入框下方是一条状态栏：左侧圆环和百分比显示上下文占用（按字符数近似，图片按固定视觉 token
估算），中间显示缓存命中率，右侧显示本轮和会话的输入/输出 token；点击任一项会弹出更详细的说明，
输出上限不再默认平铺。Chat 和 Agent 的助手回答在流式生成时也会按约 100ms 节流解析 Markdown，
数学公式在生成中先按 `$...$` 原文显示，结束后再做完整 KaTeX 渲染。
API 模型完成一轮对话后，会显示供应商上报的本轮和会话累计 token：
输入（其中命中缓存 / 未命中）、输出以及缓存命中率。命中率按
`缓存读 / (未缓存输入 + 缓存读 + 缓存写)` 计算；DeepSeek 的 `prompt_cache_hit_tokens` /
`prompt_cache_miss_tokens`、OpenAI 的 `prompt_tokens_details.cached_tokens` 等字段都会
归一化处理。Agent 多个工具步骤按 token 加权汇总；没有缓存字段时显示“—”，不显示成 0%。
会话累计会写入会话 JSON，重新打开会话后仍保留。本地字符级模型和没有缓存统计的 Ollama
模型不显示 token/命中率。

---

本项目采用 **MIT** 开源协议，详见 [LICENSE](LICENSE)。

