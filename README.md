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
直接回车则退出。页面有六个标签：

| 标签 | 用途 |
|---|---|
| 训练 | 选配置和语料，点【开始训练】；日志实时滚动，同时落盘到 `.cache\train-logs\` |
| 对话 | 选配置，直接打字；回复逐字出现，可随时【停止】 |
| 服务 | 起一个 OpenAI 兼容的本地桥接服务，页面上能实时看到每次问答 |
| 环境 | 检测项目 Python、安装 / 更新 torch / numpy / transformers |
| 配置 | 在表单里改 `properties\*.properties`，保存时保留注释与键顺序 |
| 签名 | 给 `model\<名字>\` 下的标准权重文件计算或校验 SHA-256 |

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
3. 点【自动检测】：显示当前 NVIDIA 显卡/驱动，并在 PyTorch 版本文本框填入推荐值，
   例如 `2.14.1+cu126`；
4. 想换版本/环境时直接改文本框：
   - `2.14.1+cu126`：CUDA 12.6；
   - `2.14.1+cu128`：CUDA 12.8；
   - `2.14.1+cpu`：CPU；
   - 留空：不指定版本，由 pip 选择该源最新兼容版；
5. 点【安装 / 更新训练依赖】，等待 pip 日志结束；
6. 再点一次【自动检测】，确认 torch / numpy / transformers / CUDA 状态。

自动检测只依据 `nvidia-smi` 和驱动版本做启发式判断：没有 NVIDIA 显卡推荐 CPU；驱动较旧时推荐 CPU 并提示更新驱动；最终以文本框里的版本和源为准。`2.14.1` 只是默认推荐版本，不代表只能装这个版本。

依赖会安装到 `.runtime\python\Lib\site-packages`，也就是 WebUI 自己使用的项目 Python。WebUI 进程本身只 import 标准库，不 import torch，因此安装期间界面仍可用。

`make_python_env` 只会在 `.runtime\python\licenses\` 生成 CPython 与 `ensurepip` 的许可证。torch、numpy、transformers 是用户自行安装到项目 Python 的依赖，不属于随包分发内容，不收集其许可证；Python 环境目录之外的声明文件由人工维护，make 脚本不创建、不覆盖。

选择 CUDA 源会一并安装 NVIDIA 的 CUDA 运行时组件，这些组件是闭源专有软件，需要遵守 NVIDIA 许可证。

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
3. 输入内容回车发送，回复会**逐字**出现，生成中可点【停止】。

- 切换模型不用重启页面：旧模型会被释放（含显存）再加载新的；
- 训练正在跑时，**只能选不同的模型**——与训练配置同名的模型会被拒绝，避免拖慢训练或读到半成品；
- 对话上下文按模型的 `block_size` 截断，拼提示词的规则是 `用户：…` / `助手：…`；
  词表里没有这两个标记的模型会自动切「续写模式」。

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
| `save_optimizer` | `True` | `True` = checkpoint 保留优化器（可续训）；`False` = 完成后去掉 checkpoint 里的优化器（发布权重）；root 模型目录始终只有权重 |
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

体积控制在 50 MB 以内的关键是 `properties\dlqq_exam.properties` 里这两行：

```properties
# 权重存半精度，体积减半
save_dtype=fp16
# 训练完成后去掉优化器，发布权重更小
save_optimizer=false
```

实测：4,562 条语料切出约 270 万 token，13.7 M 参数存成 fp16 + 训练完成后去掉优化器约 **27 MB**；
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

- `dlqq_llm` 是 31.5 M 参数，按「每参数 20 token」≈ **6.3 亿 token** 就够；
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

专门写诗的小模型：6 层 384 维（14.1 M 参数），语料是**古诗 + 宋词 + 近现代诗**，
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
| 结构 | 6 层 384 维（14.1 M） | 约 4000 万字 / 14 M 参数 ≈ 2.8 token/参数 |
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
  http_server.py        OpenAI 兼容桥接服务（多模型懒加载）
  common.py             find_root / properties 保序读写 / 父进程看门狗 / 隐藏目录
  model.py              GPT-2 封装（transformers）
  model_signature.py    权重 SHA-256 签名（只依赖标准库）
web\                    前端页面（由 webui.py 托管）
  index.html            六个标签：训练 / 对话 / 服务 / 环境 / 配置 / 签名
  chat.html             局域网只读对话页
  vendor\               本地 xterm.js / xterm.css / LICENSE.xterm（MIT，训练日志终端）
properties\             四个训练配置
res\                    语料目录（不进版本库）
model\                  训练产物（model\<名字>\ 标准 HF 目录 + checkpoint-<step>\ + .sha256）
                        每个目录含 config.json / model.safetensors / chars.json / meta.json
.cache\                 训练缓存/日志/配置备份（目录纳入版本控制，内容忽略）
.ai_res\                AI 临时目录（不进版本库，不参与训练）
THIRD-PARTY-NOTICES.txt  Python / 依赖 / xterm 的许可证清单（人工维护）
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

服务本身是 OpenAI 兼容的：`/v1/models`、`/v1/chat/completions`（支持 `stream`），
模型清单来自 `properties\` 里能对上标准模型目录的配置（不再扫描旧 `.pt` 文件），谁被调用才加载谁；
对话参数也读同一份配置。

---

本项目采用 **MIT** 开源协议，详见 [LICENSE](LICENSE)。

