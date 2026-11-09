# -*- coding: utf-8 -*-
"""
从零训练一个字符级 GPT
=======================

整体流程（也是本文件的阅读顺序）：
    1. 解析参数 / 读 properties        → CONFIG、_parse_argv()
    2. 扫描语料，统计字频，建字符表    → prepare_data()、build_vocab()
    3. 把语料编码成 uint16 二进制      → 训练时用 np.memmap 随机读，不整份进内存
    4. 模型：transformers GPT2LMHeadModel → code/model.py 统一实现
    5. 训练循环（梯度累积 + 预热/退火）→ 验证损失变好才存盘
    6. 训练完采样一段，确认能写        → GPT.generate()

数据量再大，内存也只跟 block_size / batch_size 有关，不跟语料大小有关。

用法（在工程根目录执行；内部调试用）：
    python code\\train.py --config properties\\chat_demo.properties
    python code\\train.py data_path=res\\chat_demo         # 必须指定语料目录

吃一整份大语料（txt 或 jsonl 都行，第一次运行会先花些时间切词，之后自动复用 cache）：
    python code\\train.py data_path=res\\my_corpus out_path=model\\firefly ^
        max_vocab=8000 n_layer=8 n_head=8 n_embd=512 block_size=256 batch_size=32 max_iters=60000

jsonl 里会自动找问答字段（input/target、Q/A、question/answer、src/tgt …），
也可以用 q_field= 和 a_field= 显式指定。切词缓存放 .cache\\。
"""
import os, sys, math, time, json, hashlib, gc, random   # 标准库：路径 / 退出 / 数学 / 计时 / JSON / 哈希 / 垃圾回收

try:
    import numpy as np
except ImportError:
    # numpy 是唯一的数据处理依赖；没有它连字频都统计不了
    print("[错误] 缺少 numpy。装一个：python -m pip install numpy")
    sys.exit(1)

# Windows 控制台默认是 GBK，打印 ⚠️ / ← 这类字符会直接抛 UnicodeEncodeError。
# 把标准输出切成 UTF-8 并允许降级替换，避免训练跑完却在最后一屏崩掉。
# line_buffering：输出被重定向到文件 / IDE 时也逐行刷新，否则看起来像卡死。
for _s in ("stdout", "stderr"):                 # 两个流都处理，错误信息也要能打印
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace",
                                     line_buffering=True, write_through=True)
    except Exception:
        pass                                    # 老 Python / 特殊终端不支持就算了，不影响训练

import torch                                    # 深度学习框架，仅训练/推理需要

# ==================================================================
# 配置（properties 覆盖；内部调试时可用 key=value 覆盖）
# ==================================================================
CONFIG = dict(
    data_path   = None,      # None = 未配置，训练会直接报错；填语料目录，如 res/chat_demo
    block_size  = 128,       # 上下文长度（一次能看多少个字）
    n_layer     = 4,         # 层数
    n_head      = 4,         # 注意力头数
    n_embd      = 128,       # 隐藏维度（决定宽度）
    dropout     = 0.1,       # 随机丢弃比例，数据少时调大一点防过拟合
    batch_size  = 32,        # 每步并行多少段（显存不够就调小）
    max_iters   = 2000,      # 训练步数
    eval_interval = 200,     # 每多少步跑一次验证
    eval_iters  = 50,        # 每次验证抽多少批算平均 loss
    lr          = 3e-4,      # 初始学习率
    weight_decay= 0.1,       # AdamW 的权重衰减（只加在 2 维以上参数上）
    grad_clip   = 1.0,       # 梯度裁剪阈值，防止偶发梯度爆炸
    seed        = 1337,      # 随机种子，固定后结果可复现
    out_path    = "model/mymodel",      # 标准 HF 模型目录；配置里的 model= 会覆盖它
    # ---- 大语料相关 ----
    max_vocab   = 8000,      # 词表上限：只保留最常出现的前 N 个字符
    val_frac    = 0.05,      # 从语料尾部切这个比例做验证集
    cache_dir   = ".cache",          # 切好的二进制缓存（.cache 已纳入版本控制，内容忽略）
    prompt      = "",        # 训练结束采样用的开头；留空则自动挑
    # ---- 语料格式 ----
    data_format = "auto",    # auto / txt / jsonl；auto 按扩展名和文件开头判断
    q_field     = "",        # jsonl 里「问」的字段名；留空自动推断
    a_field     = "",        # jsonl 里「答」的字段名；留空自动推断
    chat_mode   = True,      # 语料是「用户：…／助手：…」问答格式；纯续写模型（诗词）设 False
    # ---- 对话参数（存进模型文件，chat.py / http_server 按模型取用）----
    chat_temperature = 0.8,  # 采样温度：越大越发散，越小越保守
    chat_top_k       = 40,   # 只从概率最高的 K 个里采样；0 = 不截断
    chat_max_new     = 180,  # 一次最多生成多少字
    # ---- 训练调度（长训练必需）----
    grad_accum      = 1,        # 梯度累积步数：显存不够时用它凑等效 batch
    warmup_iters    = 0,        # 学习率线性预热多少步
    lr_decay        = "cosine", # cosine = 余弦退火到 min_lr_frac；none = 恒定
    min_lr_frac     = 0.1,      # 退火终点 = lr * 这个比例
    grad_checkpoint = False,    # 梯度检查点：省显存，慢约 20-30%
    resume          = False,    # True = 从 out_path 续训（要有 iter；优化器状态可选）
    save_every      = 0,        # 每 N 步强制存一次盘（0 = 只在验证变好时存）
    save_dtype      = "fp32",   # fp32 / fp16 / bf16 —— 发到 Gitee 时用 fp16 体积减半
    save_optimizer  = True,     # true=训练完成后保留优化器（可续训）；false=完成后去掉（发布权重）
)

# ------------------------------------------------------------------
# 命令行：--config 读 properties；key=value 覆盖单项；--check-config 只校验不训练
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                  # noqa: E402
from model import GPT, CHARS_FILE, META_FILE, has_weights, write_json, read_json  # noqa: E402

ROOT = common.find_root()


def _parse_argv(argv):
    """解析内部调试参数。

    WebUI 启动训练时只会传 --config / --parent-pid；key=value 覆盖是给开发者
    临时改参数用的，不属于面向用户的入口。
    """
    cfg_file, parent_pid, check_only = None, 0, False   # 配置文件 / 父进程号 / 只检查不训练
    overrides = []                                      # key=value 形式的临时覆盖
    i = 0
    while i < len(argv):
        a = argv[i]                                     # 当前参数
        if a in ("--config", "-c") and i + 1 < len(argv):
            i += 1                                      # 跳过参数名，取下一个作为值
            cfg_file = argv[i]
        elif a.startswith("--config="):                 # 支持 --config=xxx 写法
            cfg_file = a.split("=", 1)[1]
        elif a == "--check-config":                     # 只打印关键配置，不开始训练
            check_only = True
        elif a == "--parent-pid" and i + 1 < len(argv):
            i += 1
            try:
                parent_pid = int(argv[i])               # WebUI 传父进程号，用于看门狗
            except ValueError:
                parent_pid = 0                          # 格式不对就当没传
        elif a.startswith("--parent-pid="):             # 支持 --parent-pid=1234
            try:
                parent_pid = int(a.split("=", 1)[1])
            except ValueError:
                parent_pid = 0
        elif a in ("-h", "--help"):                     # 打印模块说明后退出
            print(__doc__)
            sys.exit(0)
        elif "=" in a and not a.startswith("-"):        # 形如 max_iters=500 的覆盖项
            overrides.append(tuple(a.split("=", 1)))
        else:                                           # 不认识的参数直接报错，避免拼错后静默跑错
            print(f"[用法] 不认识的参数：{a}")
            sys.exit(2)
        i += 1
    return cfg_file, parent_pid, check_only, overrides


_CFG_FILE, _PARENT_PID, _CHECK_ONLY, _OVERRIDES = _parse_argv(sys.argv[1:])
# 第一行就打印解释器：训练日志和 WebUI 里都能直接确认用的是哪个 Python
print(f"[环境] 解释器：{sys.executable}", flush=True)

if _CFG_FILE:
    # --config 既可以是配置名，也可以是相对/绝对路径；统一交给 common 解析
    _cfg_path = common.resolve_path(ROOT, _CFG_FILE)
    _props = common.load_properties(_cfg_path)      # 保序读取，保留注释与空行
    if not _props:
        print(f"[错误] 配置文件为空或读不到：{_cfg_path}")
        sys.exit(1)
    # 把 properties 的值套到 CONFIG 上；旧配置遗留的 python= 会被忽略
    _ignored = common.apply_properties(CONFIG, _props, root=ROOT)
    print(f"[配置] 已加载 {os.path.basename(_cfg_path)}"
          + (f"（忽略 {', '.join(sorted(_ignored))}）" if _ignored else ""))

# 命令行 key=value 覆盖：只允许 CONFIG 里已有的键，类型按默认值推断
for _k, _v in _OVERRIDES:
    if _k not in CONFIG:
        print(f"[警告] 未知配置项 {_k}，忽略")
        continue
    try:
        CONFIG[_k] = common.coerce_like(_v, CONFIG[_k])   # "500"→int，"0.1"→float
    except (TypeError, ValueError):
        print(f"[警告] 配置项 {_k}={_v!r} 类型不对，忽略")
        continue
    print(f"[覆盖] {_k} = {CONFIG[_k]}")

# --check-config：只打印关键参数，给 WebUI / 调试确认配置是否读对
if _CHECK_ONLY:
    print("[配置] 关键参数：")
    for _key in ("data_path", "out_path", "n_layer", "n_head", "n_embd", "block_size",
                 "batch_size", "max_iters", "max_vocab", "val_frac", "resume", "cache_dir"):
        print(f"        {_key} = {CONFIG[_key]}")
    sys.exit(0)

# WebUI 启动子进程时带 --parent-pid；父进程消失后看门狗让本进程退出
if _PARENT_PID:
    common.start_parent_watchdog(_PARENT_PID)

# ==================================================================
# 1) 数据：字符级词表 + 一次性切成二进制（内存占用与语料大小无关）
# ==================================================================
# 1 GB 中文大约 3.5 亿字。如果像以前那样先 read() 再 encode() 成 Python 列表，
# 光这一份就要吃掉十几 GB 内存，必然 OOM。所以这里改成：
#     流式统计字频 → 分块编码 → 写 uint16 二进制 → 训练时 np.memmap 随机读
# 峰值内存只取决于单个分块（几十 MB），跟语料是 1 MB 还是 1 GB 无关。
CHUNK_CHARS = 4_000_000        # 每次读多少字符（4M 字 ≈ 16 MB 的中间数组）

USER_TAG, ASSIST_TAG = "用户：", "助手："      # 和 code/chat.py 的对话模式保持一致

# jsonl 里常见的「问 / 答」字段名，按顺序尝试
FIELD_CANDIDATES = [
    ("input", "target"), ("q", "a"), ("question", "answer"),
    ("src", "tgt"), ("instruction", "output"), ("prompt", "response"),
    ("query", "response"), ("ask", "answer"), ("title", "content"),
]


def detect_encoding(path):
    """自动识别编码 —— Windows 记事本存出来的 txt 通常是 GBK，不是 UTF-8"""
    # 按“最可能且最严格”的顺序试；utf-8-sig 放前面会把带 BOM 的文件读成无 BOM
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5"):
        try:
            with open(path, encoding=enc) as f:
                f.read(2_000_000)       # 只读前 2M 字符，足够判断常见编码
            return enc                  # 能完整解码就认为命中
        except UnicodeDecodeError:
            continue                    # 解不开就换下一种
    return None                         # 都失败，调用方会跳过这个文件


def iter_chunks(path, enc, size=CHUNK_CHARS):
    """按字符数分块读，永远不会把整个文件读进内存"""
    with open(path, encoding=enc, errors="replace") as f:   # 坏字节用 � 顶替，不中断
        while True:
            s = f.read(size)            # 一次最多读 size 个字符
            if not s:                   # 读不到内容 = 文件结束
                return
            yield s                     # 把这一块交给调用方，读完再回来要下一块


def detect_format(path):
    """auto 模式：看扩展名，再看文件开头是不是 { 或 ["""
    if CONFIG["data_format"] in ("txt", "jsonl"):
        return CONFIG["data_format"]    # 配置显式指定，直接听配置
    if path.lower().endswith((".jsonl", ".ndjson", ".json")):
        return "jsonl"                  # 扩展名已经说明是 JSON 系列
    with open(path, "rb") as f:
        head = f.read(4096)             # 只读开头 4 KB，用来猜格式
    if head.startswith(b"\xef\xbb\xbf"):        # 去掉 UTF-8 BOM 再判断
        head = head[3:]
    return "jsonl" if head.lstrip()[:1] in (b"{", b"[") else "txt"


def detect_fields(path, enc):
    """从 jsonl 的前几行推断问答字段名；q_field / a_field 显式给了就校验后直接用"""
    qf, af = CONFIG["q_field"], CONFIG["a_field"]

    with open(path, encoding=enc, errors="replace") as f:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line[:1] == "[":
                print("[数据] x 这是 JSON 数组，不是 JSONL（一行一条）")
                print("[数据]   用脚本转成一行一条再喂，或者 data_format=txt 自己拼成纯文本")
                sys.exit(1)
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            keys = list(obj.keys())
            if qf and af:
                # 显式指定了字段就校验一下，免得写错名字后默默丢掉整个语料
                if qf not in obj or af not in obj:
                    print(f"[数据] x 指定的字段 q_field={qf} a_field={af} 不在数据里")
                    print(f"[数据]   这一行的字段是：{keys}")
                    sys.exit(1)
                return qf, af
            lower = {k.lower(): k for k in keys}
            for cq, ca in FIELD_CANDIDATES:
                if cq in lower and ca in lower:
                    return lower[cq], lower[ca]
            strs = [k for k in keys if isinstance(obj.get(k), str)]
            # 只有一段长文本的 jsonl（{"text": "..."}）当纯文本用，不加问答标签
            for k in ("text", "content", "raw_content", "document"):
                if isinstance(obj.get(k), str) and obj[k].strip():
                    return "", ""
            if len(strs) >= 2:
                return strs[0], strs[1]
            print(f"[数据] x 这一行的字段是 {keys}，找不到可用的问答字段")
            print("[数据]   用 q_field=字段名 a_field=字段名 显式指定")
            sys.exit(1)

    print("[数据] x 前 50 行都解析不出 JSON 对象，确认这是 JSONL 文件")
    print("[数据]   如果其实是纯文本，加 data_format=txt")
    sys.exit(1)


def _pick_plain(obj):
    """纯文本 jsonl 里取正文的字段"""
    for k in ("text", "content", "raw_content", "document", "input"):
        v = obj.get(k)                            # 按常见字段名依次尝试
        if isinstance(v, str) and v.strip():      # 非空字符串才算有效正文
            return v
    return None                                   # 没有正文就返回 None


def iter_text_chunks(path, enc, fmt, qf, af, stats=None):
    """按块产出纯文本。
    txt  : 原样分块读。
    jsonl: qf/af 为空 = 纯文本 jsonl（每行一段正文，原样拼）；
           否则按问答拼成「用户：…／助手：…」—— 直接喂 jsonl 会把括号引号学进文法里。
    stats: 可选 dict，用来回传 (处理条数, 坏行数)。
    """
    if fmt == "txt":
        yield from iter_chunks(path, enc)   # 纯文本不用解析，直接按块转交
        return

    n_ok = n_bad = 0                         # 成功条数 / 坏行数

    # ---- 纯文本 jsonl：一行一段，原样拼起来 ----
    if not qf and not af:
        buf, buf_len = [], 0                 # 攒够 CHUNK_CHARS 再一次性 yield
        with open(path, encoding=enc, errors="replace") as f:
            for line in f:
                line = line.strip()          # 去掉行尾换行/空白
                if not line:                 # 空行直接跳过
                    continue
                try:
                    obj = json.loads(line)   # 一行必须是一个 JSON 对象
                except Exception:
                    n_bad += 1               # 坏行计数，不中断整个文件
                    continue
                s = _pick_plain(obj) if isinstance(obj, dict) else None   # 取正文字段
                if not s:                    # 没有正文字段也当坏行
                    n_bad += 1
                    continue
                s = s.strip()                # 去掉正文两端空白
                buf.append(s + "\n")         # 每段之间补一个换行，避免粘成一句
                buf_len += len(s) + 1
                n_ok += 1
                if buf_len >= CHUNK_CHARS:   # 攒够一块就交出去
                    if stats is not None:
                        stats["ok"], stats["bad"] = n_ok, n_bad
                    yield "".join(buf)
                    buf, buf_len = [], 0     # 清空缓冲，继续下一块
        if buf:                              # 最后不足一块的尾部
            yield "".join(buf)
        if stats is not None:
            stats["ok"], stats["bad"] = n_ok, n_bad
        return

    # ---- 问答 jsonl：拼成「用户：…／助手：…」 ----
    buf, buf_len = [], 0
    with open(path, encoding=enc, errors="replace") as f:
        for line in f:
            line = line.strip()              # 去掉行尾空白
            if not line:
                continue
            try:
                obj = json.loads(line)       # 解析一行 JSON
                q = str(obj.get(qf, "") or "").strip()   # 问题字段，缺省空串
                a = str(obj.get(af, "") or "").strip()   # 答案字段，缺省空串
            except Exception:
                n_bad += 1                   # JSON 坏了，跳过
                continue
            if not q or not a:               # 问或答为空，没法组成样本
                n_bad += 1
                continue
            buf.append(f"{USER_TAG}{q}\n{ASSIST_TAG}{a}\n")   # 统一成对话格式
            buf_len += len(q) + len(a) + 8  # +8 是标签和换行的近似长度
            n_ok += 1
            if buf_len >= CHUNK_CHARS:       # 攒够一块就交出去
                if stats is not None:
                    stats["ok"], stats["bad"] = n_ok, n_bad
                yield "".join(buf)
                buf, buf_len = [], 0
    if buf:                                  # 最后一块
        yield "".join(buf)
    if stats is not None:
        stats["ok"], stats["bad"] = n_ok, n_bad


CORPUS_EXTS = (".jsonl", ".json", ".txt")


def list_corpus_files(path):
    """语料统一放在一个目录里：目录下可以放多个 .jsonl / .txt，
    按路径排序依次读取，子目录也会扫（跳过隐藏目录）。"""
    if not os.path.isdir(path):
        return []                                   # 不是目录就当没有语料
    files = []                                      # 收集所有可读的语料文件
    for root, dirs, names in os.walk(path):         # 递归遍历子目录
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))   # 跳过 .git/.cache 这类隐藏目录
        for n in sorted(names):                     # 文件名排序，保证每次读的顺序一致
            if n.lower().endswith(CORPUS_EXTS) and not n.startswith("."):
                files.append(os.path.join(root, n)) # 只收 .jsonl/.json/.txt，跳过隐藏文件
    return sorted(files)                            # 最终再按完整路径排序


def resolve_corpus_files():
    """确定语料文件：data_path 必须是一个目录；没有配置就直接报错。"""
    p = CONFIG["data_path"]                         # 配置里的语料目录
    if p:
        if os.path.isfile(p):                       # 只接受目录，避免用户把单个文件填进来
            print(f"[数据] x data_path 是文件，现在只接受文件夹：{p}")
            print(f"[数据]   建一个目录把它放进去，然后把 data_path 指向目录，例如：")
            name = os.path.splitext(os.path.basename(p))[0]
            print(f"[数据]     mkdir res\\{name}  &&  move \"{p}\" res\\{name}\\")
            print(f"[数据]     data_path=res/{name}")
            sys.exit(1)
        files = list_corpus_files(p)                # 扫描目录下的所有语料文件
        if files:
            return files                            # 有文件就正常返回
        if os.path.isdir(p):
            print(f"[数据] x 目录里没有 .jsonl / .txt：{p}")
        else:
            print(f"[数据] x 找不到目录：{p}")
        print(f"[数据]   当前工作目录：{os.getcwd()}")
        print('[数据]   路径带反斜杠的话，用引号包起来试试，例如 data_path="E:/我的语料"')
        print("=" * 62)
        print("  没有语料，训练无法开始。")
        print("=" * 62)
        sys.exit(1)

    print("=" * 62)
    print("[数据] x 没有配置 data_path")
    print('       请在 properties\\<配置>.properties 里写 data_path=res\\<语料目录>')
    print("=" * 62)
    sys.exit(1)


def inspect_corpus(files):
    """给每个文件定好编码 / 格式 / 问答字段，并记下大小和修改时间（缓存指纹要用）"""
    spec = []                                       # 每个文件的“读取说明书”
    for path in files:
        enc = detect_encoding(path)                 # 先猜编码
        if enc is None:
            print(f"[数据] ! 跳过（编码认不出）：{path}")
            continue                                # 猜不出编码的文件直接跳过
        fmt = detect_format(path)                   # 判断 txt / jsonl
        qf = af = ""                                # jsonl 的问答字段名
        if fmt == "jsonl":
            qf, af = detect_fields(path, enc)       # 从样本行推断字段
        spec.append(dict(path=path, enc=enc, fmt=fmt, qf=qf, af=af,
                         size=os.path.getsize(path), mtime=int(os.path.getmtime(path))))
    if not spec:
        print("[数据] x 没有可用的语料文件")
        sys.exit(1)
    return spec                                     # 交给后面的统计/编码流程


def describe_spec(item):
    """一行描述：文件名 + 大小 + 格式/字段"""
    name = os.path.basename(item["path"])          # 只显示文件名，不打印完整路径
    mb = item["size"] / 1e6                        # 字节转 MB
    if item["fmt"] == "txt":
        how = "txt"                                # 纯文本
    elif not item["qf"]:
        how = "jsonl 纯文本"                       # jsonl 但没有问答字段
    else:
        how = f"jsonl 问答 {item['qf']}→{item['af']}"   # 问答 jsonl
    return f"  {name:<32} {mb:>9.1f} MB  {how}"    # 对齐成表格的一行


def count_char_freq(spec):
    """流式统计每个字符出现多少次 —— 用 numpy 一次算一整块，比 Python 循环快几百倍"""
    counts = np.zeros(0x110000, dtype=np.int64)     # 覆盖全部 Unicode 码位，下标就是码点
    done, t0 = 0, time.time()                       # 已扫描字符数 / 起始时间
    for item in spec:
        for s in iter_text_chunks(item["path"], item["enc"], item["fmt"],
                                  item["qf"], item["af"]):
            # 转成 UTF-32 再按 4 字节整数读，numpy 就能直接按码点统计
            cps = np.frombuffer(s.encode("utf-32-le", errors="ignore"), dtype="<u4")
            counts += np.bincount(cps, minlength=0x110000)   # 累加到总频次
            done += len(s)                          # 进度按字符数算
            print(f"\r[词表] 已扫描 {done:,} 字符 | {time.time() - t0:5.1f}s", end="", flush=True)
    print()                                         # 进度条换行
    return counts                                   # 长度 0x110000 的频次表


def build_vocab(counts, max_vocab):
    """按出现次数取前 max_vocab 个字符；换行/制表保留，其它控制字符丢掉。
    限制词表大小是必须的：GB 级语料里会混进几万个生僻字和符号，
    词表一大，输出层参数就爆炸，而且那些字在训练里根本学不好。"""
    idx = np.nonzero(counts)[0]                     # 所有出现过的码点
    # 保留可打印字符 + 换行(10) + 制表(9)，其它控制字符不参与训练
    keep = idx[(idx >= 32) | (idx == 10) | (idx == 9)]
    if len(keep) == 0:
        return []                                   # 没有任何可用字符
    order = keep[np.argsort(counts[keep])[::-1]]    # 按频次从高到低排序
    if max_vocab and len(order) > max_vocab:
        order = order[:max_vocab]                   # 只留前 max_vocab 个
    return [chr(int(i)) for i in order]             # 码点转回字符，顺序即 id


def tokenize_corpus(spec, chars, bin_path):
    """把语料编码成 uint16 序列写进二进制文件；词表外的字直接丢掉。
    返回 (写出的 token 数, 被丢掉的字符数, 跳过的坏行数, 解析出的条数)。"""
    codes = np.array([ord(c) for c in chars], dtype=np.uint32)   # 词表字符的码点
    order = np.argsort(codes)                       # 按码点排序，才能二分查找
    codes_sorted = codes[order]                     # 升序码点，供 searchsorted 用
    ids_of_sorted = order.astype(np.uint16)         # 排序位置 → chars 里的原 id

    total = dropped = bad_total = ok_total = 0      # 写出 token / 丢弃字符 / 坏行 / 有效条数
    t0 = time.time()
    with open(bin_path, "wb") as out:               # 直接写二进制，不经过 Python 列表
        for item in spec:
            stats = {"ok": 0, "bad": 0}             # 当前文件的解析统计
            for s in iter_text_chunks(item["path"], item["enc"], item["fmt"],
                                      item["qf"], item["af"], stats):
                # 转成 UTF-32 码点数组，一次查找一整块
                cps = np.frombuffer(s.encode("utf-32-le", errors="ignore"), dtype="<u4")
                pos = np.searchsorted(codes_sorted, cps)         # 每个字在词表里的插入位置
                np.clip(pos, 0, len(codes_sorted) - 1, out=pos)  # 防止越界
                hit = codes_sorted[pos] == cps      # 真的命中（不是被 clip 挤到边界）
                ids = ids_of_sorted[pos[hit]]       # 命中的字转成 uint16 id
                out.write(ids.tobytes())            # 连续写进 .bin
                total += len(ids)                   # 累计写出量
                dropped += len(cps) - len(ids)      # 词表外被丢掉的字符数
                msg = f"\r[编码] 已写出 {total:,} token"
                if item["fmt"] == "jsonl" and item["qf"]:
                    msg += f" | 已处理 {ok_total + stats['ok']:,} 条"
                print(f"{msg} | {time.time() - t0:5.1f}s", end="", flush=True)
            bad_total += stats["bad"]               # 汇总坏行
            ok_total += stats["ok"]                 # 汇总有效条数
    print()
    return total, dropped, bad_total, ok_total


def prepare_data():
    """准备训练数据：优先复用 cache；语料或词表设置变了就重建。
    data_path 只能是目录（里面放多个 .jsonl / .txt）。
    返回 (chars, bin 路径, 训练 token 数, 总 token 数)。"""
    files = resolve_corpus_files()                  # 先确定要读哪些文件
    spec = inspect_corpus(files)                    # 再确定每个文件的编码/格式/字段

    total_size = sum(x["size"] for x in spec)       # 语料总字节数，只用于打印
    is_dir = bool(CONFIG["data_path"]) and os.path.isdir(CONFIG["data_path"])
    os.makedirs(CONFIG["cache_dir"], exist_ok=True) # 缓存目录不存在就建

    # 缓存指纹要把「目录里每个文件的名字/大小/修改时间 + 格式/字段」都算进去：
    # 往目录里新丢一个 jsonl，指纹就变，自动重新切词。
    data_root = os.path.abspath(CONFIG["data_path"]) if CONFIG["data_path"] else ""
    want = dict(
        source=data_root,                            # 语料根目录
        # 每个文件只记相对路径/大小/mtime/格式/字段，不把内容塞进指纹
        files=[[os.path.relpath(x["path"], data_root) if is_dir else os.path.basename(x["path"]),
                x["size"], x["mtime"], x["fmt"], x["qf"], x["af"]] for x in spec],
        max_vocab=CONFIG["max_vocab"],              # 词表大小也会影响编码结果
        val_frac=CONFIG["val_frac"],                # 验证集比例影响切分点
    )
    # 指纹取 md5 前 10 位，够区分不同语料/参数组合，又不会让文件名太长
    tag = hashlib.md5(json.dumps(want, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    bin_path = os.path.join(CONFIG["cache_dir"], f"data-{tag}.bin")
    meta_path = os.path.join(CONFIG["cache_dir"], f"meta-{tag}.json")

    meta = None                                     # 命中缓存时放 meta
    if os.path.isfile(bin_path) and os.path.isfile(meta_path):
        try:
            old = json.load(open(meta_path, encoding="utf-8"))
            # 指纹完全一致且 total 有效，才认为缓存可用
            if all(old.get(k) == v for k, v in want.items()) and old.get("total"):
                meta = old
        except Exception:
            meta = None                             # meta 坏了就当没缓存

    if meta:
        print(f"[数据] 复用缓存 {bin_path}（这份语料切过词了，跳过）")
    else:
        if is_dir:
            print(f"[数据] 语料目录 {CONFIG['data_path']}（{len(spec)} 个文件，"
                  f"共 {total_size / 1e6:.1f} MB）")
            for item in spec:
                print("[数据]" + describe_spec(item))
        else:
            print(f"[数据] 语料 {spec[0]['path']}（{total_size / 1e6:.1f} MB）")
            print("[数据]" + describe_spec(spec[0]).strip())
        counts = count_char_freq(spec)              # 第一遍：统计字频
        chars = build_vocab(counts, CONFIG["max_vocab"])   # 第二遍：按频次建字符表
        if not chars:
            print("[数据] x 语料里没有任何可用字符")
            sys.exit(1)
        print(f"[词表] 保留 {len(chars)} 个字符（上限 {CONFIG['max_vocab']}）")
        total, dropped, bad, ok = tokenize_corpus(spec, chars, bin_path)   # 第三遍：编码写盘
        if ok or bad:
            print(f"[数据] 共解析 {ok:,} 条" + (f"，跳过 {bad:,} 行（空/坏/字段缺）" if bad else ""))
        if total < 1000:                            # 少于 1000 token 基本没法训
            print(f"[数据] x 编码后只剩 {total} 个 token，太少，检查语料或字段设置")
            sys.exit(1)
        n_train = int(total * (1 - CONFIG["val_frac"]))   # 尾部留一部分做验证
        meta = dict(want, chars=chars, total=total, n_train=n_train,
                    dropped=dropped, built=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)  # 记下词表和切分点，下次直接复用
        print(f"[数据] 编码完成 → {bin_path}"
              f"（{os.path.getsize(bin_path) / 1e6:.1f} MB，以后直接用这份）")

    chars = meta["chars"]                           # 词表（字符 -> id 的顺序）
    total, n_train = meta["total"], meta["n_train"] # 总 token / 训练 token
    dropped = meta.get("dropped", 0)
    print(f"[数据] 词表 {len(chars)} | 总量 {total:,} token | 训练 {n_train:,}"
          f" | 验证 {total - n_train:,}"
          + (f" | 词表外丢弃 {dropped:,} 字" if dropped else ""))
    if total < 100_000:
        print(f"[警告] 语料只有 {total:,} token，偏小 —— 模型会很快背下来（过拟合）。")
        print("       建议至少 10 万 token；想看出泛化，100 万以上更好。")
    return chars, bin_path, n_train, total


chars, DATA_BIN, N_TRAIN, N_TOTAL = prepare_data()   # 词表 / 二进制路径 / 训练 token / 总 token
VOCAB = len(chars)                                  # 词表大小，也是输出层维度
stoi = {c: i for i, c in enumerate(chars)}          # 字符 -> id
itos = {i: c for c, i in stoi.items()}              # id -> 字符
encode = lambda s: [stoi[c] for c in s if c in stoi]      # 编码：词表外的字直接跳过
decode = lambda ids: "".join(itos[i] for i in ids)        # 解码：id 拼回字符串

# 内存映射：不加载进内存，操作系统按需把用到的页换进来
train_data = np.memmap(DATA_BIN, dtype=np.uint16, mode="r",
                       offset=0, shape=(N_TRAIN,))            # 前 N_TRAIN 个 token
val_data = np.memmap(DATA_BIN, dtype=np.uint16, mode="r",
                     offset=N_TRAIN * 2, shape=(N_TOTAL - N_TRAIN,))  # 尾部验证集（uint16 = 2 字节）
print(f"[数据] 已内存映射 {DATA_BIN}（不会占用等量内存）")

# 语料太短的话 block_size 会自动缩小 —— 否则取随机起点时会拿到负数范围
_safe_T = min(len(train_data), len(val_data)) - 2
if CONFIG["block_size"] > _safe_T:
    print(f"[警告] 语料太小（训练 {len(train_data)} / 验证 {len(val_data)} token），"
          f"block_size 从 {CONFIG['block_size']} 自动降到 {max(8, _safe_T)}")
    CONFIG["block_size"] = max(8, _safe_T)

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[设备] {DEVICE}" + (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else ""))
USE_AMP = DEVICE == "cuda"          # GPU 上用 bf16 混合精度；CPU 上用会很慢（见之前的实测）

def get_batch(split):
    """从内存映射里随机取一批片段 —— 一次只读几 KB，语料多大都一样快"""
    d = train_data if split == "train" else val_data   # 训练集还是验证集
    B, T = CONFIG["batch_size"], CONFIG["block_size"]  # 批大小 / 上下文长度
    if len(d) <= T + 1:                              # 至少要有 T+1 个 token 才能切出 x/y
        raise RuntimeError(
            f"[{split}] 可用 token 只剩 {len(d)} 个，装不下 block_size={T} 的一段。"
            f"请把 block_size 调小，或者换更大的语料。")
    # dtype 必须写 int64：Windows 上 np.random.randint 默认走 int32，
    # 语料超过 21 亿 token 时直接报 "high is out of bounds for int32"
    ix = np.random.randint(0, len(d) - T - 1, size=B, dtype=np.int64)
    x = np.stack([d[i:i + T] for i in ix]).astype(np.int64)          # 输入片段
    y = np.stack([d[i + 1:i + T + 1] for i in ix]).astype(np.int64)  # 目标是「下一个字符」
    return (torch.from_numpy(x).to(DEVICE),          # numpy -> torch，搬到 GPU/CPU
            torch.from_numpy(y).to(DEVICE))

# ==================================================================
# 2) 模型：transformers 的 GPT2LMHeadModel（实现在 code/model.py）
# ==================================================================
# GPT 已从 model.py 导入；这里不再复制网络定义。

# 按配置里的结构建模型；grad_checkpoint 打开时反向会重算激活
model = GPT(VOCAB, CONFIG["block_size"], CONFIG["n_layer"],
            CONFIG["n_head"], CONFIG["n_embd"], CONFIG["dropout"],
            grad_checkpoint=bool(CONFIG["grad_checkpoint"])).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())  # 统计参数量，方便估显存
print(f"[模型] 参数量 {n_params/1e6:.2f} M")

# ==================================================================
# 3) 优化器：2 维以上参数才加 weight decay（标准做法）
# ==================================================================
decay, no_decay = [], []                          # 分两组：要衰减 / 不衰减
for p in model.parameters():
    if not p.requires_grad: continue              # 冻结参数不参与优化
    # 矩阵/卷积权重用 weight decay；LayerNorm、bias 等 1 维参数不衰减
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW([{"params": decay,    "weight_decay": CONFIG["weight_decay"]},
                         {"params": no_decay, "weight_decay": 0.0}], lr=CONFIG["lr"])

# ------------------------------------------------------------------
# 学习率调度：线性预热 → 余弦退火
# ------------------------------------------------------------------
def lr_at(it):
    """返回第 it 步应该用的学习率。"""
    base = CONFIG["lr"]                           # 初始学习率
    warm = int(CONFIG["warmup_iters"])            # 预热步数
    total = max(1, int(CONFIG["max_iters"]))      # 总步数（至少 1，防止除零）
    if warm > 0 and it < warm:                    # 预热阶段：从 0 线性升到 base
        return base * (it + 1) / warm
    if CONFIG["lr_decay"] != "cosine":            # 不衰减就一直是 base
        return base
    min_lr = base * float(CONFIG["min_lr_frac"])  # 退火终点
    span = max(1, total - warm)                   # 预热之后还剩多少步
    t = min(1.0, max(0.0, (it - warm) / span))    # 归一化到 [0,1]
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * t))  # 余弦曲线

# 用标准 LambdaLR 承载学习率曲线：scheduler.pt 才能真正续训
scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt, lr_lambda=lambda step: lr_at(step) / max(float(CONFIG["lr"]), 1e-12))

@torch.no_grad()                                 # 验证不需要梯度
def estimate_loss():
    """在训练集/验证集上各抽 eval_iters 批，返回平均 loss。"""
    model.eval()                                  # 切到推理模式（关 dropout）
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(CONFIG["eval_iters"])  # 存每一批的 loss
        for i in range(CONFIG["eval_iters"]):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                _, loss = model(*get_batch(split))  # 前向算 loss
            losses[i] = loss.item()               # 取值存起来
        out[split] = losses.mean().item()         # 这一轮的平均 loss
    model.train()                                 # 切回训练模式
    return out

# ==================================================================
# 4) 训练循环（梯度累积 + 学习率调度 + 断点续训）
# ==================================================================
print("[训练] 开始")
# 输出目录不存在就先建出来；保存的是 HuggingFace 标准目录
os.makedirs(CONFIG["out_path"], exist_ok=True)

def _save_model_dir(dest):
    """按 save_dtype 保存一个标准 HF 目录，保存后恢复训练精度。"""
    dt = str(CONFIG.get("save_dtype", "fp32")).lower()
    old_dtype = next(model.parameters()).dtype
    if dt in ("fp16", "half"):
        model.net.to(torch.float16)
    elif dt in ("bf16", "bfloat16"):
        model.net.to(torch.bfloat16)
    try:
        model.save_pretrained(dest)
    finally:
        model.net.to(old_dtype)


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(path):
    if not os.path.isfile(path):
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
        print("[续训] 已恢复随机数状态")
    except (KeyError, TypeError) as e:
        print(f"[续训] 随机数状态不完整，已跳过：{e}")


def _checkpoints(out_dir):
    """返回 [(step, path)]，按 step 升序。"""
    found = []
    if not os.path.isdir(out_dir):
        return found
    for name in os.listdir(out_dir):
        if not name.startswith("checkpoint-"):
            continue
        p = os.path.join(out_dir, name)
        try:
            step = int(name.split("-", 1)[1])
        except ValueError:
            continue
        if os.path.isdir(p):
            found.append((step, p))
    return sorted(found)


def save_ckpt(it, best, finished=False, update_root=True, global_step=None):
    """保存 Trainer 风格 checkpoint 目录。

    checkpoint-<step>/
        config.json / model.safetensors   HF 标准权重
        chars.json / meta.json            项目元数据
        optimizer.pt                      AdamW 状态
        scheduler.pt                      LambdaLR 状态
        trainer_state.json                global_step / best_metric
        rng_state.pth                     Python / numpy / torch / CUDA 随机状态

    根目录 model/<名字>/ 只保留最佳模型，供对话和服务直接加载。
    """
    out = CONFIG["out_path"]
    step = int(it if global_step is None else global_step)
    ckpt = os.path.join(out, f"checkpoint-{step}")
    os.makedirs(ckpt, exist_ok=True)

    _save_model_dir(ckpt)
    write_json(os.path.join(ckpt, CHARS_FILE), chars)
    write_json(os.path.join(ckpt, META_FILE), {"config": CONFIG})
    write_json(os.path.join(ckpt, "trainer_state.json"),
               {"global_step": step, "best_metric": best})

    keep_opt = (not finished) or bool(CONFIG.get("save_optimizer", True))
    if keep_opt:
        torch.save(opt.state_dict(), os.path.join(ckpt, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(ckpt, "scheduler.pt"))
        torch.save(_rng_state(), os.path.join(ckpt, "rng_state.pth"))

    if update_root:
        _save_model_dir(out)
        write_json(os.path.join(out, CHARS_FILE), chars)
        write_json(os.path.join(out, META_FILE), {"config": CONFIG})
        # 旧版单文件续训状态已被 checkpoint 目录取代，存在就清掉
        legacy_state = os.path.join(out, "training_state.pt")
        if os.path.isfile(legacy_state):
            os.remove(legacy_state)


def strip_optimizer(out_dir):
    """发布权重：去掉 checkpoint 里的优化器状态。"""
    removed = 0
    for _step, ckpt in _checkpoints(out_dir):
        path = os.path.join(ckpt, "optimizer.pt")
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
    if removed:
        print(f"[存档] 训练完成，已从 {removed} 个 checkpoint 去掉优化器")
    else:
        print("[存档] 已是发布权重（不含优化器）")
    return True


best_val = float("inf")                           # 历史最好验证损失
start_it = 0                                      # 从第几步开始
saved = False                                     # 是否至少存过一次

if CONFIG["resume"] and os.path.isdir(CONFIG["out_path"]):
    ckpts = _checkpoints(CONFIG["out_path"])
    try:
        if ckpts:
            _step, ckpt_path = ckpts[-1]
            model.load_weights(ckpt_path)
            opt_path = os.path.join(ckpt_path, "optimizer.pt")
            if os.path.isfile(opt_path):
                opt.load_state_dict(torch.load(opt_path, map_location=DEVICE,
                                               weights_only=False))
            sched_path = os.path.join(ckpt_path, "scheduler.pt")
            if os.path.isfile(sched_path):
                scheduler.load_state_dict(torch.load(sched_path, map_location="cpu",
                                                     weights_only=False))
            state_path = os.path.join(ckpt_path, "trainer_state.json")
            _st = read_json(state_path, {}) or {}
            start_it = int(_st.get("global_step", _step))
            best_val = float(_st.get("best_metric", float("inf")))
            _restore_rng(os.path.join(ckpt_path, "rng_state.pth"))
            saved = True
            print(f"[续训] 从 {ckpt_path} 的第 {start_it} 步接着训"
                  f"（最好验证损失 {best_val:.4f}）")
        elif has_weights(CONFIG["out_path"]):
            model.load_weights(CONFIG["out_path"])
            saved = True
            print("[续训] 只有最佳模型、没有 checkpoint；优化器将从零开始")
        else:
            print("[续训] 没有可用 checkpoint，从头训练")
    except Exception as e:
        print(f"[续训] 读取失败，改为从头训：{e}")

accum = max(1, int(CONFIG["grad_accum"]))          # 梯度累积步数至少 1
tokens_per_step = CONFIG["batch_size"] * CONFIG["block_size"] * accum   # 每步等效 token
print(f"[训练] 每步 {tokens_per_step} token"
      f"（batch {CONFIG['batch_size']} × {CONFIG['block_size']} 字 × 累积 {accum}）")
if CONFIG["lr_decay"] == "cosine":
    print(f"[训练] 学习率 {CONFIG['lr']:.1e} → {CONFIG['lr']*CONFIG['min_lr_frac']:.1e}"
          f"（余弦退火，预热 {CONFIG['warmup_iters']} 步）")

t0 = time.time()                                  # 训练计时起点
tokens_done = 0                                   # 已处理的 token 数，用于速度
last_it = start_it - 1                            # 记录实际跑到的最后一步（续训可能为空循环）
for it in range(start_it, CONFIG["max_iters"] + 1):
    last_it = it                                  # 记住当前步，训练结束时用
    lr = opt.param_groups[0]["lr"]                # 当前步的学习率（由 scheduler 维护）

    if it % CONFIG["eval_interval"] == 0:         # 到评估间隔了
        L = estimate_loss()                       # 在训练/验证集上估 loss
        elapsed = time.time() - t0                # 已用时间
        speed = tokens_done / elapsed if elapsed > 0 else 0   # token/s
        mark = ""                                 # 日志末尾的“已保存”标记
        if L["val"] < best_val:                   # 验证变好才存
            best_val = L["val"]                   # 更新历史最好
            save_ckpt(it, best_val)               # 中途存档，保留优化器
            saved = True
            mark = "  ← 已保存"
        print(f"  step {it:>6} | train {L['train']:.4f} | val {L['val']:.4f}"
              f" | 困惑度 {math.exp(L['val']):>7.1f} | {elapsed:>7.1f}s"
              f" | {speed:>6.0f} tok/s | lr {lr:.1e}{mark}")

    opt.zero_grad(set_to_none=True)               # 清空上一步的梯度（省内存）
    for _ in range(accum):                        # 梯度累积：跑 accum 个微批
        xb, yb = get_batch("train")               # 取一批训练数据
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            _, loss = model(xb, yb)               # 前向算 loss
        (loss / accum).backward()                 # 累积：每个微批的梯度只贡献 1/accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])  # 防梯度爆炸
    opt.step()                                    # 更新参数
    scheduler.step()                              # 推进一步学习率（状态会写入 scheduler.pt）
    tokens_done += tokens_per_step                # 累计 token 数

    if CONFIG["save_every"] and it > 0 and it % CONFIG["save_every"] == 0:
        save_ckpt(it, best_val, update_root=False, global_step=it + 1)
        print(f"  step {it:>6} | 定时存档")

print(f"[训练] 完成，用时 {time.time()-t0:.1f}s，最好验证损失 {best_val:.4f}")

# ==================================================================
# 5) 采样：让它自己往下写
# ==================================================================
model.eval()                                      # 采样阶段关掉 dropout
# 没配 prompt 时自动挑：对话模型用「用户：」，续写模型用第一个字符
prompt = CONFIG["prompt"] or ("用户：" if all(c in stoi for c in "用户：") else chars[0])
ids = encode(prompt) or [0]                        # 提示词编码；全不认识就退到 id 0
idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)  # (1, 长度)
out = model.generate(idx, max_new_tokens=300, temperature=0.8, top_k=40)  # 生成 300 字
print("\n===== 模型生成 =====")
if N_TOTAL < 5000:                                # 语料太小，基本是背诵
    print("（注意：语料不到 5000 token，所以下面这些它基本是【原样背出来】的）")
print()
print(decode(out[0].tolist()))                    # id 解码回文字
print("\n====================")

finished = last_it >= CONFIG["max_iters"]         # 是否正常跑到 max_iters（而不是中途退出）
if finished:
    if not saved:                                 # 一次都没存过就补一份最终存档
        save_ckpt(last_it, best_val, finished=True, global_step=last_it + 1)
        saved = True
    if not CONFIG.get("save_optimizer", True):    # 用户要求发布权重：去掉优化器
        # 发布权重：先释放当前训练状态，再读最后存档去掉优化器，避免两份同时占内存
        del model
        del opt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        strip_optimizer(CONFIG["out_path"])

if saved:
    print(f"模型已保存到 {CONFIG['out_path']}")
else:
    print("训练过程中验证损失一次都没有改善，没有保存模型。")
