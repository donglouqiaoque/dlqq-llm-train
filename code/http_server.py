# -*- coding: utf-8 -*-
r"""把训练好的模型起成一个 OpenAI 兼容的 HTTP 服务（桥接服务）
================================================================
本文件独立自足：只依赖 torch、transformers 和 Python 标准库（http.server / json / threading），
外加同目录的 common.py（读 properties 配置），不依赖 .ai_res（那是临时目录）。

供 DSH（或任何 OpenAI 兼容客户端）接入：DSH 侧只需要在界面里
「设置 → 模型 → 添加提供方 → 自定义提供方」，填上本服务的地址即可，不用改任何配置文件。

接口：
    GET  /health                 健康检查
    GET  /v1/models              模型清单（DSH 点「获取可用模型」时调它）
    POST /v1/chat/completions    对话（支持流式与非流式）
    POST /v1/embeddings          Ollama embedding 模型（float / base64）

行为约定 —— 桥接层只做两件事，内容判断一律交给模型：
    · system / developer 角色的消息**直接忽略**（这个模型从没用系统提示训过）；
    · 请求里的 tools / tool_choice / functions **一律忽略**（DSH 界面里没有关闭工具的
      开关，所以不能报错）；服务只返回普通文本，**永远不返回 tool_calls**；
      想让带 tools 的请求直接报错，就加 --reject-tools；
    · 多轮对话会拼成「用户：/助手：」的提示词，超出模型上下文时丢弃较早的轮次，
      只剩一轮还太长就按字符保留尾部；
    · 除此之外**不做任何过滤、打分或关键词匹配** —— 答成什么样是模型的事；
    · 只做基本对话，不支持图片。

用法（在工程根目录执行；一般由 WebUI 内部调用）：
    python code\http_server.py                     # 服务 properties\ 下所有能对上权重的配置
    python code\http_server.py --config dlqq_llm   # 只服务一个配置
    python code\http_server.py --port 49128 --device cpu
模型清单和对话参数都来自 properties\<配置>.properties，不再扫 model\ 目录。
"""
import os
import re
import sys
import json
import time
import base64
import struct
import uuid
import argparse
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

for _s in ("stdout", "stderr"):
    try:
        # 由 WebUI 拉起时 stdout 是管道，必须逐行刷新，否则日志要等缓冲区满了才出现
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace",
                                     line_buffering=True, write_through=True)
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ollama_client as ollama                                  # noqa: E402
import openai_client                                            # noqa: E402


# ==================================================================
# 桥接服务必须有 torch；没有就直接报错退出，绝不自动换解释器
# ==================================================================
def _require_torch():
    try:
        import torch  # noqa: F401
        return
    except Exception as e:
        print(f"[错误] 当前解释器没有 torch：{sys.executable}")
        print(f"       {type(e).__name__}: {e}")
        print("       请在 WebUI「环境」页给项目 Python 安装 torch / transformers，"
              "或重新运行 .runtime\\make_python_env.bat")
        sys.exit(1)


_require_torch()


def find_root(start):
    """往上找到含 code\train.py 的那一级，作为工程根目录"""
    cur = start
    for _ in range(6):
        if os.path.isfile(os.path.join(cur, "code", "train.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(start)


ROOT = find_root(HERE)
DEFAULT_MODEL = os.path.join(ROOT, "model", "chat_demo")
OPENAI_CONFIG_FILE = os.path.join(ROOT, ".cache", "openai_apis.json")
OPENAI_APIS = []
OPENAI_MODELS = []


def openai_model_id(api, model):
    return f"api:{api.get('id')}/{model}"


def openai_entry(model):
    raw = str(model or "")
    if not raw.startswith("api:") or "/" not in raw[4:]:
        return None
    api_id, upstream = raw[4:].split("/", 1)
    for entry in OPENAI_MODELS:
        if entry["api"].get("id") == api_id and entry["model"] == upstream:
            return entry
    return None

USER_TAG, ASSIST_TAG = "用户：", "助手："


# ==================================================================
# 模型：统一使用 code/model.py 的 transformers GPT2LMHeadModel，加载标准 HF 目录
# ==================================================================
def load_model(path, device=None):
    import torch
    from model import GPT

    if not os.path.isdir(path):
        raise RuntimeError(f"找不到模型目录：{path}")
    chars_path = os.path.join(path, "chars.json")
    try:
        with open(chars_path, encoding="utf-8") as f:
            chars = json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"模型目录缺少有效的 chars.json：{chars_path}") from e
    C = {}
    meta_path = os.path.join(path, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                C = (json.load(f).get("config") or {})
        except (OSError, ValueError):
            C = {}
    model = GPT.from_pretrained(path)
    hf = model.net.config
    C.setdefault("block_size", int(hf.n_positions))
    C.setdefault("n_layer", int(hf.n_layer))
    C.setdefault("n_head", int(hf.n_head))
    C.setdefault("n_embd", int(hf.n_embd))
    C.setdefault("chat_mode", True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    return model, chars, C, device


# ==================================================================
# 生成：把 messages 拼成提示词，逐字符采样
# ==================================================================
def read_model_props(path, config_name=""):
    r"""读 properties\<配置名>.properties 里的对话设置（改完不用重训就生效）。

    没给配置名时退回按模型文件名猜同名配置（兼容老调用）。
    """
    stem = (config_name or "").strip() or os.path.splitext(os.path.basename(path))[0]
    p = os.path.join(ROOT, "properties", stem + ".properties")
    out = {}
    if not os.path.isfile(p):
        return out
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError:
        return {}
    return out


class Engine:
    def __init__(self, path, device=None, config_name=""):
        self.model, chars, self.cfg, self.device = load_model(path, device)
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.block = self.cfg["block_size"]
        self.lock = threading.Lock()      # 同一时刻只跑一个生成，避免并发互相踩
        self.loaded = time.strftime("%Y-%m-%d %H:%M:%S")
        self.warned_tools = False         # 只在第一次遇到 tools 时提示一次
        self.config_name = config_name

        # 对话设置优先级：properties\<配置>.properties > 模型文件里存的 > 服务启动参数
        cfg = self.cfg
        props = read_model_props(path, config_name)
        self.chat_mode = bool(cfg.get("chat_mode", True))
        if "chat_mode" in props:
            self.chat_mode = props["chat_mode"].strip().lower() in ("1", "true", "yes", "y", "on")
        self.defaults = {}
        if "chat_temperature" in cfg:
            try:
                self.defaults["temperature"] = float(cfg["chat_temperature"])
            except (TypeError, ValueError):
                pass
        if "chat_top_k" in cfg:
            try:
                self.defaults["top_k"] = int(cfg["chat_top_k"])
            except (TypeError, ValueError):
                pass
        if "chat_max_new" in cfg:
            try:
                self.defaults["max_chars"] = int(cfg["chat_max_new"])
            except (TypeError, ValueError):
                pass
        for prop_key, name, cast in (("chat_temperature", "temperature", float),
                                     ("chat_top_k", "top_k", int),
                                     ("chat_max_new", "max_chars", int)):
            if prop_key in props:
                try:
                    self.defaults[name] = cast(props[prop_key])
                except (TypeError, ValueError):
                    pass

    # ---- 提示词 ----
    def build_prompt(self, messages):
        """messages → 提示词。
        只做两件事，别的都交给模型：
          1. 丢掉 system / developer 角色（这个模型从没用系统提示训过）；
          2. 历史太长时从最早的一轮开始丢，保证塞得进 block_size。
        不做任何「像不像人话」的内容判断 —— 那是模型的事，不是桥接层的事。"""
        convo = []                       # [(role, text)]
        for m in messages or []:
            role = (m.get("role") or "").lower()
            if role in ("system", "developer"):
                continue                 # 系统提示直接忽略
            if role not in ("user", "assistant"):
                continue                 # 工具结果之类的角色也忽略
            text = m.get("content")
            if isinstance(text, list):   # 兼容 {"type":"text","text":...} 形式
                text = "".join(p.get("text", "") for p in text
                               if isinstance(p, dict) and p.get("type") == "text")
            text = self.clean(text or "")
            if text:
                convo.append((role, text))

        # 最后一条必须是用户说的话，否则没法接着答
        while convo and convo[-1][0] != "user":
            convo.pop()
        if not convo:
            return "", 0

        # 纯续写模型（诗词）：把用户输入当一段文本的开头，不加任何标记
        if not self.chat_mode:
            text = convo[-1][1]
            return text[-self.block:], 1

        # 历史塞不下就从最早的一轮开始丢；只剩一轮还太长，就按字符保留尾部
        budget = max(self.block - 8, 8)
        while convo:
            prompt = "".join(f"{USER_TAG if r == 'user' else ASSIST_TAG}{t}\n"
                             for r, t in convo) + ASSIST_TAG
            if len(prompt) <= budget:
                return prompt, len(convo)
            if len(convo) == 1:
                return prompt[-budget:], 1
            convo.pop(0)
        return "", 0

    def clean(self, text):
        """去掉不在词表里的字符"""
        return "".join(c for c in str(text) if c in self.stoi)

    # ---- 采样 ----
    def generate_stream(self, prompt, max_chars, temperature=0.8, top_k=40,
                        stop_on_user=True):
        """逐字符产出，供 HTTP 真流式使用。"""
        import torch
        import torch.nn.functional as F

        ids = [self.stoi[c] for c in prompt if c in self.stoi]
        if not ids:
            return
        tags = (USER_TAG, ASSIST_TAG)
        # 回看窗口要比最长的标记多一格：判断「刚生成的字符会不会凑成标记」时，
        # 需要看到包括它在内的完整后缀，否则生成到「助手：」的冒号那一刻还看不到标记。
        lookback = max(len(t) for t in tags) + 1
        with self.lock:
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            tail = ""
            for _ in range(max(1, int(max_chars))):
                cond = idx[:, -self.block:]
                with torch.no_grad():
                    logits = self.model(cond)[:, -1, :] / max(float(temperature), 1e-6)
                if top_k:
                    v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                idx = torch.cat([idx, nxt], dim=1)
                ch = self.itos[int(nxt.item())]
                # 一旦凑出完整标记（含冒号）就停：模型开始自问自答了
                if stop_on_user and (tail + ch).endswith(tags):
                    break
                tail = (tail + ch)[-lookback:]
                yield ch

    def generate(self, prompt, max_chars, temperature=0.8, top_k=40, stop_on_user=True):
        """非流式路径：生成完整文本并做收尾清理。"""
        return _clean_reply("".join(self.generate_stream(
            prompt, max_chars, temperature, top_k, stop_on_user)))


def _clean_reply(text):
    """收尾清理：模型常在结尾自问自答地写上「助手」，把残留标记去掉。
    它在很多样本里只写「助手」不带冒号，所以停止条件抓不到，必须在这里补一刀。"""
    t = text.strip()
    # 带冒号的标记是模型「开始自问自答」的信号，从它出现的位置直接截断
    cut = [i for i in (t.find(ASSIST_TAG), t.find(USER_TAG)) if i > 0]
    if cut:
        t = t[:min(cut)]
    while True:
        before = t
        t = t.rstrip()
        for tag in (ASSIST_TAG, USER_TAG):          # 结尾的完整标记
            if t.endswith(tag):
                t = t[:-len(tag)]
        t = re.sub(r"(?:\r?\n)*\s*(?:助手|用户)[：:]?\s*$", "", t)   # 结尾的裸标记
        t = t.strip()
        if t == before:
            return t


# ==================================================================
# 配置清单 + 懒加载池
# ==================================================================
class ModelPool:
    """按需加载：/v1/models 只列配置名，不动显存；
    真正有对话请求进来才 from_pretrained 模型目录，并做 LRU 淘汰（默认只留一个在显存里）。"""

    def __init__(self, registry, device, max_loaded=1, extra_args=None):
        self.registry = dict(registry)
        self.device = device
        self.max_loaded = max(1, int(max_loaded))
        self.args = extra_args
        self.lock = threading.Lock()
        self.loaded = {}          # id -> Engine
        self.order = []           # LRU，最近用的在末尾
        self.hits = {}            # id -> 调用次数

    # ---- 清单 ----
    def ids(self):
        return sorted(self.registry)

    def path_of(self, mid):
        return self.registry.get(mid)

    def resolve(self, mid):
        """把请求里的 model 字段解析成 registry 里的 id。
        支持：完整 id、去扩展名的文件名、文件路径；大小写不敏感。"""
        if not mid:
            return None
        mid = str(mid).strip()
        if mid in self.registry:
            return mid
        base = os.path.splitext(os.path.basename(mid))[0]
        low = base.lower()

        def leaf(k):
            # id 可能是 "名字" 或 "目录/名字"，也可能带 @后缀
            return k.split("@")[0].split("/")[-1]

        for k in self.registry:
            if k == base or leaf(k) == base:
                return k
        for k in self.registry:
            if k.lower() == base.lower() or leaf(k).lower() == low:
                return k
        return None

    # ---- 加载 / 淘汰 ----
    def get(self, mid):
        with self.lock:
            eng = self.loaded.get(mid)
            if eng is not None:
                self.order.remove(mid)
                self.order.append(mid)
                self.hits[mid] = self.hits.get(mid, 0) + 1
                return eng

            path = self.registry.get(mid)
            if not path:
                raise KeyError(mid)

            # 先腾地方：超出上限就把最久没用的卸掉
            while len(self.loaded) >= self.max_loaded and self.order:
                old = self.order.pop(0)
                print(f"[卸载] {old}（显存/内存留给 {mid}）", flush=True)
                self.loaded.pop(old, None)
                _free_memory()

            print(f"[加载] {mid} ← {path}", flush=True)
            t0 = time.time()
            eng = Engine(path, self.device, mid)
            print(f"[加载] {mid} 完成：词表 {len(eng.chars)} | 上下文 {eng.block} 字 | "
                  f"{time.time() - t0:.1f}s", flush=True)
            self.loaded[mid] = eng
            self.order.append(mid)
            self.hits[mid] = self.hits.get(mid, 0) + 1
            return eng

    def status(self):
        return {
            "loaded": [{"id": m, "loaded_at": self.loaded[m].loaded,
                        "calls": self.hits.get(m, 0)} for m in self.order],
            "available": len(self.registry),
        }


def _free_memory():
    """卸模型后把缓存还给系统，避免下一个模型 OOM"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ==================================================================
# HTTP
# ==================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "dsh-local-gpt/1.0"
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload, ctype="application/json"):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, message, err_type="invalid_request_error", param=None):
        self._send(code, {"error": {"message": message, "type": err_type,
                                    "param": param, "code": None}})

    def log_message(self, fmt, *args):
        print(f"  [{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path in ("/health", "/"):
            st = POOL.status()
            self._send(200, {"status": "ok", "models": st["available"],
                             "loaded": [x["id"] for x in st["loaded"]],
                             "device": POOL.device,
                             "default": ARGS.default_model})
        elif path == "/v1/models":
            data = []
            for mid in POOL.ids():
                p = POOL.path_of(mid)
                try:
                    created = int(os.path.getmtime(p))
                except OSError:
                    created = int(time.time())
                data.append({"id": mid, "object": "model", "created": created,
                             "owned_by": "local"})
            for entry in OPENAI_MODELS:
                data.append({"id": openai_model_id(entry["api"], entry["model"]),
                             "object": "model", "created": int(time.time()),
                             "owned_by": "openai"})
            self._send(200, {"object": "list", "data": data})
        elif path.startswith("/v1/models/"):
            mid = urllib.parse.unquote(path[len("/v1/models/"):])
            entry = openai_entry(mid)
            if entry:
                self._send(200, {"id": mid, "object": "model",
                                 "created": int(time.time()), "owned_by": "openai"})
                return
            if ollama.is_remote(mid):
                if getattr(ARGS, "no_ollama", False):
                    self._err(404, f"没有这个模型：{mid}", "model_not_found", "model")
                    return
                tag = ollama.model_tag(mid)
                if tag not in ollama.list_models(getattr(ARGS, "ollama_url", None)):
                    self._err(404, f"没有这个模型：{mid}", "model_not_found", "model")
                    return
                self._send(200, {"id": mid, "object": "model",
                                 "created": int(time.time()), "owned_by": "ollama"})
                return
            real = POOL.resolve(mid)
            if not real:
                self._err(404, f"没有这个模型：{mid}", "model_not_found", "model")
                return
            p = POOL.path_of(real)
            self._send(200, {"id": real, "object": "model",
                             "created": int(os.path.getmtime(p)), "owned_by": "local"})
        else:
            self._err(404, f"unknown path: {self.path}", "not_found")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/embeddings"):
            self._err(404, f"unknown path: {self.path}（本服务支持 /v1/chat/completions 和 /v1/embeddings）",
                      "not_found")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:
            self._err(400, f"请求体不是合法 JSON：{e}")
            return

        if path == "/v1/embeddings":
            self.handle_embeddings(req)
            return

        if openai_entry(req.get("model")):
            self.handle_openai(req)
            return

        if ollama.is_remote(req.get("model")):
            self.handle_ollama(req)
            return

        self.handle_chat(req)

    def handle_embeddings(self, req):
        """OpenAI 兼容 embeddings：API 代理或 Ollama 原生 /api/embed。"""
        want = str(req.get("model") or "")
        entry = openai_entry(want)
        if entry:
            self.handle_openai_embeddings(req, entry)
            return
        if getattr(ARGS, "no_ollama", False):
            self._err(400, "Ollama 后端已关闭（--no-ollama）",
                      "unsupported_feature", "model")
            return
        if not ollama.is_remote(want):
            self._err(400, "本地模型不支持标准 embedding 接口；请使用 ollama/ 模型",
                      "unsupported_model", "model")
            return
        tag = ollama.model_tag(want)
        if not ollama.is_embedding_model(tag):
            self._err(400, f"这个模型不支持 embedding：{want}",
                      "unsupported_model", "model")
            return
        if req.get("encoding_format") == "base64":
            encoding_format = "base64"
        elif req.get("encoding_format") in (None, "", "float"):
            encoding_format = "float"
        else:
            self._err(400, "encoding_format 只能是 float 或 base64",
                      "invalid_request", "encoding_format")
            return

        raw_input = req.get("input")
        if isinstance(raw_input, str):
            inputs = [raw_input]
        elif isinstance(raw_input, list) and all(isinstance(x, str) for x in raw_input):
            inputs = raw_input
        else:
            self._err(400, "input 必须是字符串或字符串数组", "invalid_request", "input")
            return
        if not inputs or any(not x.strip() for x in inputs):
            self._err(400, "input 不能为空", "invalid_request", "input")
            return

        payload = {"model": tag, "input": inputs}
        dimensions = req.get("dimensions")
        if dimensions is not None:
            if not isinstance(dimensions, int) or dimensions <= 0:
                self._err(400, "dimensions 必须是正整数", "invalid_request", "dimensions")
                return
            payload["dimensions"] = dimensions

        t0 = time.time()
        try:
            conn, resp = ollama.open_embed(payload, getattr(ARGS, "ollama_url", None))
        except Exception as e:
            self._err(502, f"无法连接 Ollama：{type(e).__name__}: {e}", "upstream_error")
            return
        try:
            if resp.status < 200 or resp.status >= 300:
                body = resp.read().decode("utf-8", errors="replace")
                self._err(resp.status, f"Ollama embedding 失败：{body[:500]}",
                          "upstream_error")
                return
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            vectors = data.get("embeddings") if isinstance(data, dict) else None
            if not isinstance(vectors, list) or not vectors:
                self._err(502, "Ollama 没有返回 embedding", "upstream_error")
                return
            if len(vectors) != len(inputs):
                self._err(502, "Ollama 返回的向量数量与输入数量不一致", "upstream_error")
                return
            tokens = int(data.get("prompt_eval_count") or 0)
            dims = len(vectors[0]) if isinstance(vectors[0], list) else 0
            emit_event(t="embedding", model=want, count=len(inputs),
                       dimensions=dims, tokens=tokens,
                       ms=int((time.time() - t0) * 1000),
                       encoding=encoding_format,
                       input_preview=inputs[0][:160],
                       vector=vectors[0], vector_truncated=len(vectors) > 1)
            rows = []
            for i, vec in enumerate(vectors):
                if encoding_format == "base64":
                    raw = struct.pack("<%df" % len(vec), *vec)
                    embedding = base64.b64encode(raw).decode("ascii")
                else:
                    embedding = vec
                rows.append({"object": "embedding", "index": i,
                             "embedding": embedding})
            self._send(200, {
                "object": "list",
                "data": rows,
                "model": want,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            })
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except ValueError as e:
            self._err(502, f"解析 Ollama embedding 响应失败：{e}", "upstream_error")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def handle_openai_embeddings(self, req, entry):
        """代理 OpenAI 兼容 embeddings，并发出与服务页一致的向量事件。"""
        if entry.get("kind") != "embedding":
            self._err(400, f"这个模型不是 embedding 模型：{req.get('model')}",
                      "unsupported_model", "model")
            return
        api = entry["api"]
        want = str(req.get("model") or "")
        payload = dict(req)
        payload["model"] = entry["model"]
        raw_input = req.get("input")
        inputs = [raw_input] if isinstance(raw_input, str) else list(raw_input or [])
        t0 = time.time()
        try:
            conn, resp = openai_client.open_embeddings(api, payload, timeout=300)
        except Exception as e:
            self._err(502, f"无法连接 API：{type(e).__name__}: {e}", "upstream_error")
            return
        try:
            body = resp.read()
            if resp.status < 200 or resp.status >= 300:
                self._send(resp.status, {"error": {"message": body[:500].decode(
                    "utf-8", errors="replace")}})
                return
            try:
                data = json.loads(body.decode("utf-8", errors="replace"))
            except ValueError:
                self._err(502, "API 返回的 embedding 不是合法 JSON", "upstream_error")
                return
            rows = data.get("data") if isinstance(data, dict) else None
            if not isinstance(rows, list) or not rows:
                self._err(502, "API 没有返回 embedding", "upstream_error")
                return
            vectors = []
            for row in rows:
                emb = row.get("embedding") if isinstance(row, dict) else None
                if isinstance(emb, str):
                    try:
                        raw = base64.b64decode(emb)
                        emb = list(struct.unpack("<%df" % (len(raw) // 4), raw))
                    except Exception:
                        emb = []
                vectors.append(emb if isinstance(emb, list) else [])
            first = vectors[0] if vectors else []
            usage = data.get("usage") if isinstance(data, dict) else None
            tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
            emit_event(t="embedding", model=want, count=len(inputs),
                       dimensions=len(first), tokens=tokens,
                       ms=int((time.time() - t0) * 1000),
                       encoding=req.get("encoding_format") or "float",
                       input_preview=str(inputs[0] if inputs else "")[:160],
                       vector=first, vector_truncated=len(rows) > 1)
            self.send_response(resp.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def handle_openai(self, req):
        """把 OpenAI 兼容请求透明转发给配置的 OpenAI 兼容服务。"""
        want = str(req.get("model") or "")
        entry = openai_entry(want)
        if not entry:
            self._err(404, f"没有这个 API 模型：{want}", "model_not_found", "model")
            return
        if entry.get("kind") == "embedding":
            self._err(400, f"这个模型是 embedding 模型，不支持对话：{want}",
                      "unsupported_model", "model")
            return
        api = entry["api"]
        payload = dict(req)
        payload["model"] = entry["model"]
        level = payload.pop("thinking", None)
        if level is not None:
            if isinstance(level, bool):
                level = "medium" if level else "off"
            level = str(level).strip().lower()
            if level in ("", "off", "false", "0", "none", "no"):
                payload["reasoning_effort"] = "none"
            elif level in ("true", "on", "yes", "1"):
                payload["reasoning_effort"] = "medium"
            else:
                payload["reasoning_effort"] = level
        payload["stream"] = bool(payload.get("stream"))

        user = _last_user_text(req)
        t0 = time.time()
        cid = "chat-" + uuid.uuid4().hex[:20]
        emit_event(t="chat", phase="start", id=cid, model=want, user=user, turns=1)
        try:
            conn, resp = openai_client.open_chat(api, payload, timeout=300)
        except Exception as e:
            emit_event(t="chat", phase="error", id=cid, model=want,
                       error=f"无法连接 API：{type(e).__name__}: {e}")
            self._err(502, f"无法连接 API：{type(e).__name__}: {e}", "upstream_error")
            return

        ctype = (resp.getheader("Content-Type") or "application/json").lower()
        try:
            if resp.status < 200 or resp.status >= 300:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype or "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
                emit_event(t="chat", phase="error", id=cid, model=want,
                           error=f"API {resp.status}: {body[:300].decode('utf-8', errors='replace')}")
                return

            if payload["stream"]:
                self.send_response(resp.status)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                parts = []
                reasoning_parts = []
                for raw in resp:
                    self.wfile.write(raw)
                    self.wfile.flush()
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    piece = ollama.delta_text(obj)
                    reason = ollama.delta_reasoning(obj)
                    if piece:
                        parts.append(piece)
                        emit_event(t="chat", phase="delta", id=cid,
                                   model=want, text=piece)
                    if reason:
                        reasoning_parts.append(reason)
                        emit_event(t="chat", phase="reasoning", id=cid,
                                   model=want, text=reason)
                emit_event(t="chat", phase="done", id=cid, model=want, turns=1,
                           ms=int((time.time() - t0) * 1000), user=user,
                           reasoning="".join(reasoning_parts),
                           assistant="".join(parts), prompt_chars=0, stream=True)
                return

            body = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype or "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            try:
                obj = json.loads(body.decode("utf-8", errors="replace"))
            except ValueError:
                obj = {}
            text = ollama.delta_text(obj) if isinstance(obj, dict) else ""
            reasoning = ollama.delta_reasoning(obj) if isinstance(obj, dict) else ""
            emit_event(t="chat", phase="done", id=cid, model=want, turns=1,
                       ms=int((time.time() - t0) * 1000), user=user,
                       reasoning=reasoning, assistant=text,
                       prompt_chars=0, stream=False)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def handle_ollama(self, req):
        """把 OpenAI 兼容请求原样转给 Ollama。"""
        if getattr(ARGS, "no_ollama", False):
            self._err(400, "Ollama 后端已关闭（--no-ollama）", "unsupported_feature", "model")
            return

        want = str(req.get("model") or "")
        tag = ollama.model_tag(want)
        if not ollama.is_chat_model(tag):
            self._err(400, f"这个模型不支持对话：{want}",
                      "unsupported_model", "model")
            return
        payload = dict(req)
        payload["model"] = tag
        # WebUI 的“深度思考”开关：同时兼容原生 think 与 OpenAI 的 reasoning_effort。
        if "thinking" in payload:
            thinking = bool(payload.pop("thinking"))
            payload["think"] = thinking
            payload.setdefault("reasoning_effort", "medium" if thinking else "none")
        if payload.get("stream"):
            payload["stream"] = True
        else:
            payload["stream"] = False

        user = _last_user_text(req)
        t0 = time.time()
        cid = "chat-" + uuid.uuid4().hex[:20]
        # 先发 start，让服务页立刻显示用户问题；不能等 Ollama 返回首个 token，
        # 否则上游连接和首包延迟会把问题一起拖到最后才显示。
        emit_event(t="chat", phase="start", id=cid, model=want, user=user, turns=1)
        try:
            conn, resp = ollama.open_chat(payload, getattr(ARGS, "ollama_url", None))
        except Exception as e:
            emit_event(t="chat", phase="error", id=cid, model=want,
                       error=f"无法连接 Ollama：{type(e).__name__}: {e}")
            self._err(502, f"无法连接 Ollama：{type(e).__name__}: {e}", "upstream_error")
            return

        ctype = (resp.getheader("Content-Type") or "application/json").lower()
        try:
            if resp.status < 200 or resp.status >= 300:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype or "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
                return

            if payload["stream"]:
                self.send_response(resp.status)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                parts = []
                reasoning_parts = []
                for raw in resp:
                    self.wfile.write(raw)
                    self.wfile.flush()
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data and data != "[DONE]":
                            try:
                                obj = json.loads(data)
                            except ValueError:
                                obj = None
                            if isinstance(obj, dict):
                                piece = ollama.delta_text(obj)
                                reason = ollama.delta_reasoning(obj)
                                if piece:
                                    parts.append(piece)
                                    emit_event(t="chat", phase="delta", id=cid,
                                               model=want, text=piece)
                                if reason:
                                    reasoning_parts.append(reason)
                                    emit_event(t="chat", phase="reasoning", id=cid,
                                               model=want, text=reason)
                text = "".join(parts)
                emit_event(t="chat", phase="done", id=cid, model=want, turns=1,
                           ms=int((time.time() - t0) * 1000), user=user,
                           reasoning="".join(reasoning_parts), assistant=text,
                           prompt_chars=0, stream=True)
                return

            body = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype or "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            try:
                obj = json.loads(body.decode("utf-8", errors="replace"))
                text = ollama.delta_text(obj) if isinstance(obj, dict) else ""
                reasoning = ollama.delta_reasoning(obj) if isinstance(obj, dict) else ""
            except ValueError:
                text = ""
                reasoning = ""
            emit_event(t="chat", phase="done", id=cid, model=want, turns=1,
                       ms=int((time.time() - t0) * 1000), user=user,
                       reasoning=reasoning, assistant=text,
                       prompt_chars=0, stream=False)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # 客户端中途关闭连接，不需要再写错误响应
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def handle_chat(self, req):
        # DSH 是 agent 框架，请求里默认会带 tools；而界面里没有「关掉工具」的开关。
        # 所以这里**不能报错**，只能忽略 tools 字段，回一段普通文本、不返回 tool_calls。
        # 想恢复「直接拒绝」的行为，启动时加 --reject-tools。
        global WARNED_TOOLS
        if req.get("tools") or req.get("tool_choice") or req.get("functions"):
            if ARGS.reject_tools:
                self._err(400,
                          "本模型不支持工具调用（tools / tool_choice / functions）",
                          "unsupported_feature", "tools")
                return
            if not WARNED_TOOLS:
                WARNED_TOOLS = True
                n_tools = len(req.get("tools") or [])
                print(f"  [提示] 本请求带了 {n_tools} 个工具定义，已忽略 —— "
                      f"服务只会返回普通文本，不会返回 tool_calls", flush=True)

        # ---- 按请求里的 model 选模型；第一次用到才加载 ----
        want = req.get("model") or ARGS.default_model
        mid = POOL.resolve(want)
        if not mid:
            avail = ", ".join(POOL.ids()) or "(没有任何模型)"
            print(f"[404] 请求模型={want!r}，可用={avail}", flush=True)
            self._err(404, f"没有这个模型：{want}（可用：{avail}）",
                      "model_not_found", "model")
            return
        try:
            eng = POOL.get(mid)
        except Exception as e:
            self._err(500, f"加载模型 {mid} 失败：{e}", "model_load_error", "model")
            return

        model_id = mid
        prompt, turns = eng.build_prompt(req.get("messages"))
        if not prompt:
            self._err(400, "messages 里没有可用的用户消息", param="messages")
            return

        if ARGS.debug:
            self._dump_debug(req, prompt, turns)

        temperature = req.get("temperature")
        if temperature is None:
            temperature = eng.defaults.get("temperature", ARGS.temperature)
        else:
            temperature = float(temperature)
        top_k = req.get("top_k")
        if top_k is None:
            top_k = eng.defaults.get("top_k", ARGS.top_k)
        else:
            top_k = int(top_k)
        max_chars = req.get("max_tokens") or req.get("max_completion_tokens")
        if max_chars is None:
            max_chars = eng.defaults.get("max_chars", ARGS.max_chars)
        # 字符级模型：把「token 数」当成字符数，并夹在合理区间里
        max_chars = max(1, min(int(max_chars), ARGS.max_chars_cap))

        t0 = time.time()
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        stop_on_user = (eng.chat_mode and not ARGS.no_stop_on_user)
        user_text = _last_user_text(req)
        emit_event(t="chat", phase="start", id=cid, model=model_id,
                   user=user_text, turns=turns, prompt_chars=len(prompt))

        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model_id,
                       "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]})
            parts = []
            pending = ""
            gen = eng.generate_stream(prompt, max_chars, temperature, top_k,
                                      stop_on_user=stop_on_user)
            try:
                for ch in gen:
                    parts.append(ch)
                    pending += ch
                    if len(pending) >= 4:
                        emit_event(t="chat", phase="delta", id=cid, model=model_id,
                                   text=pending)
                        pending = ""
                    if not self._sse({"id": cid, "object": "chat.completion.chunk",
                                      "created": created, "model": model_id,
                                      "choices": [{"index": 0,
                                                   "delta": {"content": ch},
                                                   "finish_reason": None}]}):
                        break
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                try:
                    gen.close()
                except Exception:
                    pass
            if pending:
                emit_event(t="chat", phase="delta", id=cid, model=model_id, text=pending)
            text = _clean_reply("".join(parts))
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model_id,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass
            dt = time.time() - t0
            print(f"  → [{model_id}] 轮数 {turns} | 提示 {len(prompt)} 字 | "
                  f"生成 {len(text)} 字 | {dt:.2f}s | stream", flush=True)
            emit_event(t="chat", phase="done", id=cid, model=model_id, turns=turns,
                       ms=int(dt * 1000), user=user_text, assistant=text,
                       prompt_chars=len(prompt), stream=True)
            return

        text = eng.generate(prompt, max_chars, temperature, top_k,
                            stop_on_user=stop_on_user)
        dt = time.time() - t0
        print(f"  → [{model_id}] 轮数 {turns} | 提示 {len(prompt)} 字 | 生成 {len(text)} 字 | {dt:.2f}s", flush=True)
        emit_event(t="chat", phase="done", id=cid, model=model_id, turns=turns,
                   ms=int(dt * 1000), user=user_text, assistant=text,
                   prompt_chars=len(prompt), stream=False)

        self._send(200, {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(text),
                "total_tokens": len(prompt) + len(text),
            },
        })

    def _sse(self, obj):
        try:
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _dump_debug(self, req, prompt, turns):
        """--debug 时把客户端发来的消息与我拼出的提示词打出来，便于排查"""
        print("  ┌─ 调试 ─────────────────────────────────────────", flush=True)
        for m in req.get("messages") or []:
            role = m.get("role")
            text = m.get("content")
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            text = (text or "").replace("\n", "⏎")
            print(f"  │ {role:<10} {len(text):>5} 字 | {text[:160]}", flush=True)
        print(f"  │ 工具: {len(req.get('tools') or [])} 个 | stream={bool(req.get('stream'))} "
              f"| max_tokens={req.get('max_tokens') or req.get('max_completion_tokens')}", flush=True)
        print(f"  │ 拼出提示词（{turns} 轮 / {len(prompt)} 字）：{prompt.replace(chr(10), '⏎')}", flush=True)
        print("  └────────────────────────────────────────────────", flush=True)


ARGS = None
POOL = None
WARNED_TOOLS = False
EMIT_EVENTS = False


def _last_user_text(req):
    """取客户端消息里最后一条 user 的纯文本，供事件回显用。"""
    for m in reversed(req.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


def emit_event(**kw):
    """给 WebUI 的事件行：stdout 打一行 `@@EV@@ {...}`，由读线程按前缀识别。"""
    if not EMIT_EVENTS:
        return
    try:
        print("@@EV@@ " + json.dumps(kw, ensure_ascii=False), flush=True)
    except (OSError, ValueError):
        pass


if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common                                                     # noqa: E402


def main():
    global ARGS, POOL, EMIT_EVENTS, OPENAI_APIS, OPENAI_MODELS
    ap = argparse.ArgumentParser(description="按配置文件把模型起成 OpenAI 兼容服务")
    ap.add_argument("--config", action="append", default=[],
                    help="只服务这些配置（properties 里的名字，可写多次）；默认全部")
    ap.add_argument("--openai-api", action="append", default=[],
                    help="只代理这些 OpenAI 兼容 API 配置 id（可写多次）；默认全部")
    ap.add_argument("--default-model", default=None,
                    help="请求没带 model 时用哪个配置；默认清单里第一个")
    ap.add_argument("--max-loaded", type=int, default=1,
                    help="同时留在显存里的模型数，默认 1（切模型就卸载上一个）")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认只绑本机")
    ap.add_argument("--port", type=int, default=49128, help="监听端口，默认 49128")
    ap.add_argument("--ollama-url", default=None,
                    help="Ollama 根地址；默认取环境变量 OLLAMA_URL，再退回 http://127.0.0.1:11434")
    ap.add_argument("--no-ollama", action="store_true", help="关闭 Ollama 代理，只服务本地模型")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="推理设备，默认自动（有 CUDA 就用 CUDA）")
    ap.add_argument("--temperature", type=float, default=0.8, help="默认温度")
    ap.add_argument("--top-k", type=int, default=40, help="默认 top_k")
    ap.add_argument("--max-chars", type=int, default=180, help="默认单次生成多少字符")
    ap.add_argument("--max-chars-cap", type=int, default=600,
                    help="单次生成字符数的硬上限（客户端要再多也不给）")
    ap.add_argument("--no-stop-on-user", action="store_true",
                    help="不在生成出「用户：」时停止")
    ap.add_argument("--reject-tools", action="store_true",
                    help="带 tools 的请求直接返回 400（默认是忽略 tools 并正常回答）")
    ap.add_argument("--debug", action="store_true",
                    help="把客户端发来的消息与拼出的提示词打印出来（排查用）")
    ap.add_argument("--events", action="store_true",
                    help="每完成一次对话就往 stdout 打一行 @@EV@@ JSON（供 WebUI 显示会话）")
    ap.add_argument("--parent-pid", type=int, default=0,
                    help="父进程 pid；父进程退出时本进程自动退出（供 WebUI 调用）")
    a = ap.parse_args()

    # ---- 收集配置：每个配置一条，模型路径和对话参数都由 properties 决定 ----
    cfgs = common.chat_configs(ROOT)
    if a.config:
        want = {c.strip().lower() for c in a.config if c.strip()}
        cfgs = [c for c in cfgs if c["name"].lower() in want]
    reg = {c["name"]: c["path"] for c in cfgs}

    OPENAI_APIS = openai_client.load_apis(OPENAI_CONFIG_FILE)
    if a.openai_api:
        want_api = {str(x).strip() for x in a.openai_api if str(x).strip()}
        OPENAI_APIS = [x for x in OPENAI_APIS if x.get("id") in want_api]
    OPENAI_MODELS = []
    for api in OPENAI_APIS:
        try:
            infos = openai_client.list_model_infos(api)
        except Exception as e:
            print(f"[警告] API {api.get('name')} 模型检测失败：{type(e).__name__}: {e}")
            continue
        for info in infos:
            OPENAI_MODELS.append({"api": api, "model": info["name"],
                                  "kind": info.get("kind") or "chat"})
    if not reg and not OPENAI_MODELS:
        print("[错误] 没有任何配置能对应到现成的权重，也没有可用的 OpenAI API 模型。")
        print("       检查 properties\\<名字>.properties 里的 model=，")
        print("       它应指向 model\\<名字>\\（HF 标准目录）；也可以先训练一个。")
        sys.exit(1)

    ids = sorted(reg)
    if ids:
        default = a.default_model or ids[0]
        if default not in reg:
            hit = [k for k in ids if k.lower() == default.lower()]
            if not hit:
                print(f"[错误] --default-model 不是可用配置：{default}（可用：{', '.join(ids)}）")
                sys.exit(1)
            default = hit[0]
    else:
        first = OPENAI_MODELS[0]
        default = a.default_model or openai_model_id(first["api"], first["model"])
    a.default_model = default

    if a.device is None:
        try:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            a.device = "cpu"

    ARGS = a
    EMIT_EVENTS = bool(getattr(a, "events", False))
    if getattr(a, "parent_pid", 0):
        common.start_parent_watchdog(a.parent_pid)
    POOL = ModelPool(reg, a.device, a.max_loaded, a)

    # 端口独占：allow_reuse_address 在 Windows 上允许多个进程绑同一端口，
    # 后起的实例看似启动成功，连接却被旧进程抢走/复位。关掉它，第二个实例
    # 会直接拿到 WinError 10048，而不是悄悄共享。
    class _ExclusiveServer(ThreadingHTTPServer):
        allow_reuse_address = False

        def server_bind(self):
            # Windows 独占绑定：别的进程（包括带 SO_REUSEADDR 的）不能再绑同一端口。
            if os.name == "nt":
                try:
                    self.socket.setsockopt(socket.SOL_SOCKET,
                                           socket.SO_EXCLUSIVEADDRUSE, 1)
                except (AttributeError, OSError):
                    pass
            super().server_bind()

    try:
        srv = _ExclusiveServer((a.host, a.port), Handler)
    except OSError as e:
        if getattr(e, "winerror", None) == 10048:
            print(f"[错误] 端口 {a.port} 已被占用：可能还有上一个服务进程没退出。")
            print("       先结束占用它的进程，或换一个端口（--port <其它端口>）再启动。")
        else:
            print(f"[错误] 无法监听 {a.host}:{a.port}：{e}")
        sys.exit(1)
    srv.daemon_threads = True

    print("=" * 68)
    print("  桥接服务：本地模型（OpenAI 兼容，多模型 + 懒加载）")
    print("=" * 68)
    print(f"[清单] 共 {len(ids)} 个本地配置（模型路径与对话参数都取自 properties，懒加载）")
    for mid in ids:
        p = reg[mid]
        try:
            mb = os.path.getsize(p) / 1e6
            mt = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        except OSError:
            mb, mt = 0, "?"
        mark = "  ← 默认" if mid == default else ""
        print(f"       {mid:<24} {mb:>8.1f} MB  {mt}{mark}")
    if OPENAI_MODELS:
        print(f"[API] 共 {len(OPENAI_MODELS)} 个 OpenAI 兼容模型：")
        for entry in OPENAI_MODELS:
            api = entry["api"]
            mid = openai_model_id(api, entry["model"])
            mark = "  ← 默认" if mid == default else ""
            print(f"       {mid:<40} {api.get('name', '')}{mark}")
    print(f"[设备] {a.device} | 同时最多留 {a.max_loaded} 个模型在显存里")
    if not a.no_ollama:
        names = ollama.list_models(a.ollama_url)
        if names:
            print(f"[Ollama] {ollama.base_url(a.ollama_url)} | "
                  f"{len(names)} 个模型：{', '.join(ollama.model_id(n) for n in names)}")
        else:
            print(f"[Ollama] {ollama.base_url(a.ollama_url)} 未发现可用模型（服务仍可用本地模型）")
    print(f"[工具] " + ("带 tools 的请求直接返回 400（--reject-tools）"
                       if a.reject_tools else
                       "忽略请求里的 tools，只回普通文本，不返回 tool_calls"))
    print()
    print(f"  接口     http://{a.host}:{a.port}/v1")
    print(f"  模型清单 http://{a.host}:{a.port}/v1/models")
    print(f"  健康检查 http://{a.host}:{a.port}/health")
    print("  停止服务：在这个窗口按 Ctrl+C")
    print("=" * 68)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止] 服务已退出")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
