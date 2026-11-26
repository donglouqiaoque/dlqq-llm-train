# -*- coding: utf-8 -*-
r"""Ollama HTTP 适配层
====================

只使用标准库，负责四件事：
  1. 发现 Ollama 上已安装的模型（/api/tags）；
  2. 查询模型能力（/api/show），区分对话和 embedding；
  3. 把 OpenAI 兼容对话请求转发到 /v1/chat/completions；
  4. 把 embedding 请求转发到 /api/embed，并逐行读取对话 SSE。

Ollama 只作为外部推理后端，不加载 GGUF，也不引入 LangChain。
模型 id 统一加 `ollama/` 前缀，和本地 HF 模型区分。
"""
import json
import os
import time
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PREFIX = "ollama/"
DEFAULT_BASE = "http://127.0.0.1:11434"
_base_override = None

_cache = {"base": "", "at": 0.0, "models": []}
_caps = {"base": "", "data": {}}
_think = {"base": "", "data": {}}
CAP_TTL = 60.0


def _looks_like_embedding(name):
    """仅在 /api/show 不可用时兜底；能力字段优先。"""
    s = str(name or "").lower()
    return any(x in s for x in ("embedding", "embed", "bge", "e5", "gte",
                                "nomic", "jina"))


def model_capabilities(name, base=None, timeout=2.5, use_cache=True):
    """查询 Ollama 模型能力，如 completion / embedding / tools / vision。

    旧版 Ollama 或只开放 OpenAI 兼容接口的代理可能没有 /api/show，返回 []。
    """
    base = base_url(base)
    tag = model_tag(name)
    now = time.time()
    if _caps["base"] != base:
        _caps.update(base=base, data={})
    hit = _caps["data"].get(tag)
    if use_cache and hit and now - hit[0] < CAP_TTL:
        return list(hit[1])

    caps = []
    try:
        body = json.dumps({"model": tag}, ensure_ascii=False).encode("utf-8")
        req = Request(base + "/api/show", data=body, method="POST",
                      headers={"Content-Type": "application/json; charset=utf-8",
                               "Accept": "application/json"})
        with urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        if isinstance(data, dict) and isinstance(data.get("capabilities"), list):
            caps = [str(x) for x in data.get("capabilities") if x]
    except Exception:
        caps = []

    _caps["data"][tag] = (now, caps)
    return list(caps)


def is_chat_model(name, base=None):
    caps = set(model_capabilities(name, base))
    if caps:
        return bool(caps & {"completion", "chat"})
    return not _looks_like_embedding(name)


def _thinking_value(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value).strip().lower()


def model_thinking(name, base=None, timeout=2.5, use_cache=True):
    """返回模型的思考控制元数据。

    Ollama 的 /api/show 可能返回 thinking.values/default；部分模型只有
    capabilities=thinking 而没有等级元数据。后者优先按 Ollama 支持的低/中/高
    回退，避免把“没有元数据”误判成只能开关。
    """
    base = base_url(base)
    tag = model_tag(name)
    now = time.time()
    if _think["base"] != base:
        _think.update(base=base, data={})
    hit = _think["data"].get(tag)
    if use_cache and hit and now - hit[0] < CAP_TTL:
        return dict(hit[1]) if hit[1] else None

    meta = None
    caps = []
    try:
        body = json.dumps({"model": tag}, ensure_ascii=False).encode("utf-8")
        req = Request(base + "/api/show", data=body, method="POST",
                      headers={"Content-Type": "application/json; charset=utf-8",
                               "Accept": "application/json"})
        with urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        raw = data.get("thinking") if isinstance(data, dict) else None
        caps = [str(x) for x in (data.get("capabilities") or [])]
        if isinstance(raw, dict) and isinstance(raw.get("values"), list):
            values = []
            for value in raw["values"]:
                v = _thinking_value(value)
                if v and v not in values:
                    values.append(v)
            default = _thinking_value(raw.get("default"))
            if values:
                if all(isinstance(x, bool) for x in raw["values"]):
                    values = ["off", "on"]
                if default not in values:
                    default = values[0]
                meta = {"supported": True, "values": values, "default": default,
                        "mode": "boolean" if values == ["off", "on"] else "levels"}
    except Exception:
        caps = []
    if meta is None and "thinking" in caps:
        meta = {"supported": True, "values": None, "default": "medium",
                "mode": "default"}
    _think["data"][tag] = (now, meta)
    return dict(meta) if meta else None


def is_embedding_model(name, base=None):
    caps = set(model_capabilities(name, base))
    if caps:
        return "embedding" in caps
    return _looks_like_embedding(name)


def model_context(name, base=None, timeout=2.5):
    """读取 Ollama 模型声明的上下文长度；没有返回 None。"""
    base = base_url(base)
    tag = model_tag(name)
    try:
        body = json.dumps({"model": tag}, ensure_ascii=False).encode("utf-8")
        req = Request(base + "/api/show", data=body, method="POST",
                      headers={"Content-Type": "application/json; charset=utf-8",
                               "Accept": "application/json"})
        with urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    info = data.get("model_info") if isinstance(data, dict) else None
    if not isinstance(info, dict):
        return None
    values = []
    for key, value in info.items():
        if str(key).lower().endswith(".context_length"):
            try:
                n = int(value)
            except (TypeError, ValueError):
                continue
            if n > 0:
                values.append(n)
    return max(values) if values else None


def base_url(value=None):
    """Ollama 根地址；兼容传入带 /v1 的 OpenAI base_url。"""
    raw = value
    if not raw:
        raw = _base_override or os.environ.get("OLLAMA_URL") or DEFAULT_BASE
    v = str(raw).strip().rstrip("/")
    if v.endswith("/v1"):
        v = v[:-3].rstrip("/")
    return v or DEFAULT_BASE


def set_base_url(value):
    """设置本次运行的 Ollama 根地址；空值恢复环境变量/默认值。"""
    global _base_override
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        _base_override = None
        return None
    if raw.endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Ollama URL 必须是 http:// 或 https:// 地址")
    _base_override = raw
    return _base_override


def model_id(name):
    return PREFIX + str(name).strip()


def model_tag(mid):
    s = str(mid or "").strip()
    return s[len(PREFIX):] if s.startswith(PREFIX) else s


def is_remote(mid):
    return str(mid or "").strip().startswith(PREFIX)


def list_models(base=None, timeout=2.5, use_cache=True):
    """返回 Ollama 上可用的模型名列表；不可用时返回 []。"""
    base = base_url(base)
    now = time.time()
    if use_cache and _cache["base"] == base and now - _cache["at"] < 5:
        return list(_cache["models"])

    names = []
    try:
        req = Request(base + "/api/tags", headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        names = [str(m.get("name") or "").strip()
                 for m in (data.get("models") or []) if isinstance(m, dict)]
    except Exception:
        # 某些代理只开放 OpenAI 兼容接口，没有 /api/tags
        try:
            req = Request(base + "/v1/models", headers={"Accept": "application/json"})
            with urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            names = [str(m.get("id") or "").strip()
                     for m in (data.get("data") or []) if isinstance(m, dict)]
        except Exception:
            names = []

    names = sorted({n for n in names if n})
    _cache.update(base=base, at=now, models=names)
    return list(names)


def detect_environment(base=None, timeout=2.5):
    """检测 Ollama 服务、版本和模型数量；不加载模型。"""
    base = base_url(base)
    out = {"ok": False, "url": base, "version": "", "models": [],
           "chat_models": [], "embedding_models": [], "error": ""}
    error = ""
    try:
        req = Request(base + "/api/version", headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        if isinstance(data, dict):
            out["version"] = str(data.get("version") or "")
    except Exception as e:
        error = f"{type(e).__name__}: 无法连接 Ollama"

    names = list_models(base, timeout=timeout, use_cache=False)
    out["models"] = names
    out["chat_models"] = [n for n in names if is_chat_model(n, base)]
    out["embedding_models"] = [n for n in names if is_embedding_model(n, base)]
    out["ok"] = bool(out["version"] or names)
    if not out["ok"]:
        out["error"] = error or "无法连接 Ollama"
    return out


def open_chat(payload, base=None, timeout=300):
    """向 Ollama 发一个 /v1/chat/completions 请求。

    返回 (connection, response)，调用方负责 close。
    非 2xx 也会返回 response，由调用方读取错误体。
    """
    base = base_url(base)
    parts = urlsplit(base)
    if parts.scheme == "https":
        conn = HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout)
    else:
        conn = HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    path = (parts.path.rstrip("/") if parts.path else "") + "/v1/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    conn.request("POST", path, body=body, headers={
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "text/event-stream, application/json",
        "Content-Length": str(len(body)),
    })
    return conn, conn.getresponse()


def open_embed(payload, base=None, timeout=300):
    """向 Ollama 发一个原生 /api/embed 请求。

    返回 (connection, response)，调用方负责 close。
    """
    base = base_url(base)
    parts = urlsplit(base)
    if parts.scheme == "https":
        conn = HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout)
    else:
        conn = HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    path = (parts.path.rstrip("/") if parts.path else "") + "/api/embed"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    conn.request("POST", path, body=body, headers={
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "Content-Length": str(len(body)),
    })
    return conn, conn.getresponse()


def iter_sse(response):
    """逐条解析 Ollama 的 OpenAI SSE，返回 dict。"""
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def delta_text(obj):
    """从 chat.completion.chunk / chat.completion 中取增量文本。"""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    ch = choices[0] or {}
    delta = ch.get("delta") or {}
    if isinstance(delta, dict) and delta.get("content"):
        return str(delta["content"])
    msg = ch.get("message") or {}
    if isinstance(msg, dict) and msg.get("content"):
        return str(msg["content"])
    return ""


def delta_reasoning(obj):
    """取 Ollama 的思考内容，兼容原生与 OpenAI 兼容字段。"""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    ch = choices[0] or {}
    delta = ch.get("delta") or {}
    msg = ch.get("message") or {}
    for part in (delta, msg):
        if not isinstance(part, dict):
            continue
        for key in ("reasoning_content", "thinking", "reasoning"):
            value = part.get(key)
            if value:
                return str(value)
    return ""
