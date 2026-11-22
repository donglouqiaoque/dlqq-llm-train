# -*- coding: utf-8 -*-
r"""Ollama HTTP 适配层
====================

只使用标准库，负责三件事：
  1. 发现 Ollama 上已安装的模型（/api/tags）；
  2. 把 OpenAI 兼容对话请求转发到 /v1/chat/completions；
  3. 逐行读取 Ollama 返回的 SSE。

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

_cache = {"base": "", "at": 0.0, "models": []}


def base_url(value=None):
    """Ollama 根地址；兼容传入带 /v1 的 OpenAI base_url。"""
    v = (value or os.environ.get("OLLAMA_URL") or DEFAULT_BASE).strip().rstrip("/")
    if v.endswith("/v1"):
        v = v[:-3].rstrip("/")
    return v or DEFAULT_BASE


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
