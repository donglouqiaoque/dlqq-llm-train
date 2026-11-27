# -*- coding: utf-8 -*-
r"""OpenAI 兼容服务客户端

只使用标准库，支持用户配置任意 OpenAI 兼容服务：
base_url + API Key 环境变量名。API Key 本身绝不写入配置文件；Ollama 等本地
服务可以留空环境变量名，此时不发送 Authorization。

配置文件结构（由 webui.py 读写）：
{
  "apis": [
    {"id": "...", "name": "...", "base_url": "...", "api_key_env": ""}
  ]
}
"""
import json
import os
import re
import time
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlsplit

ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VERSION_RE = re.compile(r"/v\d+(?:\.\d+)?$")


class OpenAIError(RuntimeError):
    pass


def _norm_env(value, required=False):
    env = str(value or "").strip()
    if not env:
        if required:
            raise ValueError("API Key 环境变量名不能为空")
        return ""
    if not ENV_RE.match(env):
        raise ValueError("API Key 环境变量名只能包含字母、数字和下划线，且不能以数字开头")
    return env


def normalize_api(item, fallback_id=""):
    if not isinstance(item, dict):
        raise ValueError("API 配置必须是对象")
    name = str(item.get("name") or "").strip()
    if not name:
        raise ValueError("API 配置缺少名称")
    base_url = str(item.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("API 配置缺少 Base URL")
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Base URL 必须是 http:// 或 https:// 地址")
    env = _norm_env(item.get("api_key_env"))
    api_id = str(item.get("id") or fallback_id or "").strip()
    if not api_id:
        api_id = "api-" + str(int(time.time() * 1000))
    return {
        "id": api_id,
        "name": name,
        "base_url": base_url,
        "api_key_env": env,
    }


def load_apis(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    items = data.get("apis") if isinstance(data, dict) else data
    out = []
    for i, item in enumerate(items if isinstance(items, list) else []):
        try:
            out.append(normalize_api(item, f"api-{i + 1}"))
        except ValueError:
            continue
    return out


def save_apis(path, apis):
    clean = []
    for i, item in enumerate(apis if isinstance(apis, list) else []):
        clean.append(normalize_api(item, f"api-{i + 1}"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"apis": clean}, f, ensure_ascii=False, indent=2)
    return clean


def find_api(apis, api_id):
    for item in apis:
        if item.get("id") == api_id:
            return item
    return None


def key_status(api):
    env = str(api.get("api_key_env") or "")
    return bool(env and os.environ.get(env))


def api_key(api):
    env = _norm_env(api.get("api_key_env"))
    if not env:
        return ""
    value = os.environ.get(env)
    if not value:
        raise OpenAIError(f"环境变量 {env} 未设置，无法调用 API")
    return value


def _base(api):
    base = str(api.get("base_url") or "").strip().rstrip("/")
    parts = urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise OpenAIError("Base URL 不合法")
    return base


def chat_url(api):
    base = _base(api)
    if base.endswith("/chat/completions"):
        return base
    if VERSION_RE.search(base):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def models_url(api):
    base = _base(api)
    if base.endswith("/models"):
        return base
    if VERSION_RE.search(base):
        return base + "/models"
    return base + "/v1/models"


def embeddings_url(api):
    base = _base(api)
    if base.endswith("/embeddings"):
        return base
    if VERSION_RE.search(base):
        return base + "/embeddings"
    return base + "/v1/embeddings"


def _connection(url, timeout):
    parts = urlsplit(url)
    if parts.scheme == "https":
        conn = HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout)
    else:
        conn = HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return conn, path


def _headers(api, accept="application/json"):
    key = api_key(api)
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": accept,
        "User-Agent": "dlqq-llm-webui/1.0",
    }
    if key:
        headers["Authorization"] = "Bearer " + key
    return headers


DEFAULT_CHAT_TIMEOUT = 300
OLLAMA_CHAT_TIMEOUT = 600


def is_ollama_api(api):
    """Ollama 的 OpenAI 兼容接口在本地推理时可能长时间无新 token。"""
    name = str((api or {}).get("name") or "").lower()
    base = str((api or {}).get("base_url") or "").lower()
    return "ollama" in name or "11434" in base


def chat_timeout(api):
    return OLLAMA_CHAT_TIMEOUT if is_ollama_api(api) else DEFAULT_CHAT_TIMEOUT


def open_chat(api, payload, timeout=DEFAULT_CHAT_TIMEOUT):
    """返回 (connection, response)，调用方负责 close。"""
    url = chat_url(api)
    conn, path = _connection(url, timeout)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = _headers(api, "text/event-stream, application/json")
    headers["Content-Length"] = str(len(body))
    conn.request("POST", path, body=body, headers=headers)
    return conn, conn.getresponse()


def open_embeddings(api, payload, timeout=300):
    url = embeddings_url(api)
    conn, path = _connection(url, timeout)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = _headers(api, "application/json")
    headers["Content-Length"] = str(len(body))
    conn.request("POST", path, body=body, headers=headers)
    return conn, conn.getresponse()


def _looks_non_chat(model):
    name = str(model or "").lower()
    return any(x in name for x in ("embed", "rerank", "tts", "whisper", "image",
                                   "audio", "speech", "transcribe", "bge",
                                   "nomic", "minilm", "jina"))


DEFAULT_THINKING_LEVELS = ["off", "low", "medium", "high"]
DEFAULT_CONTEXT = 1000000
DEFAULT_MAX_TOKENS = 256000


def _context_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    text = str(value or "").strip().lower().replace(",", "")
    if not text:
        return None
    mult = 1
    if text.endswith("k"):
        mult, text = 1000, text[:-1]
    elif text.endswith("m"):
        mult, text = 1000000, text[:-1]
    try:
        n = int(float(text) * mult)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _context_length(item):
    """从 /models 的可选字段读取上下文；没有就按 1M 默认。"""
    if not isinstance(item, dict):
        return DEFAULT_CONTEXT, True
    candidates = [
        item.get("context_length"), item.get("context_window"),
        item.get("max_context_length"), item.get("max_model_len"),
        item.get("max_input_tokens"), item.get("context"),
    ]
    for box in (item.get("top_provider"), item.get("limits")):
        if isinstance(box, dict):
            candidates.extend([box.get("context_length"), box.get("context_window"),
                               box.get("max_context_length")])
    for value in candidates:
        n = _context_value(value)
        if n:
            return n, False
    return DEFAULT_CONTEXT, True


def _output_limit(item):
    if not isinstance(item, dict):
        return DEFAULT_MAX_TOKENS, True
    candidates = [
        item.get("max_output_tokens"), item.get("max_completion_tokens"),
        item.get("output_tokens"),
    ]
    box = item.get("top_provider")
    if isinstance(box, dict):
        candidates.extend([box.get("max_completion_tokens"),
                           box.get("max_output_tokens")])
    for value in candidates:
        n = _context_value(value)
        if n:
            return n, False
    return DEFAULT_MAX_TOKENS, True


def _level_text(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value).strip().lower()


def _level_list(values):
    out = []
    for value in values or []:
        text = _level_text(value)
        if text and text not in out:
            out.append(text)
    return out


def _thinking_info(item):
    """解析 /models 里可选的思考能力字段。

    标准 OpenAI /models 没有这些字段，返回 supported=None；调用方按
    “获取不到支持/等级 → 默认四档”处理。true/false 档位只用于旧接口明确
    返回布尔 values 的情况。
    """
    supported = None
    values = None
    default = None
    raw = item.get("thinking") if isinstance(item, dict) else None
    if isinstance(raw, bool):
        supported = raw
        if raw:
            # 旧接口只给 true/false，没有等级字段：只支持开/关。
            values = ["off", "on"]
            default = "off"
    elif isinstance(raw, dict):
        if isinstance(raw.get("supported"), bool):
            supported = raw["supported"]
        elif isinstance(raw.get("enabled"), bool):
            supported = raw["enabled"]
        raw_values = raw.get("values")
        if raw_values is None:
            raw_values = raw.get("levels")
        if isinstance(raw_values, list):
            parsed = _level_list(raw_values)
            if parsed:
                values = parsed
                if all(isinstance(x, bool) for x in raw_values):
                    values = ["off", "on"]
                supported = True
            elif supported is None:
                # 旧接口明确用空/布尔列表表示没有等级；支持状态未知时按不支持。
                supported = False
        if raw.get("default") is not None:
            default = _level_text(raw.get("default"))
    elif isinstance(raw, str) and raw.strip():
        values = [_level_text(raw)]
        supported = True

    reasoning = item.get("reasoning") if isinstance(item, dict) else None
    if isinstance(reasoning, bool):
        if supported is None:
            supported = reasoning
    elif isinstance(reasoning, dict):
        if isinstance(reasoning.get("supported"), bool) and supported is None:
            supported = reasoning["supported"]
        raw_values = reasoning.get("values")
        if raw_values is None:
            raw_values = reasoning.get("levels")
        if isinstance(raw_values, list) and values is None:
            parsed = _level_list(raw_values)
            if parsed:
                values = ["off", "on"] if all(isinstance(x, bool) for x in raw_values) else parsed
                supported = True
        if default is None:
            default = reasoning.get("default")
        if default is not None:
            default = _level_text(default)

    for key in ("supports_thinking", "supports_reasoning"):
        if isinstance(item.get(key), bool) and supported is None:
            supported = item[key]

    caps = item.get("capabilities")
    if isinstance(caps, list) and caps:
        lowered = {str(x).lower() for x in caps}
        if supported is None:
            if lowered & {"thinking", "reasoning", "reasoning_effort"}:
                supported = True
            else:
                # capabilities 是明确列表；没有思考相关项时视为明确不支持。
                supported = False

    params = item.get("supported_parameters")
    if isinstance(params, list) and params:
        lowered = {str(x).lower() for x in params}
        if supported is None:
            if lowered & {"thinking", "reasoning", "reasoning_effort", "include_reasoning"}:
                supported = True
            else:
                supported = False

    effort = item.get("reasoning_effort") if isinstance(item, dict) else None
    if isinstance(effort, list):
        parsed = _level_list(effort)
        if parsed:
            values = parsed
            supported = True
    elif isinstance(effort, dict):
        raw_values = effort.get("values")
        if isinstance(raw_values, list):
            parsed = _level_list(raw_values)
            if parsed:
                values = parsed
                supported = True

    if values:
        if default is None or _level_text(default) not in values:
            default = values[0]
    elif supported is True:
        values = None
        if default is None:
            default = "medium"
    elif supported is False:
        values = None
        default = None
    else:
        # 获取不到支持、也获取不到等级：默认四档，第一次使用建议关闭。
        supported = None
        values = None
        default = "off"

    return {"supported": supported, "values": values,
            "default": default,
            "mode": "boolean" if values == ["off", "on"] else
                    ("levels" if values else
                     ("default" if supported is True else
                      ("unsupported" if supported is False else "unknown")))}


def _looks_tool_capable(model):
    """兼容旧调用；新逻辑不再按模型名猜工具能力。"""
    return None


def _tool_use(item):
    """返回 native / none / unknown，不按模型名猜测。"""
    if not isinstance(item, dict):
        return "unknown"
    for key in ("supports_tools", "tools", "function_calling"):
        value = item.get(key)
        if isinstance(value, bool):
            return "native" if value else "none"
    caps = item.get("capabilities")
    if isinstance(caps, list) and caps:
        lowered = {str(x).lower() for x in caps}
        if lowered & {"tools", "tool_calling", "function_calling", "functions"}:
            return "native"
        return "unknown"
    params = item.get("supported_parameters")
    if isinstance(params, list) and params:
        lowered = {str(x).lower() for x in params}
        if lowered & {"tools", "tool_choice", "functions", "function_calling"}:
            return "native"
        return "unknown"
    return "unknown"


def _model_supports_tools(item):
    return _tool_use(item) == "native"


def _input_modalities(item):
    if not isinstance(item, dict):
        return ["text"]
    for key in ("input_modalities", "input", "modalities"):
        values = item.get(key)
        if isinstance(values, list):
            out = []
            for value in values:
                text = str(value or "").strip().lower()
                if text in ("text", "image") and text not in out:
                    out.append(text)
            if out:
                return out
    return ["text"]


def _reasoning_efforts(item, thinking=None):
    """把能力字段整理成 档位 -> API 参数值；没有声明时返回 None。"""
    if isinstance(item, dict):
        raw = item.get("reasoning_efforts")
        if isinstance(raw, dict):
            out = {}
            for level, value in raw.items():
                key = _level_text(level)
                if not key:
                    continue
                if value is None:
                    out[key] = None
                elif isinstance(value, bool):
                    out[key] = level if value else None
                elif str(value).strip():
                    out[key] = str(value).strip()
            if out:
                return out
        if isinstance(raw, list):
            values = _level_list(raw)
            if values:
                return {level: (None if level == "off" else level)
                        for level in values}
    if isinstance(thinking, dict):
        values = thinking.get("values")
        if isinstance(values, list) and values:
            return {level: (None if level == "off" else level)
                    for level in values}
    return None


def list_model_infos(api, timeout=20):
    url = models_url(api)
    conn, path = _connection(url, timeout)
    try:
        conn.request("GET", path, headers=_headers(api))
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        if resp.status < 200 or resp.status >= 300:
            raise OpenAIError(f"HTTP {resp.status}: {raw[:500]}")
        try:
            data = json.loads(raw)
        except ValueError:
            raise OpenAIError("模型列表不是合法 JSON")
        out = []
        for item in data.get("data") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("id") or "").strip()
            if not name:
                continue
            context, context_default = _context_length(item)
            output, output_default = _output_limit(item)
            thinking = _thinking_info(item)
            tool_use = _tool_use(item)
            out.append({"name": name,
                        "kind": "embedding" if _looks_non_chat(name) else "chat",
                        "context": context,
                        "context_default": context_default,
                        "context_source": "unknown" if context_default else "detected",
                        "output": output,
                        "output_default": output_default,
                        "output_source": "unknown" if output_default else "detected",
                        "tool_use": tool_use,
                        "supports_tools": tool_use == "native",
                        "input_modalities": _input_modalities(item),
                        "reasoning_efforts": _reasoning_efforts(item, thinking),
                        "thinking": thinking})
        return out
    finally:
        conn.close()


def list_models(api, timeout=20):
    return [x["name"] for x in list_model_infos(api, timeout)]
