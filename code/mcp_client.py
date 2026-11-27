# -*- coding: utf-8 -*-
r"""MCP 客户端（STDIO / Streamable HTTP）
========================================
只依赖标准库，负责：
  1. 连接用户配置的 MCP 服务器；
  2. initialize / tools/list / tools/call；
  3. 把 MCP inputSchema 转成 OpenAI/Ollama 的工具定义。

STDIO 使用一行一个 JSON；HTTP 使用 JSON-RPC POST，并保留 Mcp-Session-Id。
"""
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class MCPError(RuntimeError):
    pass


_process_hook = None


def set_process_hook(fn):
    """WebUI 用它把 MCP 子进程加入自己的 Job Object。"""
    global _process_hook
    _process_hook = fn


def _split_args(value):
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parts = shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        parts = text.split()
    return [p.strip('"').strip("'") for p in parts if p]


class MCPClient:
    def __init__(self, config, timeout=20):
        self.config = config or {}
        self.timeout = timeout
        self._next_id = 0
        self._pending = {}
        self._lock = threading.Lock()
        self._request_lock = threading.RLock()
        self._notifications = deque(maxlen=100)
        self._closed = False
        self.session_id = ""

    def alive(self):
        return not self._closed

    def _new_id(self):
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _request(self, method, params=None, timeout=None):
        raise NotImplementedError

    def _notify(self, method, params=None):
        raise NotImplementedError

    def initialize(self):
        result = self._request("initialize", {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "dlqq-llm-webui", "version": "1.0"},
        })
        self._notify("notifications/initialized")
        return result

    def list_tools(self):
        result = self._request("tools/list", {})
        return result.get("tools") or []

    def call_tool(self, name, arguments=None, meta=None):
        params = {
            "name": str(name),
            "arguments": arguments or {},
        }
        if meta:
            params["_meta"] = meta
        return self._request("tools/call", params,
                             timeout=max(self.timeout, 120))

    def ping(self, timeout=5):
        """轻量健康检查；HTTP 客户端没有进程可 poll 时使用。"""
        self._request("tools/list", {}, timeout=timeout)
        return True

    def close(self):
        self._closed = True

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class StdioMCPClient(MCPClient):
    def __init__(self, config, timeout=20):
        super().__init__(config, timeout)
        self.proc = None
        self._stderr = deque(maxlen=80)
        self._reader = None
        self._stderr_reader = None

    def alive(self):
        return bool(not self._closed and self.proc and self.proc.poll() is None)

    def start(self):
        command = str(self.config.get("command") or "").strip()
        if not command:
            raise MCPError("STDIO 缺少 command")
        args = _split_args(self.config.get("args"))
        cwd = str(self.config.get("cwd") or "").strip() or None
        kwargs = {
            "cwd": cwd,
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen([command] + args, **kwargs)
        except OSError as e:
            raise MCPError(f"无法启动 MCP 进程：{e}")
        if _process_hook:
            try:
                _process_hook(self.proc)
            except Exception:
                pass
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def _read_stderr(self):
        try:
            for line in self.proc.stderr:
                self._stderr.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    self._stderr.append("非 JSON 输出：" + line[:200])
                    continue
                if not isinstance(obj, dict):
                    continue
                mid = obj.get("id")
                if mid is not None and ("result" in obj or "error" in obj):
                    with self._lock:
                        hit = self._pending.pop(mid, None)
                    if hit:
                        hit[1] = obj
                        hit[0].set()
                else:
                    self._notifications.append(obj)
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                pending = list(self._pending.values())
                self._pending.clear()
            for ev, box in pending:
                box[1] = {"error": {"message": "MCP 进程已退出"}}
                ev.set()

    def _send(self, obj):
        if not self.proc or self.proc.poll() is not None:
            tail = "\n".join(self._stderr)[-500:]
            raise MCPError("MCP 进程未运行" + (("：" + tail) if tail else ""))
        raw = json.dumps(obj, ensure_ascii=False)
        try:
            self.proc.stdin.write(raw + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPError(f"写入 MCP 进程失败：{e}")

    def _request(self, method, params=None, timeout=None):
        with self._request_lock:
            mid = self._new_id()
            ev = threading.Event()
            box = [ev, None]
            with self._lock:
                self._pending[mid] = box
            payload = {"jsonrpc": "2.0", "id": mid, "method": method}
            if params is not None:
                payload["params"] = params
            self._send(payload)
            if not ev.wait(timeout or self.timeout):
                with self._lock:
                    self._pending.pop(mid, None)
                raise MCPError(f"MCP 请求超时：{method}")
            obj = box[1] or {}
            if "error" in obj:
                err = obj.get("error") or {}
                raise MCPError(str(err.get("message") or err))
            return obj.get("result") or {}

    def _notify(self, method, params=None):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def close(self):
        super().close()
        if self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
            except OSError:
                pass


class HttpMCPClient(MCPClient):
    def __init__(self, config, timeout=20):
        super().__init__(config, timeout)
        self.url = str(config.get("url") or "").strip()
        if not self.url:
            raise MCPError("HTTP 缺少 url")

    def _post(self, payload, timeout=None):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=timeout or self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                ctype = (resp.headers.get("Content-Type") or "").lower()
                raw = resp.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise MCPError(f"HTTP {e.code}：{body[:500]}")
        except (URLError, OSError) as e:
            raise MCPError(f"无法连接 MCP HTTP：{e}")
        if "text/event-stream" in ctype:
            return self._parse_sse(raw)
        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except ValueError as e:
            raise MCPError(f"MCP HTTP 返回不是 JSON：{e}")

    @staticmethod
    def _parse_sse(raw):
        events = []
        for block in raw.replace("\r\n", "\n").split("\n\n"):
            data = []
            for line in block.splitlines():
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
            if not data:
                continue
            text = "\n".join(data)
            if text == "[DONE]":
                continue
            try:
                obj = json.loads(text)
            except ValueError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        return {"_sse_events": events}

    def _unwrap(self, obj, mid):
        if not isinstance(obj, dict):
            raise MCPError("MCP HTTP 响应为空")
        if "_sse_events" in obj:
            for item in obj["_sse_events"]:
                if item.get("id") == mid:
                    return item
            raise MCPError("MCP SSE 中没有对应响应")
        return obj

    def _request(self, method, params=None, timeout=None):
        with self._request_lock:
            mid = self._new_id()
            payload = {"jsonrpc": "2.0", "id": mid, "method": method}
            if params is not None:
                payload["params"] = params
            obj = self._unwrap(self._post(payload, timeout), mid)
            if "error" in obj:
                err = obj.get("error") or {}
                raise MCPError(str(err.get("message") or err))
            return obj.get("result") or {}

    def _notify(self, method, params=None):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._post(payload)


def client_for(config, timeout=20):
    transport = str((config or {}).get("transport") or "stdio").lower()
    if transport == "http":
        client = HttpMCPClient(config, timeout=timeout)
    else:
        client = StdioMCPClient(config, timeout=timeout)
        client.start()
    try:
        client.initialize()
    except Exception:
        client.close()
        raise
    return client


def openai_tools(tools):
    """MCP tools -> OpenAI function tools。"""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description") or "")[:2000],
                "parameters": schema,
            },
        })
    return out


def tool_result_text(result):
    """把 MCP tools/call 结果压成一段完整文本，不做截断。"""
    if not isinstance(result, dict):
        return str(result)
    parts = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            continue
        if item.get("text"):
            parts.append(str(item["text"]))
        elif item.get("type") == "resource" and item.get("resource"):
            parts.append(json.dumps(item["resource"], ensure_ascii=False))
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts) if parts else json.dumps(result, ensure_ascii=False)
