# -*- coding: utf-8 -*-
r"""本地 WebUI 服务
================================================================
一个进程同时提供：静态页面（web\）、JSON API、SSE 事件流，并管理所有子进程。

启动（一般由 .runtime\start.bat 调用）：
    .runtime\python\python.exe code\webui.py              # 默认 49134，端口被占用会提示输入
    .runtime\python\python.exe code\webui.py --port 8080  # 指定端口
    .runtime\python\python.exe code\webui.py --no-browser # 不开浏览器

生命周期：
    · 关掉 start.bat 的 cmd 窗口 = 退出，所有子进程一起结束；
    · 关浏览器标签不影响任何进程，重新访问控制台打印的地址即可。

安全：只绑 127.0.0.1；API 需要启动时生成的一次性 token（地址里带 ?t=）。
"""
import argparse
import atexit
import codecs
import concurrent.futures
import ctypes
import datetime
import hashlib
import json
import os
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common                                                     # noqa: E402
import ollama_client as ollama                                    # noqa: E402
import mcp_client                                                 # noqa: E402
import openai_client                                              # noqa: E402

common.setup_stdio()
ROOT = common.find_root()
WEB_DIR = os.path.join(ROOT, "web")
LOGO_PATH = os.path.join(ROOT, ".runtime", "logo.ico")
CACHE_DIR = os.path.join(ROOT, ".cache")
DEFAULT_WORKSPACE_DIR = os.path.join(ROOT, ".runtime", "default_workspace")
LOG_DIR = os.path.join(CACHE_DIR, "train-logs")
OLLAMA_URL_FILE = os.path.join(CACHE_DIR, "ollama_url.txt")
MCP_CONFIG_FILE = os.path.join(CACHE_DIR, "mcp_servers.json")
MCP_STATUS_FILE = os.path.join(CACHE_DIR, "mcp_status.json")
MCP_LOCK = threading.Lock()
MCP_STATUS_LOCK = threading.RLock()
MCP_CLIENTS = {}
MCP_CLIENTS_LOCK = threading.RLock()
MCP_RECONNECT_BASE = 0.5
MCP_RECONNECT_MAX = 30.0
MCP_RECONNECT_ATTEMPTS = 10
MCP_HEALTH_INTERVAL = 30.0
MCP_MONITOR_STOP = threading.Event()
MCP_MONITOR_THREAD = None
MCP_RECONNECTING = set()
OPENAI_CONFIG_FILE = os.path.join(CACHE_DIR, "openai_apis.json")
OPENAI_STATUS_FILE = os.path.join(CACHE_DIR, "openai_status.json")
OPENAI_MODEL_OVERRIDES_FILE = os.path.join(CACHE_DIR, "openai_model_overrides.json")
WORKSPACE_FILE = os.path.join(CACHE_DIR, "workspaces.json")
LEGACY_WORKSPACE_FILE = os.path.join(CACHE_DIR, "workspace.json")
SESSIONS_DIR = CACHE_DIR
SESSION_LOCK = threading.RLock()
STATE = {"token": "", "started": time.time(), "train": None, "server": None}


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    common.hide_dir(CACHE_DIR)


def _canonical_path(path):
    return os.path.abspath(os.path.expanduser(str(path or ROOT)))


def workspace_id(path=None):
    return hashlib.sha256(
        os.path.normcase(_canonical_path(path or ROOT)).encode("utf-8")
    ).hexdigest()[:16]


def legacy_workspace_id():
    """旧版 workspace_id 没有 normcase；用于兼容旧会话文件。"""
    return hashlib.sha256(os.path.abspath(ROOT).encode("utf-8")).hexdigest()[:16]


def ensure_session_dir():
    ensure_cache_dir()
    return workspace_id(), CACHE_DIR


def _new_workspace(path, title=""):
    canonical = _canonical_path(path)
    now = time.time()
    return {
        "id": workspace_id(canonical),
        "path": canonical,
        "title": str(title or os.path.basename(canonical) or canonical)[:80],
        "created_at": now,
        "updated_at": now,
    }


def _default_workspace_dir():
    os.makedirs(DEFAULT_WORKSPACE_DIR, exist_ok=True)
    return DEFAULT_WORKSPACE_DIR


def _default_workspace_record():
    return _new_workspace(_default_workspace_dir(), "默认工作区")


def _normalize_workspace_registry(data):
    """确保注册表始终有默认工作区；旧工程根工作区迁移到 .runtime/default_workspace。"""
    changed = False
    if not isinstance(data, dict):
        data = {}
        changed = True
    if not isinstance(data.get("workspaces"), list):
        data["workspaces"] = []
        changed = True
    workspaces = []
    for w in data["workspaces"]:
        if not isinstance(w, dict) or not w.get("id") or not w.get("path"):
            changed = True
            continue
        workspaces.append(w)
    data["workspaces"] = workspaces
    default_dir = os.path.normcase(_default_workspace_dir())
    root_dir = os.path.normcase(_canonical_path(ROOT))
    default_ws = None
    root_ws = None
    for w in workspaces:
        path = os.path.normcase(_canonical_path(w.get("path")))
        if path == default_dir:
            default_ws = w
        elif path == root_dir:
            root_ws = w
    if default_ws is None:
        if root_ws is not None:
            root_ws["path"] = _canonical_path(DEFAULT_WORKSPACE_DIR)
            root_ws["title"] = "默认工作区"
            root_ws["updated_at"] = time.time()
            default_ws = root_ws
        else:
            default_ws = _default_workspace_record()
            data["workspaces"].append(default_ws)
        changed = True
    if not isinstance(data.get("last_agent_session"), dict):
        data["last_agent_session"] = {}
        changed = True
    if data.get("last_agent_workspace") != default_ws.get("id"):
        if data.get("last_agent_workspace") not in {
                w.get("id") for w in data["workspaces"]}:
            data["last_agent_workspace"] = default_ws.get("id")
            changed = True
    return data, changed


def load_workspace_registry():
    ensure_cache_dir()
    data = None
    legacy = False
    for path in (WORKSPACE_FILE, LEGACY_WORKSPACE_FILE):
        try:
            with open(path, "r", encoding="utf-8") as f:
                candidate = json.load(f)
            if isinstance(candidate, dict):
                data = candidate
                legacy = path == LEGACY_WORKSPACE_FILE
                if isinstance(candidate.get("workspaces"), list):
                    break
                if candidate.get("path") and candidate.get("id"):
                    break
        except (OSError, ValueError):
            continue
    if isinstance(data, dict) and isinstance(data.get("workspaces"), list):
        workspaces = [w for w in data["workspaces"]
                      if isinstance(w, dict) and w.get("id") and w.get("path")]
        data["workspaces"] = workspaces
    elif isinstance(data, dict) and data.get("path") and data.get("id"):
        # 迁移旧的单工作区 workspace.json
        w = _new_workspace(data["path"], data.get("title") or "")
        w["id"] = data["id"]
        last = data.get("last_session")
        last = last if isinstance(last, dict) else {}
        data = {"workspaces": [w],
                "last_chat_session": str(last.get("chat") or ""),
                "last_agent_session": ({w["id"]: str(last["agent"])}
                                       if last.get("agent") else {}),
                "last_agent_workspace": w["id"]}
    else:
        data = {"workspaces": [], "last_chat_session": "",
                "last_agent_session": {}, "last_agent_workspace": ""}
    data, changed = _normalize_workspace_registry(data)
    if changed or legacy:
        save_workspace_registry(data)
    return data


def save_workspace_registry(data):
    ensure_cache_dir()
    with SESSION_LOCK:
        with open(WORKSPACE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def default_workspace(registry=None):
    reg = registry or load_workspace_registry()
    default_dir = os.path.normcase(_canonical_path(DEFAULT_WORKSPACE_DIR))
    for w in reg.get("workspaces") or []:
        if os.path.normcase(_canonical_path(w.get("path"))) == default_dir:
            return w
    workspaces = reg.get("workspaces") or []
    return workspaces[0] if workspaces else _default_workspace_record()


def find_workspace(wid):
    for w in load_workspace_registry().get("workspaces") or []:
        if w.get("id") == wid:
            return w
    return None


def add_workspace(path, title=""):
    canonical = _canonical_path(path)
    if not os.path.isdir(canonical):
        raise ValueError("工作区路径不存在或不是目录")
    if ".dlqq" in canonical.lower().split(os.sep):
        raise ValueError("不能把 .dlqq 目录作为工作区")
    reg = load_workspace_registry()
    for w in reg.get("workspaces") or []:
        if os.path.normcase(_canonical_path(w.get("path"))) == os.path.normcase(canonical):
            return w
    w = _new_workspace(canonical, title)
    w["id"] = workspace_id(canonical)
    reg.setdefault("workspaces", []).append(w)
    save_workspace_registry(reg)
    return w


def pick_workspace_folder():
    """在 WebUI 所在桌面弹出原生目录选择框，返回绝对路径。"""
    if os.name != "nt":
        raise ValueError("当前系统不支持原生目录选择")
    script = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
        "$d.Description='选择 Agent 工作区目录';"
        "$d.ShowNewFolderButton=$true;"
        "if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
        "{[Console]::Out.Write($d.SelectedPath)}"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
        capture_output=True, timeout=300, creationflags=flags)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(err or "目录选择窗口打开失败")
    path = proc.stdout.decode("utf-8", errors="replace").strip()
    if not path:
        return None
    if not os.path.isdir(path):
        raise ValueError("选择的目录不存在")
    return _canonical_path(path)


def rename_workspace(wid, title):
    reg = load_workspace_registry()
    for w in reg.get("workspaces") or []:
        if w.get("id") == wid:
            w["title"] = str(title or "").strip()[:80] or os.path.basename(w.get("path") or "")
            w["updated_at"] = time.time()
            save_workspace_registry(reg)
            return w
    return None


def delete_workspace(wid):
    reg = load_workspace_registry()
    default = default_workspace(reg)
    if not default or wid == default.get("id"):
        return False
    before = len(reg.get("workspaces") or [])
    reg["workspaces"] = [w for w in reg.get("workspaces") or []
                         if w.get("id") != wid]
    if len(reg["workspaces"]) == before:
        return False
    reg.get("last_agent_session", {}).pop(wid, None)
    if reg.get("last_agent_workspace") == wid:
        reg["last_agent_workspace"] = default_workspace(reg).get("id")
    save_workspace_registry(reg)
    return True


def load_workspace():
    """兼容旧接口：返回默认工作区及最近会话。"""
    reg = load_workspace_registry()
    w = dict(default_workspace(reg))
    w["last_session"] = {
        "chat": reg.get("last_chat_session") or "",
        "agent": (reg.get("last_agent_session") or {}).get(w.get("id")) or "",
    }
    return w


def save_workspace(data):
    """兼容旧调用；现代码统一写 workspaces 注册表。"""
    if isinstance(data, dict) and isinstance(data.get("workspaces"), list):
        save_workspace_registry(data)
        return
    if isinstance(data, dict) and data.get("path"):
        w = _new_workspace(data["path"], data.get("title") or "")
        w["id"] = data.get("id") or w["id"]
        save_workspace_registry({"workspaces": [w],
                                 "last_chat_session": "",
                                 "last_agent_session": {},
                                 "last_agent_workspace": w["id"]})


def _session_id(value):
    text = str(value or "").strip()
    if re.fullmatch(r"[0-9a-f]{32}", text):
        return text
    return ""


def _session_path(sid):
    ws, d = ensure_session_dir()
    clean = _session_id(sid)
    if not clean:
        return None
    return os.path.join(d, "session-" + clean + ".json")


def session_title(history, body_title=""):
    title = str(body_title or "").strip()
    if title:
        return title[:80]
    for msg in history or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(str(x.get("text") or "") for x in content
                                   if isinstance(x, dict))
            text = str(content or "").strip().replace("\n", " ")
            if text:
                return text[:40]
    return "新会话"


def list_sessions(kind=None, workspace=None):
    ws, d = ensure_session_dir()
    reg = load_workspace_registry()
    default_id = (default_workspace(reg) or {}).get("id")
    legacy_id = legacy_workspace_id()
    out = []
    try:
        names = [n for n in os.listdir(d)
                 if n.startswith("session-") and n.endswith(".json")]
    except OSError:
        return out
    for name in names:
        try:
            with open(os.path.join(d, name), "r", encoding="utf-8") as f:
                data = json.load(f)
            item_kind = data.get("kind") or "chat"
            if kind and item_kind != kind:
                continue
            if item_kind == "agent" and workspace:
                wid = data.get("workspace_id")
                legacy_ok = (wid == legacy_id and workspace == default_id)
                if wid != workspace and not legacy_ok:
                    continue
            out.append({"id": data.get("id"), "kind": item_kind,
                        "workspace_id": data.get("workspace_id") or "",
                        "title": data.get("title") or "新会话",
                        "updated_at": data.get("updated_at") or 0,
                        "model": data.get("model") or ""})
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda x: x.get("updated_at") or 0, reverse=True)


def load_session(sid):
    path = _session_path(sid)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _history_preview(content, limit=80):
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif block.get("type") == "image_url":
                    parts.append("[图片]")
        text = " ".join(parts)
    else:
        text = str(content or "")
    text = " ".join(text.split())
    return text[:limit]


def session_history_page(data, before=None, limit=24):
    """只返回最近一段历史，并附带用于跳转的全部用户轮次大纲。"""
    history = data.get("history") if isinstance(data, dict) else None
    if not isinstance(history, list):
        history = []
    total = len(history)
    try:
        limit = max(1, min(100, int(limit or 24)))
    except (TypeError, ValueError):
        limit = 24
    try:
        end = total if before is None else max(0, min(total, int(before)))
    except (TypeError, ValueError):
        end = total
    start = max(0, end - limit)
    outline = []
    for index, msg in enumerate(history):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        outline.append({
            "index": index,
            "ts": str(msg.get("ts") or ""),
            "prompt": _history_preview(msg.get("content")),
        })
    return {
        "history": history[start:end],
        "history_total": total,
        "history_start": start,
        "history_end": end,
        "history_has_more": start > 0,
        "history_outline": outline,
    }


def _strip_ui_messages(messages):
    """去掉仅用于界面分页/时间显示的字段，避免把它们发给模型。"""
    clean = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        item = {"role": msg.get("role"), "content": msg.get("content")}
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in msg:
                item[key] = msg[key]
        clean.append(item)
    return clean


def merge_session_history(sid, messages, history_start=0):
    """把懒加载返回的最近历史片段和会话文件中的旧历史拼成模型输入。"""
    clean = _strip_ui_messages(messages)
    try:
        start = max(0, int(history_start or 0))
    except (TypeError, ValueError):
        start = 0
    if not sid or start <= 0:
        return clean
    data = load_session(sid)
    old = data.get("history") if isinstance(data, dict) else None
    if not isinstance(old, list) or len(old) < start:
        return clean
    return _strip_ui_messages(old[:start]) + clean


def save_session(body):
    sid = _session_id(body.get("id"))
    history = body.get("history")
    if not isinstance(history, list):
        return None
    kind = "agent" if body.get("kind") == "agent" else "chat"
    old = load_session(sid) if sid else None
    now = time.time()
    # 懒加载会话只回传最近一段；保存时必须把未加载的旧历史和这段拼回去。
    try:
        history_start = int(body.get("history_start") or 0)
    except (TypeError, ValueError):
        history_start = 0
    old_history = old.get("history") if isinstance(old, dict) else None
    if (history_start > 0 and isinstance(old_history, list)
            and len(old_history) >= history_start):
        history = old_history[:history_start] + list(history)
    if not sid:
        sid = uuid.uuid4().hex
    title = str(body.get("title") or "").strip()
    if not title and old and old.get("title"):
        # 自动保存不能把用户手动改过的会话名覆盖回首条消息。
        title = old["title"]
    title = session_title(history, title)
    if kind == "agent":
        reg = load_workspace_registry()
        wid = str(body.get("workspace_id") or "").strip()
        if not wid and old:
            wid = str(old.get("workspace_id") or "")
        if not find_workspace(wid):
            wid = default_workspace(reg).get("id")
    else:
        wid = ""
    data = {
        "id": sid,
        "workspace_id": wid,
        "kind": kind,
        "title": title,
        "created_at": (old or {}).get("created_at") or now,
        "updated_at": now,
        "history": history,
        "config": str(body.get("config") or ""),
        "model": str(body.get("model") or ""),
        "server": str(body.get("server") or ""),
        "tools": body.get("tools") if isinstance(body.get("tools"), list) else [],
        "thinking": str(body.get("thinking") or "off"),
        "model_history": ((old or {}).get("model_history")
                          if kind == "agent" else None),
        "last_usage": (old or {}).get("last_usage"),
        "usage_totals": (old or {}).get("usage_totals"),
        "last_time_context": (old or {}).get("last_time_context"),
    }
    path = _session_path(sid)
    with SESSION_LOCK:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        reg = load_workspace_registry()
        if kind == "chat":
            reg["last_chat_session"] = sid
        else:
            last = reg.get("last_agent_session")
            if not isinstance(last, dict):
                last = {}
            last[wid] = sid
            reg["last_agent_session"] = last
            reg["last_agent_workspace"] = wid
        save_workspace_registry(reg)
    return data


def delete_session(sid):
    path = _session_path(sid)
    if not path or not os.path.isfile(path):
        return False
    with SESSION_LOCK:
        try:
            os.remove(path)
        except OSError:
            # 某些隐藏缓存目录不允许删除文件；清空后 list/load 会忽略它。
            try:
                with open(path, "w", encoding="utf-8"):
                    pass
            except OSError:
                return False
    return True


def save_session_model_history(sid, messages, updates=None):
    """保存 Agent 的完整模型消息，保证下一轮仍是上一轮请求的前缀。"""
    clean = _strip_ui_messages([m for m in messages or [] if isinstance(m, dict)])
    path = _session_path(sid)
    if not path or not clean:
        return
    with SESSION_LOCK:
        data = load_session(sid) or {"id": sid, "kind": "agent"}
        data["model_history"] = clean
        data["updated_at"] = time.time()
        if isinstance(updates, dict):
            data.update(updates)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass


def rename_session(sid, title):
    path = _session_path(sid)
    if not path or not os.path.isfile(path):
        return None
    with SESSION_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        data["title"] = str(title or "").strip()[:80] or "新会话"
        data["updated_at"] = time.time()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            return None
    return data


def move_session(sid, workspace_id):
    """把 Agent 会话移动到另一个工作区；下一次 tools/call 自动使用新路径。"""
    data = load_session(sid)
    if not data or (data.get("kind") or "chat") != "agent":
        return None
    target = find_workspace(workspace_id)
    if not target:
        return None
    old_wid = str(data.get("workspace_id") or "")
    if old_wid == target["id"]:
        return data
    data["workspace_id"] = target["id"]
    data["updated_at"] = time.time()
    path = _session_path(sid)
    with SESSION_LOCK:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            return None
        reg = load_workspace_registry()
        last = reg.get("last_agent_session")
        if not isinstance(last, dict):
            last = {}
        if last.get(old_wid) == sid:
            last.pop(old_wid, None)
        last[target["id"]] = sid
        reg["last_agent_session"] = last
        reg["last_agent_workspace"] = target["id"]
        save_workspace_registry(reg)
    return data




def load_ollama_url():
    try:
        with open(OLLAMA_URL_FILE, "r", encoding="utf-8") as f:
            value = f.read().strip()
        if value:
            ollama.set_base_url(value)
    except (OSError, ValueError):
        pass


def save_ollama_url(value):
    try:
        ensure_cache_dir()
        with open(OLLAMA_URL_FILE, "w", encoding="utf-8") as f:
            f.write(value or "")
    except OSError:
        pass


def _normalize_mcp_server(item, fallback_id=""):
    if not isinstance(item, dict):
        raise ValueError("MCP 配置必须是对象")
    name = str(item.get("name") or "").strip()
    if not name:
        raise ValueError("MCP 配置缺少名称")
    transport = str(item.get("transport") or "stdio").lower()
    if transport not in ("stdio", "http"):
        raise ValueError("MCP 传输只能是 stdio 或 http")
    server_id = str(item.get("id") or fallback_id or "").strip()
    if not server_id:
        server_id = "mcp-" + str(int(time.time() * 1000))
    out = {"id": server_id, "name": name, "transport": transport}
    if transport == "stdio":
        command = str(item.get("command") or "").strip()
        if not command:
            raise ValueError("STDIO 配置缺少 command")
        out.update(command=command,
                   args=str(item.get("args") or "").strip(),
                   cwd=str(item.get("cwd") or "").strip())
    else:
        url = str(item.get("url") or "").strip()
        if not url:
            raise ValueError("HTTP 配置缺少 url")
        out["url"] = url
    return out


def load_mcp_servers():
    try:
        with open(MCP_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    servers = data.get("servers") if isinstance(data, dict) else data
    out = []
    for i, item in enumerate(servers if isinstance(servers, list) else []):
        try:
            out.append(_normalize_mcp_server(item, f"mcp-{i + 1}"))
        except ValueError:
            continue
    return out


def save_mcp_servers(servers):
    clean = []
    for i, item in enumerate(servers if isinstance(servers, list) else []):
        clean.append(_normalize_mcp_server(item, f"mcp-{i + 1}"))
    with MCP_LOCK:
        ensure_cache_dir()
        with open(MCP_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"servers": clean}, f, ensure_ascii=False, indent=2)
    mcp_reconcile()
    return clean


def _mcp_signature(server):
    return json.dumps(server, ensure_ascii=False, sort_keys=True)


def stop_mcp_client(server_id):
    with MCP_CLIENTS_LOCK:
        entry = MCP_CLIENTS.pop(server_id, None)
    if entry:
        try:
            entry["client"].close()
        except Exception:
            pass


def start_mcp_client(server, force=False):
    """返回常驻 MCP 客户端；配置变化或进程退出时重启。"""
    sid = server["id"]
    sig = _mcp_signature(server)
    with MCP_CLIENTS_LOCK:
        entry = MCP_CLIENTS.get(sid)
        if (not force and entry and entry["signature"] == sig and
                entry["client"].alive()):
            entry["state"] = "connected"
            return entry["client"]
        if entry:
            try:
                entry["client"].close()
            except Exception:
                pass
            MCP_CLIENTS.pop(sid, None)
        client = mcp_client.client_for(server, timeout=30)
        MCP_CLIENTS[sid] = {"client": client, "signature": sig,
                            "config": server, "state": "connected",
                            "attempts": 0, "next_retry": 0.0,
                            "last_health": time.time(), "last_error": ""}
        return client


def ensure_mcp_client(server):
    """Agent 取客户端时，已标记断连的连接要强制重建。"""
    with MCP_CLIENTS_LOCK:
        entry = MCP_CLIENTS.get(server["id"])
        bad = bool(entry and entry.get("state") in ("reconnecting", "failed"))
    return start_mcp_client(server, force=bad)


def mcp_entry(sid):
    with MCP_CLIENTS_LOCK:
        return MCP_CLIENTS.get(sid)


def mcp_update_status(sid, **changes):
    with MCP_STATUS_LOCK:
        status = load_mcp_status()
        entry = status.get(sid)
        if not isinstance(entry, dict):
            entry = {}
        entry.update(changes)
        status[sid] = entry
        save_mcp_status(status)


def mcp_mark_unhealthy(sid, error=""):
    with MCP_CLIENTS_LOCK:
        entry = MCP_CLIENTS.get(sid)
        if not entry:
            return
        if entry.get("state") == "failed":
            return
        entry["state"] = "reconnecting"
        entry["last_error"] = str(error or "")
        attempts = int(entry.get("attempts") or 0)
        entry["attempts"] = attempts
        delay = min(MCP_RECONNECT_MAX,
                    MCP_RECONNECT_BASE * (2 ** min(attempts, 6)))
        entry["next_retry"] = time.time() + delay
    mcp_update_status(sid, state="reconnecting", attempts=entry["attempts"],
                      last_error=entry["last_error"])


def mcp_reconnect_worker(sid, server, claimed=False):
    with MCP_CLIENTS_LOCK:
        if sid in MCP_RECONNECTING and not claimed:
            return
        MCP_RECONNECTING.add(sid)
    try:
        client = start_mcp_client(server, force=True)
        with MCP_CLIENTS_LOCK:
            entry = MCP_CLIENTS.get(sid)
            if entry:
                entry["state"] = "connected"
                entry["attempts"] = 0
                entry["next_retry"] = 0.0
                entry["last_error"] = ""
                entry["last_health"] = time.time()
        mcp_update_status(sid, state="connected", attempts=0,
                          last_error="")
        try:
            tools = client.list_tools()
            mcp_update_status(sid, ok=True, tools=[
                {"name": t.get("name"), "description": t.get("description") or ""}
                for t in tools if isinstance(t, dict)
            ], error="")
        except Exception:
            pass
    except Exception as e:
        with MCP_CLIENTS_LOCK:
            entry = MCP_CLIENTS.get(sid)
            if not entry:
                entry = {"attempts": 0}
                MCP_CLIENTS[sid] = entry
            attempts = int(entry.get("attempts") or 0) + 1
            entry["attempts"] = attempts
            entry["last_error"] = f"{type(e).__name__}: {e}"
            if attempts >= MCP_RECONNECT_ATTEMPTS:
                entry["state"] = "failed"
                entry["next_retry"] = 0.0
            else:
                entry["state"] = "reconnecting"
                entry["next_retry"] = time.time() + min(
                    MCP_RECONNECT_MAX, MCP_RECONNECT_BASE * (2 ** min(attempts, 6)))
        mcp_update_status(sid, state=entry.get("state") or "reconnecting",
                          attempts=entry.get("attempts") or 0,
                          last_error=entry.get("last_error") or "",
                          error=entry.get("last_error") or "")
    finally:
        with MCP_CLIENTS_LOCK:
            MCP_RECONNECTING.discard(sid)


def mcp_supervise_loop():
    while not MCP_MONITOR_STOP.wait(2.0):
        try:
            servers = {s["id"]: s for s in load_mcp_servers()}
        except Exception:
            continue
        for sid, server in servers.items():
            with MCP_CLIENTS_LOCK:
                entry = MCP_CLIENTS.get(sid)
            if not entry:
                with MCP_CLIENTS_LOCK:
                    if sid in MCP_RECONNECTING:
                        continue
                    MCP_RECONNECTING.add(sid)
                threading.Thread(target=mcp_reconnect_worker,
                                 args=(sid, server, True), daemon=True).start()
                continue
            state = entry.get("state") or "connected"
            if state == "failed":
                continue
            client = entry.get("client")
            healthy = bool(client and client.alive())
            if healthy and state == "connected" and client.__class__.__name__ == "HttpMCPClient":
                now = time.time()
                if now - float(entry.get("last_health") or 0) >= MCP_HEALTH_INTERVAL:
                    try:
                        client.ping(timeout=5)
                        entry["last_health"] = now
                    except Exception as e:
                        healthy = False
                        mcp_mark_unhealthy(sid, f"{type(e).__name__}: {e}")
            if not healthy:
                mcp_mark_unhealthy(sid, "MCP 连接已断开")
            if entry.get("state") in ("reconnecting", "disconnected"):
                if time.time() >= float(entry.get("next_retry") or 0):
                    with MCP_CLIENTS_LOCK:
                        if sid in MCP_RECONNECTING:
                            continue
                        MCP_RECONNECTING.add(sid)
                    threading.Thread(target=mcp_reconnect_worker,
                                     args=(sid, server, True), daemon=True).start()


def mcp_monitor_start():
    global MCP_MONITOR_THREAD
    MCP_MONITOR_STOP.clear()
    if MCP_MONITOR_THREAD and MCP_MONITOR_THREAD.is_alive():
        return
    MCP_MONITOR_THREAD = threading.Thread(target=mcp_supervise_loop, daemon=True)
    MCP_MONITOR_THREAD.start()


def mcp_start_all():
    for server in load_mcp_servers():
        try:
            start_mcp_client(server)
        except Exception:
            pass


def mcp_reconcile():
    servers = load_mcp_servers()
    keep = {s["id"]: s for s in servers}
    with MCP_CLIENTS_LOCK:
        stale = [sid for sid, entry in MCP_CLIENTS.items()
                 if sid not in keep or
                 entry["signature"] != _mcp_signature(keep[sid])]
    for sid in stale:
        stop_mcp_client(sid)
    for server in servers:
        try:
            start_mcp_client(server)
        except Exception:
            pass


def mcp_stop_all():
    MCP_MONITOR_STOP.set()
    with MCP_CLIENTS_LOCK:
        entries = list(MCP_CLIENTS.values())
        MCP_CLIENTS.clear()
    for entry in entries:
        try:
            entry["client"].close()
        except Exception:
            pass


atexit.register(mcp_stop_all)


def find_mcp_server(server_id):
    for item in load_mcp_servers():
        if item.get("id") == server_id:
            return item
    return None


def load_mcp_status():
    try:
        with open(MCP_STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_mcp_status(status):
    try:
        ensure_cache_dir()
        with MCP_LOCK:
            with open(MCP_STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _mcp_status_one(server, timeout=20):
    try:
        client = start_mcp_client(server)
        tools = client.list_tools()
        return {"ok": True, "state": "connected", "tools": [
            {"name": t.get("name"), "description": t.get("description") or ""}
            for t in tools if isinstance(t, dict) and t.get("name")
        ], "error": ""}
    except Exception as e:
        return {"ok": False, "tools": [], "error": f"{type(e).__name__}: {e}"}


def detect_mcp_servers():
    servers = load_mcp_servers()
    status = load_mcp_status()
    for server in servers:
        status[server["id"]] = _mcp_status_one(server)
    save_mcp_status(status)
    return status


def find_mcp_status(server_id):
    return load_mcp_status().get(server_id) or {}


def agent_models():
    out = []
    for api in load_openai_apis():
        for model in openai_models_for(api):
            if model.get("kind") == "embedding":
                continue
            if model.get("tool_use") == "none":
                continue
            out.append({"name": openai_model_name(api, model["name"]),
                        "model": model["name"], "label": f"{api['name']}/{model['name']}",
                        "backend": "openai", "api_id": api["id"],
                        "capabilities": ["chat"] +
                            (["tools"] if model.get("tool_use") == "native" else []),
                        "tool_use": model.get("tool_use") or "unknown",
                        "input_modalities": model.get("input_modalities") or ["text"],
                        "context": model.get("context") or 1000000,
                        "context_default": bool(model.get("context_default", True)),
                        "context_source": model.get("context_source") or "unknown",
                        "output": model.get("output") or 256000,
                        "output_default": bool(model.get("output_default", True)),
                        "output_source": model.get("output_source") or "unknown",
                        "reasoning_efforts": model.get("reasoning_efforts"),
                        "thinking": model.get("thinking") or
                                    {"supported": None, "values": None,
                                     "default": "off", "mode": "unknown"}})
    return out


load_ollama_url()


def tail_train_log(n=400):
    """取最新一份训练日志的末尾 n 行。

    内存队列被读过就出队了，刷新页面只能补一次；日志同时落盘在 train-logs\\，
    这里直接读文件，刷新多少次都能补回历史。
    """
    try:
        # 只认训练日志的命名：<配置名>-YYYYmmdd-HHMMSS.log，
        # 排除 server- / chat- / sign- / browser-open.log 等其它日志
        names = [f for f in os.listdir(LOG_DIR)
                 if re.search(r"-\d{8}-\d{6}\.log$", f)]
    except OSError:
        return []
    if not names:
        return []
    paths = sorted((os.path.join(LOG_DIR, f) for f in names), key=os.path.getmtime)
    try:
        with open(paths[-1], encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    return lines[-n:]


def tail_train_raw(max_bytes=200000):
    """取最新训练日志尾部的原始文本（保留 CR / ANSI），给 xterm 回填。"""
    try:
        names = [f for f in os.listdir(LOG_DIR)
                 if re.search(r"-\d{8}-\d{6}\.log$", f)]
    except OSError:
        return ""
    if not names:
        return ""
    paths = sorted((os.path.join(LOG_DIR, f) for f in names), key=os.path.getmtime)
    try:
        with open(paths[-1], "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", "replace")


# ==================================================================
# 子进程生命周期：Job Object 兜底
# ==================================================================
# 关掉 start.bat 的 cmd 窗口时，WebUI 可能来不及执行清理代码就被系统结束。
# 把子进程放进一个 KILL_ON_JOB_CLOSE 的 Job：WebUI 进程一消失，内核自动
# 关掉 job 句柄，job 里的训练/服务/对话进程一起被杀。--parent-pid 看门狗
# 仍然保留，两层兜底。
JOB_HANDLE = None


def _make_job_object():
    """建一个“句柄关掉就杀光里面进程”的 Job；失败返回 None（只靠看门狗）。"""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class _Io(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic),
                ("IoInfo", _Io),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000    # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                                ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return None
        return job
    except (AttributeError, OSError):
        return None


def _join_job(proc):
    """把刚起的子进程放进 JOB_HANDLE；失败不影响功能（看门狗仍然有效）。"""
    if not JOB_HANDLE:
        return
    try:
        ctypes.windll.kernel32.AssignProcessToJobObject(JOB_HANDLE, int(proc._handle))
    except (AttributeError, OSError):
        pass


mcp_client.set_process_hook(_join_job)


# ==================================================================
# 子进程管理
# ==================================================================
class Child:
    """一个子进程 + 读线程；退出时统一清理。"""

    def __init__(self, args, cwd=None, log_path=None, on_line=None,
                 stdin_pipe=False, raw=False, on_raw=None):
        self.args = args
        self.raw = raw
        self.on_raw = on_raw
        self.log = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            common.hide_dir(CACHE_DIR)
            # raw 模式必须原样保留 \r，不能做 Windows 的 \n -> \r\n 转换
            self.log = open(log_path, "a", encoding="utf-8", buffering=1,
                            newline="" if raw else None)
        env = os.environ.copy()
        # 所有子进程都固定在项目 Python：不加载用户 site-packages，
        # 也不把当前目录/用户 PYTHONPATH 混进 import 搜索路径。
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        kwargs = dict(
            cwd=cwd or ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if (stdin_pipe and not raw) else subprocess.DEVNULL,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if raw:
            # 训练日志必须保留原始 \r / ANSI；不能让文本模式做 universal-newline。
            kwargs.update(text=False, bufsize=0)
        else:
            kwargs.update(text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.proc = subprocess.Popen(args, **kwargs)
        _join_job(self.proc)
        self.lines = queue.Queue(maxsize=20000)
        self.on_line = on_line
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        if self.raw:
            return self._read_raw()
        try:
            for line in self.proc.stdout:
                line = line.rstrip("\n")
                if self.log:
                    try:
                        self.log.write(line + "\n")
                    except OSError:
                        pass
                if self.on_line:
                    try:
                        self.on_line(line)
                    except Exception:
                        traceback.print_exc()
                try:
                    self.lines.put_nowait(line)
                except queue.Full:
                    pass
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.proc.stdout.close()
            except (OSError, ValueError):
                pass

    def _read_raw(self):
        """二进制读取并保留 CR；xterm.js 自己解释回车和 ANSI。"""
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            fd = self.proc.stdout.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                text = dec.decode(chunk)
                if not text:
                    continue
                if self.log:
                    try:
                        self.log.write(text)
                        self.log.flush()
                    except OSError:
                        pass
                if self.on_raw:
                    try:
                        self.on_raw(text)
                    except Exception:
                        traceback.print_exc()
            tail = dec.decode(b"", final=True)
            if tail:
                if self.log:
                    self.log.write(tail)
                    self.log.flush()
                if self.on_raw:
                    self.on_raw(tail)
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.proc.stdout.close()
            except (OSError, ValueError):
                pass

    def send(self, obj):
        """给子进程发一行 JSON（只对 stdin_pipe=True 的进程有效）。"""
        if not self.proc.stdin:
            return False
        try:
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def alive(self):
        return self.proc.poll() is None

    def stop(self, timeout=5):
        if not self.alive():
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True)
        except OSError:
            pass
        if self.log:
            try:
                self.log.close()
            except OSError:
                pass


CHILDREN = []


def stop_all():
    for c in list(CHILDREN):
        try:
            c.stop()
        except Exception:
            traceback.print_exc()
    CHILDREN.clear()


def spawn(args, log_name=None, on_line=None, stdin_pipe=False,
          raw=False, on_raw=None):
    log_path = os.path.join(LOG_DIR, log_name) if log_name else None
    c = Child(args, ROOT, log_path, on_line, stdin_pipe, raw=raw, on_raw=on_raw)
    CHILDREN.append(c)
    return c


# ==================================================================
# SSE 广播
# ==================================================================
class Hub:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()

    def add(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            self.clients.append(q)
        return q

    def drop(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def push(self, event, data):
        payload = (event, data)
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass


HUB = Hub()


# ==================================================================
# 业务：配置 / 模型 / 语料
# ==================================================================
def list_configs():
    out = []
    d = os.path.join(ROOT, "properties")
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".properties"):
            continue
        p = os.path.join(d, name)
        props = common.load_properties(p)
        desc = ""
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("#") and len(s) > 2:
                        desc = s.lstrip("# ").strip()
                        break
        except OSError:
            pass
        out.append({"name": name, "desc": desc, "props": props,
                    "mtime": os.path.getmtime(p)})
    return out


def list_models():
    out = []
    weights = ("model.safetensors", "pytorch_model.bin")
    for dirpath, dirs, files in os.walk(os.path.join(ROOT, "model")):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if os.path.basename(dirpath).startswith("checkpoint-"):
            continue
        if not os.path.isfile(os.path.join(dirpath, "config.json")):
            continue
        if not any(os.path.isfile(os.path.join(dirpath, w)) for w in weights):
            continue
        total = 0
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
        weight = next((os.path.join(dirpath, w) for w in weights
                       if os.path.isfile(os.path.join(dirpath, w))), "")
        sig = weight + ".sha256"
        out.append({
            "id": os.path.basename(dirpath),
            "path": os.path.relpath(dirpath, ROOT).replace("\\", "/"),
            "size_mb": round(total / 1e6, 1),
            "mtime": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(dirpath))),
            "signed": os.path.isfile(sig),
        })
    return sorted(out, key=lambda m: m["path"])


def load_openai_apis():
    return openai_client.load_apis(OPENAI_CONFIG_FILE)


def save_openai_apis(apis):
    with MCP_LOCK:
        ensure_cache_dir()
    return openai_client.save_apis(OPENAI_CONFIG_FILE, apis)


def load_openai_status():
    try:
        with open(OPENAI_STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_openai_status(status):
    try:
        ensure_cache_dir()
        with MCP_LOCK:
            with open(OPENAI_STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def load_model_overrides():
    try:
        with open(OPENAI_MODEL_OVERRIDES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_model_overrides(data):
    ensure_cache_dir()
    with MCP_LOCK:
        with open(OPENAI_MODEL_OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _override_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return n if n > 0 else None
    text = str(value or "").strip().lower().replace(",", "")
    mult = 1
    if text.endswith("k"):
        mult, text = 1000, text[:-1]
    elif text.endswith("m"):
        mult, text = 1000000, text[:-1]
    try:
        n = int(float(text) * mult)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def apply_model_overrides(api, models):
    """把用户对模型的手工值合入检测结果，手工值不再被自动检测覆盖。"""
    overrides = load_model_overrides().get(api.get("id") or "", {})
    out = []
    for raw in models:
        model = dict(raw)
        ov = overrides.get(model.get("name") or "")
        if isinstance(ov, dict):
            for key in ("context", "output"):
                if key in ov:
                    n = _override_int(ov.get(key))
                    if n:
                        model[key] = n
                        model[key + "_default"] = False
                        model[key + "_source"] = "manual"
            if ov.get("tool_use") in ("native", "none", "unknown"):
                model["tool_use"] = ov["tool_use"]
                model["supports_tools"] = ov["tool_use"] == "native"
            if isinstance(ov.get("input_modalities"), list):
                vals = [str(x) for x in ov["input_modalities"]
                        if str(x) in ("text", "image")]
                if vals:
                    model["input_modalities"] = list(dict.fromkeys(vals))
            if isinstance(ov.get("reasoning_efforts"), dict):
                model["reasoning_efforts"] = {
                    str(k): (None if v is None else str(v))
                    for k, v in ov["reasoning_efforts"].items()
                }
        out.append(model)
    return out


def is_ollama_api(api):
    try:
        raw = ollama.base_url(api.get("base_url"))
        default = ollama.base_url()
        if "ollama" in str(api.get("name") or "").lower():
            return True
        if raw.rstrip("/").lower() == default.rstrip("/").lower():
            return True
        parts = urlsplit(raw)
        return parts.hostname in ("127.0.0.1", "localhost") and parts.port == 11434
    except Exception:
        return False


def augment_ollama_models(api, models):
    """Ollama 的 /v1/models 不带能力，用原生 /api/show 补齐。"""
    if not is_ollama_api(api):
        return models
    base = ollama.base_url(api.get("base_url"))
    for model in models:
        name = model.get("name") or ""
        caps = set(ollama.model_capabilities(name, base))
        if caps:
            is_embed = "embedding" in caps
            model["kind"] = "embedding" if is_embed else "chat"
            model["tool_use"] = "none" if is_embed else (
                "native" if "tools" in caps else "none")
            model["supports_tools"] = model["tool_use"] == "native"
            model["input_modalities"] = ["text"] if is_embed else (
                ["text", "image"] if "vision" in caps else ["text"])
            if is_embed:
                model["thinking"] = {"supported": False, "values": None,
                                     "default": None, "mode": "unsupported"}
                model["reasoning_efforts"] = None
            else:
                info = ollama.model_thinking(name, base)
                if info:
                    model["thinking"] = info
                    values = info.get("values")
                    if values:
                        model["reasoning_efforts"] = {
                            level: (None if level == "off" else level)
                            for level in values
                        }
                    else:
                        model["reasoning_efforts"] = None
                elif "thinking" not in caps:
                    model["thinking"] = {"supported": False, "values": None,
                                         "default": None, "mode": "unsupported"}
                    model["reasoning_efforts"] = None
        context = ollama.model_context(name, base)
        if context:
            model["context"] = context
            model["context_default"] = False
    return models


def detect_openai_apis():
    """启动时检测全部 API 的 /models；GET 不消耗生成 Token。"""
    status = load_openai_status()
    for api in load_openai_apis():
        try:
            models = openai_client.list_model_infos(api)
            models = augment_ollama_models(api, models)
            detected = [dict(m) for m in models]
            models = apply_model_overrides(api, models)
            status[api["id"]] = {"ok": True, "models": models,
                                 "detected": detected, "error": ""}
        except Exception as e:
            status[api["id"]] = {"ok": False, "models": [],
                                 "error": f"{type(e).__name__}: {e}"}
    save_openai_status(status)
    return status


def find_openai_status(api_id):
    return load_openai_status().get(api_id) or {}


def find_openai_api(api_id):
    return openai_client.find_api(load_openai_apis(), api_id)


def public_openai_api(api):
    out = dict(api)
    out["key_set"] = openai_client.key_status(api)
    st = find_openai_status(api["id"])
    base = st.get("detected") or st.get("models") or []
    out["models"] = apply_model_overrides(api, base)
    out["ok"] = bool(st.get("ok"))
    out["error"] = st.get("error") or ""
    out["model_overrides"] = load_model_overrides().get(api["id"]) or {}
    return out


def openai_model_name(api, model):
    return f"api:{api['id']}/{model}"


def parse_openai_model(model):
    raw = str(model or "")
    if not raw.startswith("api:"):
        return None, ""
    rest = raw[4:]
    if "/" not in rest:
        return None, ""
    api_id, upstream = rest.split("/", 1)
    return api_id, upstream


def openai_models_for(api):
    st = find_openai_status(api["id"])
    base = st.get("detected") or st.get("models") or []
    return apply_model_overrides(api, base)


def _thinking_level(value):
    """把旧布尔值或思考等级统一成 off/low/medium/high。"""
    if isinstance(value, bool):
        return "medium" if value else "off"
    text = str(value or "").strip().lower()
    if text in ("", "off", "false", "0", "none", "no"):
        return "off"
    if text in ("true", "on", "yes", "1"):
        return "medium"
    if text in ("low", "medium", "high", "minimal", "max", "xhigh"):
        return text
    return "off"


def openai_model_thinking(api, model):
    for item in openai_models_for(api):
        if item.get("name") == model:
            info = dict(item.get("thinking") or {})
            if item.get("reasoning_efforts"):
                info["reasoning_efforts"] = item.get("reasoning_efforts")
            return info
    return {}


def openai_model_context(api, model):
    for item in openai_models_for(api):
        if item.get("name") == model:
            try:
                return max(1024, int(item.get("context") or 1000000))
            except (TypeError, ValueError):
                return 1000000
    return 1000000


def openai_model_output(api, model):
    for item in openai_models_for(api):
        if item.get("name") == model:
            try:
                return max(1024, int(item.get("output") or 256000))
            except (TypeError, ValueError):
                return 256000
    return 256000


def openai_output_budget(api, model):
    context = openai_model_context(api, model)
    output = openai_model_output(api, model)
    # 单轮输出不能超过上下文；给输入至少保留一半空间。
    return min(output, max(1024, context // 2))


def _context_strip_images(value):
    """上下文估算不把 base64 图片按字符数重复计算。"""
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "image_url":
                out.append({"type": "image_url", "image_url": {"url": "<image>"}})
            else:
                out.append(_context_strip_images(item))
        return out
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "url" and isinstance(item, str) and item.startswith("data:"):
                out[key] = "<image>"
            else:
                out[key] = _context_strip_images(item)
        return out
    return value


def context_tokens(messages, tools=None):
    """按 4 字符/token 估算实际请求压力；图片按固定视觉 token 计。"""
    try:
        clean = _context_strip_images(messages or [])
        text = json.dumps(clean, ensure_ascii=False)
        if tools:
            text += json.dumps(tools, ensure_ascii=False)
        images = 0
        stack = [messages or []]
        while stack:
            value = stack.pop()
            if isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, dict):
                if value.get("type") == "image_url":
                    images += 1
                else:
                    stack.extend(value.values())
        return max(1, (len(text) + 3) // 4 + images * 1200)
    except (TypeError, ValueError):
        return 1


def _usage_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    n = int(value)
    return n if n >= 0 else None


def _same_model_message(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if left.get("role") != right.get("role"):
        return False
    try:
        return (json.dumps(left.get("content"), ensure_ascii=False, sort_keys=True)
                == json.dumps(right.get("content"), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return False


def _agent_tool_guidance(tools):
    names = sorted({str(t.get("function", {}).get("name") or "")
                    for t in tools or []
                    if isinstance(t, dict) and t.get("function")})
    names = [n for n in names if n]
    if not names:
        return "当前没有可用工具，不要生成工具调用。"
    return ("可用工具：" + "、".join(names) + "。\n"
            "工具名和参数必须来自 tools 定义；写入大文件时分块创建/追加，"
            "不要在一次工具调用中塞入过长内容。调用失败时先检查参数和工作区，"
            "不要反复用相同参数重试。")


def _workspace_instructions(workspace_path, limit=32768):
    """读取工作区根目录的 AGENTS.md / CLAUDE.md，作为首个请求的持久上下文。"""
    if not workspace_path:
        return ""
    found = []
    for name in ("AGENTS.md", "CLAUDE.md"):
        path = os.path.join(workspace_path, name)
        try:
            if not os.path.isfile(path) or os.path.getsize(path) > 1024 * 1024:
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        except OSError:
            continue
        if text and text not in found:
            found.append(text)
    if not found:
        return ""
    text = "\n\n".join(found)
    if len(text) > limit:
        text = text[:limit] + "\n\n[工作区指令已截断]"
    return text


def _agent_runtime_message(workspace_path, include_time=True):
    """运行时上下文用 user 快照追加，避免每步重写 system prompt 破坏前缀缓存。"""
    now = datetime.datetime.now().astimezone()
    lines = [
        "<system-reminder>",
        "以下是当前运行时上下文，不是新的用户指令。",
        f"当前工作区绝对路径：{workspace_path or '未设置'}",
    ]
    if include_time:
        lines.append(f"系统时间：{now.isoformat(timespec='milliseconds')}")
        lines.append(f"系统时区：{now.tzname() or ''}")
    instructions = _workspace_instructions(workspace_path)
    if instructions:
        lines.extend(["", "工作区指令：", instructions])
    lines.append("</system-reminder>")
    return {"role": "user", "content": "\n".join(lines)}


def _agent_time_message(workspace_path):
    now = datetime.datetime.now().astimezone()
    return {
        "role": "user",
        "content": (
            "<system-reminder>\n"
            f"Time sampled for this step: {now.isoformat(timespec='milliseconds')}\n"
            f"System time zone: {now.tzname() or ''}\n"
            f"Current workspace: {workspace_path or '未设置'}\n"
            "This is runtime context, not a new user instruction.\n"
            "</system-reminder>"
        ),
    }


def _usage_first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _usage_from_obj(obj):
    """把不同 provider 的 usage 归一成未缓存输入、缓存读/写和输出。"""
    raw = obj.get("usage") if isinstance(obj, dict) else None
    if not isinstance(raw, dict):
        return None
    details = raw.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = {}
    cached = _usage_first(
        _usage_int(raw.get("prompt_cache_hit_tokens")),
        _usage_int(details.get("cached_tokens")),
        _usage_int(raw.get("cache_read_input_tokens")),
        _usage_int(raw.get("cache_read_tokens")),
    )
    cache_write = _usage_first(
        _usage_int(raw.get("cache_creation_input_tokens")),
        _usage_int(raw.get("cache_write_tokens")),
    )
    if cache_write is None:
        cache_write = 0
    miss = _usage_int(raw.get("prompt_cache_miss_tokens"))
    prompt = _usage_int(raw.get("prompt_tokens"))
    input_tokens = _usage_int(raw.get("input_tokens"))
    if prompt is None and miss is not None and cached is not None:
        prompt = miss + cached
    if prompt is None and input_tokens is not None:
        prompt = input_tokens + (cached or 0) + cache_write
    if prompt is None and miss is not None:
        prompt = miss + (cached or 0) + cache_write
    if miss is not None:
        uncached = miss
    elif input_tokens is not None:
        # Anthropic 风格 input_tokens 不包含 cache read/write。
        uncached = input_tokens
    elif prompt is not None and cached is not None:
        uncached = max(0, prompt - cached)
    elif prompt is not None:
        uncached = prompt
    else:
        uncached = None
    completion = _usage_first(
        _usage_int(raw.get("completion_tokens")),
        _usage_int(raw.get("output_tokens")),
    )
    if prompt is None and uncached is None:
        return None
    out = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "uncached_input_tokens": uncached,
        "cache_read_tokens": cached,
        "cache_write_tokens": cache_write,
        "cache_measured": cached is not None,
    }
    out["cache_hit"] = _usage_cache_hit(out)
    return out


def _usage_cache_hit(usage):
    if not isinstance(usage, dict) or not usage.get("cache_measured"):
        return None
    cached = usage.get("cache_read_tokens")
    uncached = usage.get("uncached_input_tokens")
    if cached is None or uncached is None:
        return None
    denom = cached + uncached + (usage.get("cache_write_tokens") or 0)
    if denom <= 0:
        return None
    return min(1.0, cached / denom)


def _merge_usage(total, part):
    """合并 Agent 多个模型请求；缓存只在每个请求都报告时才算。"""
    if not part:
        return total
    if not total:
        out = dict(part)
        out["cache_hit"] = _usage_cache_hit(out)
        return out
    out = dict(total)
    for key in ("prompt_tokens", "completion_tokens", "uncached_input_tokens"):
        a, b = total.get(key), part.get(key)
        out[key] = a + b if a is not None and b is not None else None
    for key in ("cache_read_tokens", "cache_write_tokens"):
        a, b = total.get(key), part.get(key)
        out[key] = a + b if a is not None and b is not None else None
    out["cache_measured"] = bool(total.get("cache_measured")) and bool(
        part.get("cache_measured"))
    out["cache_hit"] = _usage_cache_hit(out)
    return out


def _usage_event(usage):
    if not usage:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cache_read_tokens": usage.get("cache_read_tokens"),
        "cache_write_tokens": usage.get("cache_write_tokens"),
        "uncached_input_tokens": usage.get("uncached_input_tokens"),
        "cache_hit": usage.get("cache_hit"),
        "cache_measured": bool(usage.get("cache_measured")),
    }


def _usage_totals_add(total, usage):
    """把一次模型请求累加到会话总量；缓存字段按“所有请求都有”才可信。"""
    if not usage:
        return total
    out = dict(total or {})
    for key in ("prompt_tokens", "completion_tokens", "uncached_input_tokens",
                "cache_read_tokens", "cache_write_tokens"):
        value = usage.get(key)
        if value is not None:
            out[key] = (out.get(key) or 0) + int(value)
    out["turns"] = int(out.get("turns") or 0) + 1
    measured = bool(usage.get("cache_measured"))
    if "cache_measured" in out:
        out["cache_measured"] = bool(out["cache_measured"]) and measured
    else:
        out["cache_measured"] = measured
    out["cache_hit"] = _usage_cache_hit(out)
    return out


def update_session_usage(sid, usage, kind="agent"):
    """把本轮 usage 累加到会话并返回会话总量。"""
    clean_sid = _session_id(sid)
    if not clean_sid or not usage:
        return None
    path = _session_path(clean_sid)
    if not path:
        return None
    with SESSION_LOCK:
        data = load_session(clean_sid) or {"id": clean_sid}
        data.setdefault("kind", "agent" if kind == "agent" else "chat")
        last = _usage_event(usage)
        totals = _usage_totals_add(data.get("usage_totals"), usage)
        data["last_usage"] = last
        data["usage_totals"] = totals
        data["updated_at"] = time.time()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            return None
        return totals


class ChatStreamError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {str(body or '')[:300]}")
        self.status = status
        self.body = body or ""


def _open_chat_stream(api, payload, timeout):
    """请求流式 usage；代理不支持 stream_options 时自动去掉重试一次。"""
    data = dict(payload)
    want_usage = not openai_client.is_ollama_api(api)
    if want_usage:
        data["stream_options"] = {"include_usage": True}
    conn, resp = openai_client.open_chat(api, data, timeout=timeout)
    if want_usage and resp.status == 400:
        try:
            raw = resp.read().decode("utf-8", errors="replace")
        finally:
            conn.close()
        low = raw.lower()
        if "stream_options" in low or "include_usage" in low:
            data.pop("stream_options", None)
            conn, resp = openai_client.open_chat(api, data, timeout=timeout)
            return conn, resp
        raise ChatStreamError(resp.status, raw)
    return conn, resp



def _trim_for_context(messages, context, reserve=256000):
    """按保守字符数裁剪历史；以工具调用组为单位，保留 system 和最近消息。"""
    budget = max(1024, int(context) - reserve)
    if not messages:
        return messages
    groups = _message_groups(messages)
    system = [g for g in groups if g and isinstance(g[0], dict)
              and g[0].get("role") == "system"]
    rest = [g for g in groups if g not in system]
    kept = []
    used = sum(context_tokens([m for m in g]) for g in system)
    for group in reversed(rest):
        n = context_tokens([m for m in group])
        if used + n > budget and kept:
            break
        kept.append(group)
        used += n
    return [m for g in system + list(reversed(kept)) for m in g]


def _message_content_text(msg):
    content = msg.get("content") if isinstance(msg, dict) else ""
    if isinstance(content, list):
        content = "\n".join(str(x.get("text") or "") for x in content
                            if isinstance(x, dict))
    text = str(content or "")
    calls = msg.get("tool_calls") if isinstance(msg, dict) else None
    if calls:
        text += "\n[tool_calls] " + json.dumps(calls, ensure_ascii=False)
    return text


def _message_groups(messages):
    """把 assistant tool_calls 与对应 tool 结果分成不可拆分的组。"""
    groups = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        group = [msg]
        if (isinstance(msg, dict) and msg.get("role") == "assistant"
                and isinstance(msg.get("tool_calls"), list) and msg["tool_calls"]):
            ids = {str(c.get("id") or "") for c in msg["tool_calls"]
                   if isinstance(c, dict)}
            j = i + 1
            while j < len(messages):
                nxt = messages[j]
                if isinstance(nxt, dict) and nxt.get("role") == "tool":
                    group.append(nxt)
                    j += 1
                else:
                    break
            i = j
        else:
            i += 1
        groups.append(group)
    return groups


def _group_text(group):
    parts = []
    for msg in group:
        role = msg.get("role") if isinstance(msg, dict) else "unknown"
        parts.append(f"[{role}]\n{_message_content_text(msg)}")
    return "\n\n".join(parts)


def _prune_old_tool_results(old_groups, max_chars=24000):
    """压缩摘要前截断过大的工具结果，保留头尾便于模型理解。"""
    for group in old_groups:
        for msg in group:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            text = _message_content_text(msg)
            if len(text) <= max_chars:
                continue
            head = max_chars // 3
            tail = max_chars - head
            msg["content"] = (text[:head] + "\n\n[工具结果过长，中间部分已省略]\n\n"
                              + text[-tail:])
    return old_groups


def _summary_history(api, upstream, old_messages, max_output=8192,
                     prefix=None, tools=None):
    """用原请求前缀 + 旧历史 + 末尾压缩指令生成摘要，尽量复用 prompt cache。"""
    if not api or not old_messages:
        return ""
    old = []
    for group in old_messages:
        old.extend(group)
    instruction = {
        "role": "user",
        "content": (
            "You are now acting as a compaction engine for this AI assistant. "
            "Condense the conversation ABOVE into a structured checkpoint that lets "
            "another model resume the work without losing essential context.\n\n"
            "Output only Markdown with these sections, in order:\n"
            "## Primary Request and Intent\n## Key Technical Concepts\n## Files and Code\n"
            "## Errors and Fixes\n## Pending Jobs\n## Current Work\n## Next Step\n"
            "## Critical Context\n\n"
            "Rules: keep exact file paths, commands, error strings and identifiers; "
            "do not invent facts; omit full code/diffs/repeated logs; output only the "
            "checkpoint text and do not call any tool."
        ),
    }
    payload_messages = list(prefix or []) + old + [instruction]
    payload = {"model": upstream, "stream": False,
               "messages": payload_messages, "max_tokens": max_output}
    if tools:
        payload["tools"] = tools
    try:
        conn, resp = openai_client.open_chat(
            api, payload, timeout=openai_client.chat_timeout(api))
        try:
            raw = resp.read().decode("utf-8", errors="replace")
        finally:
            conn.close()
        if resp.status < 200 or resp.status >= 300:
            return ""
        data = json.loads(raw)
        choice = (data.get("choices") or [{}])[0]
        return str((choice.get("message") or {}).get("content") or "").strip()
    except Exception:
        return ""


def compact_for_context(api, upstream, messages, context, output, tools=None,
                        force=False):
    """上下文压缩：优先摘要旧历史；失败时退回确定性裁剪。"""
    context = max(1024, int(context or 1000000))
    output = max(0, int(output or 256000))
    threshold = max(1024, min(int(context * 0.8),
                             context - output - 65536))
    if not force and context_tokens(messages, tools) <= threshold:
        return messages
    groups = _message_groups(messages)
    system = [m for m in groups if m and isinstance(m[0], dict)
              and m[0].get("role") == "system"]
    rest = [g for g in groups if g not in system]
    if len(rest) < 2:
        return _trim_for_context(messages, context, output)
    retain = max(2048, int(max(0, context - output) * 0.16))
    keep = []
    used = 0
    for group in reversed(rest):
        n = context_tokens([m for m in group])
        if used + n > retain and keep:
            break
        keep.append(group)
        used += n
    keep.reverse()
    old = rest[:len(rest) - len(keep)]
    old = _prune_old_tool_results(old)
    prefix = system[0] if system else []
    summary = _summary_history(api, upstream, old, prefix=prefix, tools=tools)
    if not summary:
        return _trim_for_context(messages, context, output)
    summary = summary[:16000]
    tail = [m for group in keep for m in group]
    out = list(prefix) + [{
        "role": "user",
        "content": (
            "This is an automatically generated checkpoint condensing an earlier "
            "span of the conversation to free up context. Treat the captured context "
            "as established background and build on it without restating it. "
            "Continue the task directly from the messages that follow, without "
            "acknowledging this checkpoint.\n\n"
            "<compacted-summary>\n" + summary + "\n</compacted-summary>"
        ),
    }] + tail
    if not force and context_tokens(out, tools) >= context_tokens(messages, tools):
        return _trim_for_context(messages, context, output)
    return out


def _is_context_overflow_text(text):
    low = str(text or "").lower()
    return any(x in low for x in (
        "context_length_exceeded", "maximum context length", "context window",
        "too many tokens", "token limit", "context length", "reduce the length",
        "maximum number of tokens",
    ))


def openai_thinking_payload(info, level):
    """把界面上的思考档位映射成上游 API 参数。"""
    info = info or {}
    efforts = info.get("reasoning_efforts")
    if info.get("supported") is False and not isinstance(efforts, dict):
        return {}
    if isinstance(efforts, dict):
        if level == "off":
            return {}
        if level in efforts:
            value = efforts[level]
            if value is None:
                return {}
            return {"reasoning_effort": str(value)}
        return {}
    if level == "off":
        if info.get("mode") == "boolean":
            return {"thinking": False}
        return {}
    if info.get("mode") == "boolean":
        return {"thinking": True}
    return {"reasoning_effort": level}


def ollama_thinking_payload(info, level):
    """Ollama 原生 think 支持布尔和字符串；OpenAI 兼容层同时带 reasoning_effort。"""
    info = info or {}
    if info.get("supported") is False:
        return {}
    if level == "off":
        return {"think": False, "reasoning_effort": "none"}
    think = True if info.get("mode") == "boolean" else level
    return {"think": think, "reasoning_effort": level}


def list_chat_configs(include_api=True):
    """对话下拉用：本地配置 + OpenAI 兼容 API 的对话模型。"""
    out = []
    for c in common.chat_configs(ROOT):
        out.append({"name": c["name"], "model": c["model"], "backend": "local"})
    if include_api:
        for api in load_openai_apis():
            for model in openai_models_for(api):
                if model.get("kind") == "embedding":
                    continue
                out.append({"name": openai_model_name(api, model["name"]),
                            "model": model["name"],
                            "label": f"{api['name']}/{model['name']}",
                            "backend": "openai", "api_id": api["id"],
                            "tool_use": model.get("tool_use") or "unknown",
                            "supports_tools": model.get("tool_use") == "native",
                            "context": model.get("context") or 1000000,
                            "context_default": bool(model.get("context_default", True)),
                            "context_source": model.get("context_source") or "unknown",
                            "output": model.get("output") or 256000,
                            "output_default": bool(model.get("output_default", True)),
                            "output_source": model.get("output_source") or "unknown",
                            "input_modalities": model.get("input_modalities") or ["text"],
                            "reasoning_efforts": model.get("reasoning_efforts"),
                            "thinking": model.get("thinking") or
                                        {"supported": None, "values": None,
                                         "default": "off", "mode": "unknown"}})
    return out


def list_service_configs():
    """服务页下拉用：本地配置 + OpenAI 兼容 API 的对话 / 嵌入模型。"""
    out = []
    for c in common.chat_configs(ROOT):
        out.append({"name": c["name"], "model": c["model"],
                    "backend": "local", "kind": "chat"})
    for api in load_openai_apis():
        for model in openai_models_for(api):
            out.append({"name": openai_model_name(api, model["name"]),
                        "model": model["name"],
                        "label": f"{api['name']}/{model['name']}",
                        "backend": "openai", "api_id": api["id"],
                        "kind": model.get("kind") or "chat",
                        "context": model.get("context") or 1000000,
                        "context_default": bool(model.get("context_default", True)),
                        "output": model.get("output") or 256000,
                        "output_default": bool(model.get("output_default", True))})
    return out


def list_corpora():
    out = []
    base = os.path.join(ROOT, "res")
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        total = 0
        n = 0
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
                n += 1
        out.append({"dir": f"res/{name}", "files": n, "gb": round(total / 1e9, 2)})
    return out


# ==================================================================
# HTTP
# ==================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "dlqq-webui/1.0"

    def log_message(self, fmt, *args):        # 静音默认访问日志
        return

    # ---- 工具 ----
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"ok": False, "error": msg}, code)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, event, data):
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _auth(self, qs):
        tok = (qs.get("t") or [""])[0]
        if tok != STATE["token"]:
            self._err(403, "token 无效，请用控制台打印的完整地址打开")
            return False
        return True

    # ---- 静态 ----
    def _static(self, path):
        rel = path.lstrip("/") or "index.html"
        if ".." in rel:
            self._err(400, "bad path")
            return
        if rel == "logo.ico":                      # 站点图标用 .runtime 里那张
            fp = LOGO_PATH
        else:
            fp = os.path.join(WEB_DIR, rel)
        if not os.path.isfile(fp):
            self._err(404, f"没有文件：{rel}")
            return
        ext = os.path.splitext(fp)[1].lower()
        ctype = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml",
                 ".ico": "image/x-icon", ".woff2": "font/woff2", ".woff": "font/woff",
                 ".ttf": "font/ttf", ".otf": "font/otf",
                 ".txt": "text/plain; charset=utf-8"}.get(ext, "application/octet-stream")
        if ext == "" and os.path.basename(fp).lower().startswith("license"):
            ctype = "text/plain; charset=utf-8"
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # 页面/脚本更新后避免浏览器拿旧 index.html 或旧 JS
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query, keep_blank_values=True)
        path = u.path
        if not path.startswith("/api/"):
            self._static(path)
            return
        if not self._auth(qs):
            return

        if path == "/api/configs":
            self._json({"ok": True, "configs": list_configs()})
        elif path == "/api/sessions":
            reg = load_workspace_registry()
            sid = (qs.get("id") or [""])[0]
            if sid:
                data = load_session(sid)
                if not data:
                    return self._err(404, "会话不存在")
                extra = {}
                if (qs.get("history_page") or [""])[0]:
                    page = session_history_page(
                        data,
                        (qs.get("history_before") or [None])[0],
                        (qs.get("history_limit") or ["24"])[0])
                    data = dict(data)
                    data["history"] = page.pop("history")
                    extra = page
                return self._json({"ok": True, "workspaces": reg.get("workspaces") or [],
                                   "workspace": default_workspace(reg),
                                   "session": data, **extra})
            kind = (qs.get("kind") or [""])[0] or None
            workspace = (qs.get("workspace") or [""])[0] or None
            self._json({"ok": True, "workspaces": reg.get("workspaces") or [],
                        "workspace": default_workspace(reg),
                        "last_agent_workspace": reg.get("last_agent_workspace") or "",
                        "last_agent_session": reg.get("last_agent_session") or {},
                        "last_chat_session": reg.get("last_chat_session") or "",
                        "sessions": list_sessions(kind, workspace)})
        elif path == "/api/workspaces":
            reg = load_workspace_registry()
            self._json({"ok": True, "workspaces": reg.get("workspaces") or [],
                        "default": default_workspace(reg),
                        "last_agent_workspace": reg.get("last_agent_workspace") or ""})
        elif path == "/api/workspace":
            self._json({"ok": True, "workspace": load_workspace()})
        elif path == "/api/models":
            self._json({"ok": True, "models": list_models()})
        elif path == "/api/chat/configs":
            self._json({"ok": True, "configs": list_chat_configs()})
        elif path == "/api/service/configs":
            self._json({"ok": True, "configs": list_service_configs()})
        elif path == "/api/corpora":
            self._json({"ok": True, "corpora": list_corpora()})
        elif path == "/api/train/status":
            c = STATE["train"]
            self._json({"ok": True, "running": bool(c and c.alive()),
                        "config": STATE.get("train_config")})
        elif path == "/api/train/log":
            c = STATE["train"]
            if c:                       # 顺手清空内存队列，避免长时间训练把内存吃满
                while True:
                    try:
                        c.lines.get_nowait()
                    except queue.Empty:
                        break
            self._json({"ok": True, "lines": tail_train_log(400),
                        "raw": tail_train_raw()})
        elif path == "/api/server/status":
            c = STATE["server"]
            self._json({"ok": True, "running": bool(c and c.alive()),
                        "port": STATE.get("server_port")})
        elif path == "/api/env":
            check = (qs.get("check") or ["0"])[0] not in ("", "0", "false")
            py, source = common.resolve_python(ROOT)
            c = STATE.get("env_install")
            installing = bool(c and c.alive())
            if not py:
                return self._json({"ok": False, "python": "", "source": source,
                                   "error": source, "info": None, "missing": [],
                                   "installing": installing})
            known = verified_info(py)
            info = known["info"] if known else None
            if check:
                info = probe_python(py, use_cache=False)
                if info.get("ok"):
                    remember_verified(py, info)
            missing = [n for n in ("torch", "numpy", "transformers")
                       if not (info or {}).get(n)]
            gpu = detect_nvidia()
            recommend = recommend_torch(gpu, info)
            return self._json({
                "ok": bool(info and info.get("ok") and not missing),
                "python": py,
                "source": source,
                "verified": bool(info and info.get("ok")),
                "info": info,
                "missing": missing,
                "installing": installing,
                "webui_python": sys.executable,
                "system": {"gpu": gpu, "torch_recommend": recommend},
            })
        elif path == "/api/env/latest":
            py, _ = common.resolve_python(ROOT)
            known = verified_info(py) if py else None
            info = known["info"] if known else None
            if py and not info:
                info = probe_python(py, use_cache=True)
            gpu = detect_nvidia()
            latest = latest_versions(info, gpu)
            self._json(latest)
        elif path == "/api/env/ollama":
            urls = qs.get("url")
            custom = ""
            if urls is not None:
                custom = (urls[0] or "").strip()
                try:
                    custom = ollama.set_base_url(custom)
                    save_ollama_url(custom or "")
                except ValueError as e:
                    return self._err(400, str(e))
            self._json(ollama.detect_environment(custom))
        elif path == "/api/mcp/configs":
            self._json({"ok": True, "servers": load_mcp_servers(),
                        "status": load_mcp_status()})
        elif path == "/api/openai/configs":
            self._json({"ok": True,
                        "apis": [public_openai_api(a) for a in load_openai_apis()],
                        "status": load_openai_status()})
        elif path == "/api/agent/configs":
            self._json({"ok": True, "models": agent_models(),
                        "servers": load_mcp_servers(),
                        "status": load_mcp_status()})
        elif path == "/api/agent/tools":
            sid = (qs.get("server") or [""])[0]
            server = find_mcp_server(sid)
            if not server:
                return self._err(404, "MCP 服务器不存在")
            status = find_mcp_status(sid)
            if not status.get("tools"):
                status = _mcp_status_one(server)
                st = load_mcp_status()
                st[sid] = status
                save_mcp_status(st)
            self._json({"ok": bool(status.get("ok")), "tools": status.get("tools") or [],
                        "error": status.get("error") or ""})
        elif path == "/api/events":
            self._events()
        else:
            self._err(404, f"未知接口：{path}")

    def do_POST(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        path = u.path
        if not self._auth(qs):
            return
        body = self._body()

        try:
            if path == "/api/config/save":
                self._config_save(body)
            elif path == "/api/env/install":
                self._env_install(body)
            elif path == "/api/train/start":
                self._train_start(body)
            elif path == "/api/train/stop":
                self._stop("train")
            elif path == "/api/chat":
                self._chat(body, qs)
            elif path == "/api/server/start":
                self._server_start(body)
            elif path == "/api/server/stop":
                self._stop("server")
            elif path == "/api/mcp/configs":
                try:
                    servers = save_mcp_servers(body.get("servers") or [])
                except ValueError as e:
                    return self._err(400, str(e))
                self._json({"ok": True, "servers": servers})
            elif path == "/api/mcp/test":
                self._mcp_test(body)
            elif path == "/api/mcp/detect":
                status = detect_mcp_servers()
                self._json({"ok": True, "status": status})
            elif path == "/api/mcp/restart":
                self._mcp_restart(body)
            elif path == "/api/openai/configs":
                try:
                    apis = save_openai_apis(body.get("apis") or [])
                except ValueError as e:
                    return self._err(400, str(e))
                status = detect_openai_apis()
                self._json({"ok": True,
                            "apis": [public_openai_api(a) for a in apis],
                            "status": status})
            elif path == "/api/openai/detect":
                status = detect_openai_apis()
                self._json({"ok": True, "status": status})
            elif path == "/api/openai/test":
                self._openai_test(body)
            elif path == "/api/openai/model":
                self._openai_model_save(body)
            elif path == "/api/sessions":
                data = save_session(body)
                if not data:
                    return self._err(400, "会话数据不合法")
                self._json({"ok": True, "session": data,
                            "sessions": list_sessions(),
                            "workspaces": load_workspace_registry().get("workspaces") or []})
            elif path == "/api/sessions/delete":
                ok = delete_session(body.get("id") or "")
                self._json({"ok": ok, "sessions": list_sessions()})
            elif path == "/api/sessions/rename":
                data = rename_session(body.get("id") or "", body.get("title") or "")
                if not data:
                    return self._err(404, "会话不存在")
                self._json({"ok": True, "session": data})
            elif path == "/api/sessions/move":
                data = move_session(body.get("id") or "", body.get("workspace_id") or "")
                if not data:
                    return self._err(400, "会话或工作区不存在，且只有 Agent 会话可以移动")
                self._json({"ok": True, "session": data,
                            "sessions": list_sessions()})
            elif path == "/api/workspaces/pick":
                try:
                    picked = pick_workspace_folder()
                except Exception as e:
                    return self._err(500, f"{type(e).__name__}: {e}")
                self._json({"ok": True, "path": picked or ""})
            elif path == "/api/workspaces":
                action = str(body.get("action") or "add").lower()
                if action == "rename":
                    w = rename_workspace(body.get("id") or "", body.get("title") or "")
                    if not w:
                        return self._err(404, "工作区不存在")
                    reg = load_workspace_registry()
                    return self._json({"ok": True, "workspace": w,
                                       "workspaces": reg.get("workspaces") or []})
                if action == "delete":
                    ok = delete_workspace(body.get("id") or "")
                    reg = load_workspace_registry()
                    return self._json({"ok": ok,
                                       "workspaces": reg.get("workspaces") or [],
                                       "default": default_workspace(reg)})
                try:
                    w = add_workspace(body.get("path") or "", body.get("title") or "")
                except ValueError as e:
                    return self._err(400, str(e))
                reg = load_workspace_registry()
                self._json({"ok": True, "workspace": w,
                            "workspaces": reg.get("workspaces") or []})
            elif path == "/api/agent/chat":
                self._agent_chat(body)
            elif path == "/api/sign":
                self._sign(body)
            else:
                self._err(404, f"未知接口：{path}")
        except Exception as e:
            traceback.print_exc()
            self._err(500, f"{type(e).__name__}: {e}")

    # ---- 业务处理 ----
    def _config_save(self, body):
        name = body.get("name") or ""
        updates = body.get("updates") or {}
        if not name.endswith(".properties") or "/" in name or "\\" in name:
            return self._err(400, "配置名不合法")
        p = os.path.join(ROOT, "properties", name)
        if not os.path.isfile(p):
            return self._err(404, f"没有这个配置：{name}")
        backup = os.path.join(CACHE_DIR, "cfg-backup")
        ensure_cache_dir()
        os.makedirs(backup, exist_ok=True)
        import shutil
        shutil.copy2(p, os.path.join(backup, f"{name}.{int(time.time())}.bak"))
        changed, added = common.save_properties(p, updates)
        self._json({"ok": True, "changed": changed, "added": added})

    def _env_install(self, body):
        """用项目内置 Python 安装训练 / 对话依赖。"""
        c = STATE.get("env_install")
        if c and c.alive():
            return self._err(409, "环境依赖正在安装，请等待完成")
        if (STATE.get("train") and STATE["train"].alive()) or (
                STATE.get("server") and STATE["server"].alive()):
            return self._err(409, "训练或桥接服务正在运行，请先停止再安装依赖")
        with CHAT_LOCK:
            _stop_worker()                # 避免 pip 覆盖正在使用的 torch DLL
        py, source = common.resolve_python(ROOT)
        if not py:
            return self._err(400, source)
        index = (body.get("torch_index") or "auto").strip()
        torch_type = (body.get("torch_type") or "").strip().lower()
        spec = (body.get("torch_version") or "").strip()
        torch_version = spec
        suffix = ""
        if spec:
            m = re.fullmatch(r"([0-9][0-9A-Za-z.!_-]*)(?:\+(cpu|cu\d+|xpu))?", spec)
            if not m:
                return self._err(400, "torch_version 不合法，示例：2.14.1 或 2.14.1+cu126")
            torch_version = m.group(1)
            suffix = m.group(2) or ""
        if suffix:
            index = suffix
        elif torch_type in ("cpu", "xpu"):
            index = torch_type
        elif torch_type == "gpu" and index not in ("cu126", "cu128"):
            index = recommend_torch(detect_nvidia())["index"]
        if (index not in ("auto", "cpu", "cu126", "cu128", "cu130", "xpu")
                and not index.startswith(("https://", "http://"))):
            return self._err(400, "torch_index 不合法")
        if index == "auto":
            index = recommend_torch(detect_nvidia())["index"]
        numpy_version = (body.get("numpy_version") or "").strip()
        transformers_version = (body.get("transformers_version") or "").strip()
        for label, value in (("numpy_version", numpy_version),
                             ("transformers_version", transformers_version)):
            if value and not re.fullmatch(r"[0-9][0-9A-Za-z.!_-]*", value):
                return self._err(400, f"{label} 不合法，示例：2.5.4")

        known = verified_info(py)
        info = known["info"] if known else probe_python(py, use_cache=True)
        installed_torch = (info.get("torch_version") or "").strip()
        installed_base = installed_torch.split("+", 1)[0]
        m = re.search(r"\+(cpu|cu\d+|xpu)", installed_torch)
        installed_channel = m.group(1) if m else ""
        torch_match = (index == installed_channel) if installed_channel else (index == "cpu")
        skip_torch = bool(info.get("torch")) and (
            (not torch_version or torch_version == installed_base) and torch_match)
        skip_numpy = bool(info.get("numpy")) and (
            not numpy_version or numpy_version == (info.get("numpy_version") or ""))
        skip_transformers = bool(info.get("transformers")) and (
            not transformers_version or
            transformers_version == (info.get("transformers_version") or ""))
        if skip_torch and skip_numpy and skip_transformers:
            return self._json({"ok": True, "skipped": True,
                               "message": "已安装，无需重复安装"})

        args = [py, os.path.join(ROOT, "code", "env_install.py"),
                "--torch-index", index]
        if skip_torch:
            args.append("--skip-torch")
        if skip_numpy:
            args.append("--skip-numpy")
        if skip_transformers:
            args.append("--skip-transformers")
        if torch_version and not skip_torch:
            args += ["--torch-version", torch_version]
        if numpy_version and not skip_numpy:
            args += ["--numpy-version", numpy_version]
        if transformers_version and not skip_transformers:
            args += ["--transformers-version", transformers_version]
        if body.get("no_cache"):
            args.append("--no-cache-dir")
        name = f"env-install-{time.strftime('%Y%m%d-%H%M%S')}.log"
        PROBE_CACHE.clear()
        VERIFIED.clear()
        STATE["env_install"] = spawn(
            args, name,
            on_line=lambda line: HUB.push("env-log", {"line": line}))
        self._json({"ok": True, "log": name})

    def _train_start(self, body):
        cfg = body.get("config") or ""
        if not cfg.endswith(".properties"):
            return self._err(400, "请选择一个 .properties 配置")
        p = os.path.join(ROOT, "properties", os.path.basename(cfg))
        if not os.path.isfile(p):
            return self._err(404, f"没有这个配置：{cfg}")
        c = STATE["train"]
        if c and c.alive():
            return self._err(409, "已有训练在跑")
        name = os.path.splitext(os.path.basename(p))[0]
        py, src = common.resolve_python(ROOT)
        if not py:
            return self._err(400, src)
        known = verified_info(py)
        if not known:
            return self._err(400, "项目 Python 还没检测过；请先在「环境」页点“检测环境”")
        info = known["info"]
        miss = [n for n in ("torch", "numpy", "transformers") if not info.get(n)]
        if miss:
            return self._err(400, f"项目 Python 缺少 {' / '.join(miss)}；请先在「环境」页安装训练依赖")
        try:
            ver = tuple(int(x) for x in info["version"].split(".")[:2])
        except (KeyError, ValueError):
            ver = (0, 0)
        if ver < (3, 11):
            return self._err(400, f"Python 版本过低（需要 3.11+）：{info['version']}")
        log_name = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        args = [py, os.path.join(ROOT, "code", "train.py"),
                "--config", os.path.relpath(p, ROOT), "--parent-pid", str(os.getpid())]
        # resume 由界面按钮决定：开始训练=从头，继续训练=接着存档（没有存档时训练脚本会自动从头）
        args.append("resume=true" if body.get("resume") else "resume=false")
        STATE["train"] = spawn(
            args, log_name, raw=True,
            on_raw=lambda data: HUB.push("train-raw", {"data": data}))
        STATE["train_config"] = os.path.basename(p)
        self._json({"ok": True, "log": log_name})

    def _stop(self, which):
        c = STATE.get(which)
        if c and c.alive():
            c.stop()
        STATE[which] = None
        self._json({"ok": True})

    def _chat(self, body, qs, lan=False):
        raw_cfg = str(body.get("config") or "").strip()
        if raw_cfg.startswith("api:"):
            api_id, upstream = parse_openai_model(raw_cfg)
            if not api_id or not upstream:
                return self._err(400, "API 模型名不合法")
            return self._chat_openai(api_id, upstream, body, lan)
        cfg = common.config_stem(raw_cfg)
        model = body.get("model") or ""
        # 前端传 messages（多轮）；也兼容旧的单条 text
        msgs = body.get("messages")
        if isinstance(msgs, list) and msgs:
            parts = []
            for i, m in enumerate(msgs):
                c = m.get("content") or ""
                if isinstance(c, list):
                    c = "".join(p.get("text", "") for p in c
                                if isinstance(p, dict) and p.get("type") == "text")
                c = str(c)
                if m.get("role") == "user":
                    parts.append(c if i == 0 else "用户：" + c)
                else:
                    parts.append("助手：" + c)
            text = "\n".join(parts)
        else:
            text = (body.get("text") or "").strip()
        if not text:
            return self._err(400, "消息为空")

        # Ollama 模型：直接把原始 messages 转给 Ollama，不进入本地 ChatSession
        if ollama.is_remote(cfg):
            if not ollama.is_chat_model(ollama.model_tag(cfg)):
                return self._err(400, "这个 Ollama 模型不支持对话")
            return self._chat_ollama(cfg, body, text, lan)

        # 按配置加载：模型路径和对话参数都来自 properties\<配置>.properties
        if cfg:
            if "/" in cfg or "\\" in cfg or cfg.startswith("."):
                return self._err(400, "配置名不合法")
            path, _props = common.config_model(ROOT, cfg)
            if not path:
                return self._err(404, f"配置 {cfg} 对应的权重不存在（检查 model=）")
        else:
            # 兼容老页面：只给相对模型目录
            path = os.path.join(ROOT, model.replace("/", os.sep)) if model else ""
            if not path or not os.path.isdir(path):
                return self._err(404, f"模型目录不存在：{model}")
        # 训练中且同名模型 → 拒绝（与方案一致）
        running = STATE.get("train") and STATE["train"].alive()
        training_model = ""
        if running and STATE.get("train_config"):
            props = common.load_properties(os.path.join(ROOT, "properties", STATE["train_config"]))
            training_model = (props.get("model") or "").strip()
        mid = os.path.basename(os.path.normpath(path))
        if running and training_model and mid == training_model:
            return self._err(409, f"{mid} 正在训练，换一个模型对话")

        key = cfg or mid
        py, src = common.resolve_python(ROOT)
        if not py:
            return self._err(400, src)
        known = verified_info(py)
        if not known:
            if not lan:
                return self._err(400, "项目 Python 还没检测过；请先在「环境」页点“检测环境”")
            probe = probe_python(py, use_cache=False)
            if not probe.get("ok") or not probe.get("torch") or not probe.get("transformers"):
                return self._err(400, "对话失败：项目 Python 环境检测不通过：" +
                                 (probe.get("error") or "缺少 torch / transformers"))
            remember_verified(py, probe)
            known = {"info": probe}
        if not known["info"].get("torch") or not known["info"].get("transformers"):
            return self._err(400, "项目 Python 缺少 torch / transformers；请先在「环境」页安装训练依赖")
        with CHAT_LOCK:
            try:
                w = ensure_worker(key, py, key)
            except RuntimeError as e:
                return self._err(502, str(e))
            rid = f"{int(time.time() * 1000)}-{threading.get_ident()}"
            if not w["child"].send({"id": rid, "op": "chat", "text": text}):
                _stop_worker()
                return self._err(502, "对话进程已退出，请重试")
            self._sse_start()
            deadline = time.time() + CHAT_TIMEOUT
            try:
                while True:
                    if time.time() > deadline:
                        self._sse("error", {"error": "生成超时"})
                        break
                    try:
                        msg = w["queue"].get(timeout=1.0)
                    except queue.Empty:
                        if not w["child"].alive():
                            _stop_worker()
                            self._sse("error", {"error": "对话进程已退出"})
                            break
                        continue
                    if msg.get("id") != rid:
                        continue
                    ev = msg.get("event")
                    if ev == "meta":
                        self._sse("meta", {"model": mid, "config": cfg or None,
                                           "info": msg.get("info") or {}})
                    elif ev == "token":
                        self._sse("token", {"text": msg.get("text") or ""})
                    elif ev == "done":
                        self._sse("done", {"text": msg.get("text") or ""})
                        break
                    elif ev == "error":
                        self._sse("error", {"error": msg.get("error") or "未知错误"})
                        break
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    def _chat_ollama(self, cfg, body, text, lan=False):
        """把 WebUI 的对话请求转给 Ollama，并把 SSE 转成前端 token 事件。"""
        tag = ollama.model_tag(cfg)
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            msgs = [{"role": "user", "content": text}]
        payload = {"model": tag, "messages": msgs, "stream": True}
        for key in ("temperature", "top_p", "max_tokens",
                    "max_completion_tokens", "tools"):
            if key in body and body[key] is not None:
                payload[key] = body[key]
        if "thinking" in body:
            level = _thinking_level(body["thinking"])
            payload.update(ollama_thinking_payload(
                ollama.model_thinking(tag), level))

        try:
            conn, resp = ollama.open_chat(payload, timeout=600)
        except Exception as e:
            return self._err(502, f"无法连接 Ollama：{type(e).__name__}: {e}")

        if resp.status < 200 or resp.status >= 300:
            try:
                raw = resp.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            conn.close()
            return self._err(resp.status, f"Ollama 返回错误：{raw[:500]}")

        self._sse_start()
        parts = []
        reasoning_parts = []
        try:
            self._sse("meta", {"model": cfg, "config": cfg,
                               "info": {"backend": "ollama",
                                        "thinking": bool(body.get("thinking"))}})
            for obj in ollama.iter_sse(resp):
                reason = ollama.delta_reasoning(obj)
                if reason:
                    reasoning_parts.append(reason)
                    self._sse("reasoning", {"text": reason})
                piece = ollama.delta_text(obj)
                if piece:
                    parts.append(piece)
                    self._sse("token", {"text": piece})
            self._sse("done", {"text": "".join(parts),
                               "reasoning": "".join(reasoning_parts)})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            try:
                self._sse("error", {"error": f"{type(e).__name__}: {e}"})
            except OSError:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _chat_openai(self, api_id, model, body, lan=False):
        """把 WebUI 对话转给用户配置的 OpenAI 兼容服务。"""
        api = find_openai_api(api_id)
        if not api:
            return self._err(404, "API 服务不存在")
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            text = (body.get("text") or "").strip()
            if not text:
                return self._err(400, "消息为空")
            msgs = [{"role": "user", "content": text}]
        msgs = merge_session_history(
            _session_id(body.get("session_id")), msgs, body.get("history_start"))
        payload = {"model": model, "messages": msgs, "stream": True}
        for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens"):
            if key in body and body[key] is not None:
                payload[key] = body[key]
        output_budget = openai_output_budget(api, model)
        payload.setdefault("max_tokens", output_budget)
        if "thinking" in body:
            level = _thinking_level(body.get("thinking"))
            payload.update(openai_thinking_payload(
                openai_model_thinking(api, model), level))
        overflow_retries = 0
        while True:
            payload["messages"] = compact_for_context(
                api, model, msgs, openai_model_context(api, model),
                output_budget, body.get("tools"),
                force=overflow_retries > 0)
            try:
                conn, resp = _open_chat_stream(
                    api, payload, timeout=openai_client.chat_timeout(api))
            except ChatStreamError as e:
                if overflow_retries < 1 and _is_context_overflow_text(e.body):
                    overflow_retries += 1
                    continue
                return self._err(e.status, f"API 返回错误：{e.body[:500]}")
            except Exception as e:
                return self._err(502, f"无法连接 API：{type(e).__name__}: {e}")
            if resp.status < 200 or resp.status >= 300:
                try:
                    raw = resp.read().decode("utf-8", errors="replace")
                except Exception:
                    raw = ""
                conn.close()
                if overflow_retries < 1 and _is_context_overflow_text(raw):
                    overflow_retries += 1
                    continue
                return self._err(resp.status, f"API 返回错误：{raw[:500]}")
            break

        self._sse_start()
        parts = []
        reasoning_parts = []
        usage_total = None
        try:
            self._sse("context", {
                "used": context_tokens(payload.get("messages")),
                "total": openai_model_context(api, model),
                "output": openai_output_budget(api, model),
            })
            self._sse("meta", {"model": f"{api['name']}/{model}", "config": f"{api['name']}/{model}",
                               "info": {"backend": "openai",
                                        "thinking": _thinking_level(body.get("thinking")) != "off"}})
            for obj in ollama.iter_sse(resp):
                part = _usage_from_obj(obj)
                if part:
                    usage_total = part
                reason = ollama.delta_reasoning(obj)
                if reason:
                    reasoning_parts.append(reason)
                    self._sse("reasoning", {"text": reason})
                piece = ollama.delta_text(obj)
                if piece:
                    parts.append(piece)
                    self._sse("token", {"text": piece})
            done = {"text": "".join(parts),
                    "reasoning": "".join(reasoning_parts)}
            usage = _usage_event(usage_total)
            if usage:
                done["usage"] = usage
                totals = update_session_usage(
                    _session_id(body.get("session_id")), usage_total, "chat")
                if totals:
                    done["session_usage"] = totals
            self._sse("done", done)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            try:
                self._sse("error", {"error": f"{type(e).__name__}: {e}"})
            except OSError:
                pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _server_start(self, body):
        c = STATE["server"]
        if c and c.alive():
            return self._err(409, "服务已在运行")
        port = int(body.get("port") or 49128)
        if port_in_use(port):
            return self._err(409, f"端口 {port} 已被占用（可能有上次残留的服务进程），"
                                  "请先结束它，或换一个端口")
        raw_targets = [str(n).strip() for n in (body.get("configs") or [])
                       if isinstance(n, str) and str(n).strip()]
        cfgs = [common.config_stem(n) for n in raw_targets
                if not n.startswith("api:")]
        api_ids = [n[4:].split("/", 1)[0] for n in raw_targets
                   if n.startswith("api:")]
        targets = raw_targets or [c["name"] for c in common.chat_configs(ROOT)]
        local_targets = [common.config_stem(n) for n in targets
                         if not n.startswith("api:") and not ollama.is_remote(n)]
        if not local_targets and not api_ids:
            # “全部配置”时 API 配置由 http_server 默认全部加载
            api_ids = []
        probe = local_targets[0] if local_targets else ""
        if not probe:
            all_cfgs = common.chat_configs(ROOT)
            probe = all_cfgs[0]["name"] if all_cfgs else ""
        if not local_targets and not api_ids and raw_targets:
            return self._err(400, "没有可用配置，无法启动桥接服务")
        py, source = common.resolve_python(ROOT)
        if not py:
            return self._err(400, source)
        known = verified_info(py)
        if not known:
            info = probe_python(py, use_cache=False)
            if not info.get("ok"):
                return self._err(400, info.get("error") or "项目 Python 检测失败")
            remember_verified(py, info)
            known = {"info": info}
        if not known["info"].get("torch") or not known["info"].get("transformers"):
            return self._err(400, "项目 Python 缺少 torch / transformers；请先在「环境」页安装训练依赖")
        args = [py, os.path.join(ROOT, "code", "http_server.py"),
                "--port", str(port), "--events", "--parent-pid", str(os.getpid()),
                "--ollama-url", ollama.base_url()]
        for name in local_targets:
            args += ["--config", name]
        for api_id in api_ids:
            args += ["--openai-api", api_id]

        def on_line(line):
            if line.startswith("@@EV@@ "):
                try:
                    ev = json.loads(line[7:])
                    if isinstance(ev, dict) and ev.get("t") == "embedding":
                        HUB.push("server-embedding", ev)
                    else:
                        HUB.push("server-chat", ev)
                except ValueError:
                    pass
            HUB.push("server-log", {"line": line})

        STATE["server"] = spawn(args, f"server-{port}.log", on_line=on_line)
        STATE["server_port"] = port
        self._json({"ok": True, "port": port})

    def _mcp_test(self, body):
        try:
            cfg = _normalize_mcp_server(body.get("server") or {})
            saved = find_mcp_server(cfg["id"])
            persistent = bool(saved and _mcp_signature(saved) == _mcp_signature(cfg))
            if persistent:
                tools = start_mcp_client(saved).list_tools()
            else:
                client = mcp_client.client_for(cfg)
                try:
                    tools = client.list_tools()
                finally:
                    client.close()
        except Exception as e:
            return self._err(502, f"MCP 连接失败：{type(e).__name__}: {e}")
        status = load_mcp_status()
        status[cfg["id"]] = {
            "ok": True,
            "tools": [{"name": t.get("name"), "description": t.get("description") or ""}
                      for t in tools if isinstance(t, dict) and t.get("name")],
            "error": "",
        }
        save_mcp_status(status)
        self._json({"ok": True, "server": cfg["name"], "tools": [
            {"name": t.get("name"), "description": t.get("description") or ""}
            for t in tools if isinstance(t, dict)
        ]})

    def _mcp_restart(self, body):
        sid = str(body.get("id") or "").strip()
        server = find_mcp_server(sid)
        if not server:
            return self._err(404, "MCP 服务器不存在")
        stop_mcp_client(sid)
        try:
            client = start_mcp_client(server)
            tools = client.list_tools()
        except Exception as e:
            mcp_update_status(sid, ok=False, state="failed",
                              attempts=MCP_RECONNECT_ATTEMPTS,
                              last_error=f"{type(e).__name__}: {e}",
                              error=f"{type(e).__name__}: {e}")
            return self._err(502, f"MCP 重启失败：{type(e).__name__}: {e}")
        mcp_update_status(sid, ok=True, state="connected", attempts=0,
                          last_error="", error="", tools=[
            {"name": t.get("name"), "description": t.get("description") or ""}
            for t in tools if isinstance(t, dict) and t.get("name")])
        self._json({"ok": True, "state": "connected", "tools": [
            {"name": t.get("name"), "description": t.get("description") or ""}
            for t in tools if isinstance(t, dict)
        ]})

    def _openai_test(self, body):
        try:
            cfg = openai_client.normalize_api(body.get("api") or {})
        except ValueError as e:
            return self._err(400, str(e))
        try:
            models = openai_client.list_model_infos(cfg)
            models = augment_ollama_models(cfg, models)
            detected = [dict(m) for m in models]
            models = apply_model_overrides(cfg, models)
        except Exception as e:
            return self._err(502, f"API 连接失败：{type(e).__name__}: {e}")
        saved = find_openai_api(cfg.get("id") or "")
        if saved:
            status = load_openai_status()
            status[cfg["id"]] = {"ok": True, "models": models,
                                 "detected": detected, "error": ""}
            save_openai_status(status)
        self._json({"ok": True, "name": cfg["name"], "models": models,
                    "saved": bool(saved)})

    def _openai_model_save(self, body):
        api_id = str(body.get("api_id") or "").strip()
        model_name = str(body.get("model") or "").strip()
        if not api_id or not model_name:
            return self._err(400, "缺少 api_id 或 model")
        if not find_openai_api(api_id):
            return self._err(404, "API 配置不存在")
        raw = body.get("override")
        if not isinstance(raw, dict):
            raw = {}
        ov = {}
        for key in ("context", "output"):
            if key in raw:
                if raw.get(key) in (None, ""):
                    continue
                n = _override_int(raw.get(key))
                if not n:
                    return self._err(400, f"{key} 必须是正整数")
                ov[key] = n
        if raw.get("tool_use") in ("native", "none", "unknown"):
            ov["tool_use"] = raw["tool_use"]
        modalities = raw.get("input_modalities")
        if isinstance(modalities, list):
            vals = [str(x) for x in modalities if str(x) in ("text", "image")]
            if vals:
                ov["input_modalities"] = list(dict.fromkeys(vals))
        efforts = raw.get("reasoning_efforts")
        if isinstance(efforts, dict):
            clean = {}
            for level, value in efforts.items():
                level = str(level or "").strip().lower()
                if not level:
                    continue
                if value is None or value == "":
                    clean[level] = None
                else:
                    clean[level] = str(value).strip()
            if clean:
                ov["reasoning_efforts"] = clean

        data = load_model_overrides()
        per_api = data.get(api_id)
        if not isinstance(per_api, dict):
            per_api = {}
        if ov:
            per_api[model_name] = ov
        else:
            per_api.pop(model_name, None)
        if per_api:
            data[api_id] = per_api
        else:
            data.pop(api_id, None)
        save_model_overrides(data)

        status = load_openai_status()
        entry = status.get(api_id) or {}
        base = entry.get("detected") or entry.get("models") or []
        models = apply_model_overrides({"id": api_id}, base)
        if entry:
            status[api_id] = dict(entry, models=models)
            save_openai_status(status)
        self._json({"ok": True, "models": models,
                    "overrides": data.get(api_id) or {}})

    def _agent_chat(self, body):
        model = str(body.get("model") or "").strip()
        server_id = str(body.get("server") or "").strip()
        api = None
        if model.startswith("api:"):
            api_id, upstream = parse_openai_model(model)
            if not api_id or not upstream:
                return self._err(400, "API 模型名不合法")
            api = find_openai_api(api_id)
            if not api:
                return self._err(400, "API 服务不存在，请先在 API 页保存")
            allowed = {m.get("name") for m in openai_models_for(api)
                       if m.get("tool_use") != "none"}
            if upstream not in allowed:
                return self._err(400, "这个 API 模型明确不支持工具调用")
            try:
                openai_client.api_key(api)
            except Exception as e:
                return self._err(400, str(e))
            tag = upstream
            backend = "openai"
            model_label = f"{api['name']}/{upstream}"
        else:
            if not ollama.is_remote(model):
                return self._err(400, "Agent 只支持 Ollama 模型或 API 服务")
            tag = ollama.model_tag(model)
            caps = set(ollama.model_capabilities(tag))
            if "tools" not in caps or not (caps & {"completion", "chat"}):
                return self._err(400, "这个 Ollama 模型不支持工具调用")
            backend = "ollama"
            model_label = model
        server = find_mcp_server(server_id) if server_id else None
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            text = str(body.get("text") or "").strip()
            if not text:
                return self._err(400, "消息不能为空")
            msgs = [{"role": "user", "content": text}]
        thinking = _thinking_level(body.get("thinking"))
        thinking_info = openai_model_thinking(api, upstream) if backend == "openai" else {}

        self._sse_start()
        client = None
        tools = []
        allowed_names = set()
        try:
            selected = body.get("tools")
            selected_names = None
            if isinstance(selected, list) and selected:
                selected_names = {str(x) for x in selected}
            if server and selected_names:
                client = ensure_mcp_client(server)
                all_tools = client.list_tools()
                all_tools = [t for t in all_tools
                             if isinstance(t, dict) and t.get("name") in selected_names]
                tools = mcp_client.openai_tools(all_tools)
                allowed_names = {t.get("function", {}).get("name")
                                 for t in tools if isinstance(t, dict)}
            session_id = _session_id(body.get("session_id")) or _session_id(body.get("id"))
            session = load_session(session_id) if session_id else None
            stored_history = (session.get("model_history")
                              if isinstance(session, dict) else None)
            if isinstance(stored_history, list) and stored_history:
                msgs = _strip_ui_messages(msgs)
            else:
                msgs = merge_session_history(
                    session_id, msgs, body.get("history_start"))
            reg = load_workspace_registry()
            wid = ""
            if session:
                # 会话一旦保存，工作区以会话归属为准，避免页面切换后把工具 cwd 带错。
                wid = str(session.get("workspace_id") or "")
            if not wid:
                wid = str(body.get("workspace_id") or "").strip()
            workspace = find_workspace(wid) or default_workspace(reg)
            if body.get("compact"):
                history = (session.get("model_history")
                           if isinstance(session, dict) else None)
                if not isinstance(history, list) or not history:
                    self._sse("error", {"error": "当前会话还没有可压缩的历史"})
                    return
                if backend != "openai":
                    self._sse("error", {"error": "手动压缩只支持 API 模型"})
                    return
                ctx = openai_model_context(api, upstream)
                out_budget = openai_output_budget(api, upstream)
                before = context_tokens(history, tools)
                compacted = compact_for_context(
                    api, upstream, history, ctx, out_budget, tools, force=True)
                after = context_tokens(compacted, tools)
                save_session_model_history(
                    session_id, compacted,
                    {"last_time_context": session.get("last_time_context")})
                self._sse("meta", {"model": model_label,
                                   "server": server["name"] if server else "",
                                   "tools": len(tools), "thinking": thinking,
                                   "backend": backend})
                self._sse("context", {"used": after, "total": ctx,
                                      "output": out_budget})
                self._sse("done", {"text": "", "compacted": {
                    "before": before, "after": after,
                    "removed": max(0, before - after)}})
                return
            agent_meta = {
                "dsh/sessionId": session_id or ("adhoc-" + uuid.uuid4().hex),
                "dsh/agentId": "dlqq-llm-agent",
                "dsh/workspace": workspace.get("path") or _default_workspace_dir(),
            }
            self._sse("meta", {"model": model_label,
                               "server": server["name"] if server else "",
                               "tools": len(tools), "thinking": thinking,
                               "backend": backend})
            last_user = None
            for item in reversed(msgs):
                if isinstance(item, dict) and item.get("role") == "user":
                    last_user = item
                    break
            system_prompt = (
                "你是带 MCP 工具调用的 Agent。只能调用 tools 列表中真实存在的工具名，"
                "不要编造工具名，也不要把工具调用写进普通文本。创建文件时优先使用文件"
                "写入工具；内容过长时拆成多次创建/追加，不要在一次工具调用里塞入过长内容。"
                if tools else
                "你是普通对话助手。当前没有启用任何 MCP 工具，不要生成工具调用，直接回答用户。")
            system_prompt += "\n\n" + _agent_tool_guidance(tools)
            model_history = stored_history
            workspace_path = workspace.get("path") or _default_workspace_dir()
            time_context_at = time.time()
            if isinstance(model_history, list) and model_history:
                messages = _strip_ui_messages(model_history)
                if messages and messages[0].get("role") == "system":
                    messages[0] = {"role": "system", "content": system_prompt}
                else:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                added_user = False
                if (last_user is not None and
                        (not messages or not _same_model_message(
                            messages[-1], last_user))):
                    messages.append(last_user)
                    added_user = True
                # 超过 10 分钟才追加新的时钟快照；追加在最后，不重写已有前缀。
                last_tc = 0.0
                if isinstance(session, dict):
                    try:
                        last_tc = float(session.get("last_time_context") or 0)
                    except (TypeError, ValueError):
                        last_tc = 0.0
                if added_user and time.time() - last_tc >= 600:
                    messages.append(_agent_time_message(workspace_path))
                    time_context_at = time.time()
                else:
                    time_context_at = last_tc
            else:
                messages = list(msgs)
                messages.insert(0, {"role": "system", "content": system_prompt})
                messages.insert(1, _agent_runtime_message(workspace_path))
            user_query = ""
            for item in reversed(msgs):
                if isinstance(item, dict) and item.get("role") == "user":
                    user_query = _message_content_text(item)
                    break
            turn_usage = None
            turn_usage_missing = False
            overflow_retries = 0
            while True:
                output_budget = (openai_output_budget(api, upstream)
                                 if backend == "openai" else 256000)
                payload = {"model": tag, "messages": messages,
                           "stream": True, "max_tokens": output_budget}
                if tools:
                    payload["tools"] = tools
                if backend == "openai":
                    compacted = compact_for_context(
                        api, upstream, messages,
                        openai_model_context(api, upstream),
                        output_budget, payload.get("tools"))
                    payload["messages"] = compacted
                    messages = compacted
                elif context_tokens(messages, payload.get("tools")) > max(
                        1024, (ollama.model_context(tag) or 1000000) - output_budget):
                    messages = _trim_for_context(
                        messages, ollama.model_context(tag) or 1000000,
                        output_budget)
                    payload["messages"] = messages
                context_total = (openai_model_context(api, upstream)
                                 if backend == "openai" else
                                 (ollama.model_context(tag) or 1000000))
                self._sse("context", {
                    "used": context_tokens(payload.get("messages"),
                                           payload.get("tools")),
                    "total": context_total,
                    "output": output_budget,
                })
                if backend == "openai":
                    payload.update(openai_thinking_payload(thinking_info, thinking))
                    try:
                        conn, resp = _open_chat_stream(
                            api, payload, timeout=openai_client.chat_timeout(api))
                    except ChatStreamError as e:
                        if (overflow_retries < 1 and
                                _is_context_overflow_text(e.body)):
                            overflow_retries += 1
                            before = context_tokens(messages, tools)
                            messages = compact_for_context(
                                api, upstream, messages,
                                openai_model_context(api, upstream),
                                output_budget, tools, force=True)
                            if context_tokens(messages, tools) < before:
                                continue
                        self._sse("error", {"error": f"API 返回错误：{e.body[:500]}"})
                        return
                else:
                    payload.update(ollama_thinking_payload(
                        ollama.model_thinking(tag), thinking))
                    conn, resp = ollama.open_chat(payload, timeout=600)
                if resp.status < 200 or resp.status >= 300:
                    raw = resp.read().decode("utf-8", errors="replace")
                    conn.close()
                    if (backend == "openai" and overflow_retries < 1 and
                            _is_context_overflow_text(raw)):
                        overflow_retries += 1
                        before = context_tokens(messages, tools)
                        messages = compact_for_context(
                            api, upstream, messages,
                            openai_model_context(api, upstream),
                            output_budget, tools, force=True)
                        if context_tokens(messages, tools) < before:
                            continue
                    self._sse("error", {"error": f"{backend} {resp.status}：{raw[:500]}"})
                    return
                parts = []
                calls = {}
                finish_reason = ""
                call_usage = None
                try:
                    for obj in ollama.iter_sse(resp):
                        if backend == "openai":
                            part = _usage_from_obj(obj)
                            if part:
                                call_usage = part
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        finish_reason = (choices[0] or {}).get("finish_reason") or finish_reason
                        delta = (choices[0] or {}).get("delta") or {}
                        reason = ollama.delta_reasoning(obj)
                        if reason:
                            self._sse("reasoning", {"text": reason})
                        piece = ollama.delta_text(obj)
                        if piece:
                            parts.append(piece)
                            self._sse("token", {"text": piece})
                        for tc in delta.get("tool_calls") or []:
                            idx = int(tc.get("index") or 0)
                            slot = calls.setdefault(idx, {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "_index": idx,
                            })
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                if not slot.get("_announced"):
                                    slot["_announced"] = True
                                    self._sse("tool_pending",
                                              {"name": fn["name"], "index": idx})
                                slot["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
                finally:
                    conn.close()
                if backend == "openai":
                    if call_usage:
                        turn_usage = _merge_usage(turn_usage, call_usage)
                    else:
                        turn_usage_missing = True
                call_list = [calls[k] for k in sorted(calls)]
                indexes = []
                for i, call in enumerate(call_list):
                    indexes.append(call.pop("_index", i))
                    call.pop("_announced", None)
                    if not call.get("id"):
                        call["id"] = f"call_{i}"
                if finish_reason == "length":
                    self._sse("error", {"error":
                        f"模型输出达到单轮 max_tokens={output_budget} 的上限（不是上下文上限）；"
                        "回答或工具参数可能被截断。请减少工具数量，或让模型把内容拆成多次创建/追加。"})
                    return
                if call_list:
                    messages.append({"role": "assistant",
                                     "content": "".join(parts),
                                     "tool_calls": call_list})
                    round_results = []
                    for i, call in enumerate(call_list):
                        fn = call.get("function") or {}
                        name = str(fn.get("name") or "")
                        raw_args = fn.get("arguments") or "{}"
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            if not isinstance(args, dict):
                                args = {}
                        except ValueError:
                            args = {}
                        self._sse("tool", {"name": name, "arguments": args,
                                           "index": indexes[i]})
                        if name not in allowed_names or client is None:
                            text = f"工具不存在：{name}"
                            is_error = True
                        else:
                            try:
                                entry_now = mcp_entry(server["id"]) if server else None
                                if (not client.alive() or
                                        (entry_now and entry_now.get("state") in
                                         ("reconnecting", "failed"))):
                                    client = ensure_mcp_client(server)
                                result = client.call_tool(name, args, meta=agent_meta)
                                text = mcp_client.tool_result_text(result)
                                is_error = bool(result.get("isError")) if isinstance(result, dict) else False
                            except Exception as e:
                                text = f"MCP 工具调用失败：{type(e).__name__}: {e}"
                                is_error = True
                                if server and ((client is None or not client.alive()) or any(
                                        word in str(e) for word in
                                        ("超时", "未运行", "无法连接", "断开"))):
                                    mcp_mark_unhealthy(server.get("id"), text)
                        self._sse("tool_result", {"name": name, "text": text,
                                                   "index": indexes[i],
                                                   "is_error": is_error})
                        messages.append({"role": "tool",
                                         "tool_call_id": call.get("id") or f"call_{i}",
                                         "content": text})
                        round_results.append((name, text))
                    # Ollama/Qwen 在“纯 tool 结果”轮次里会丢失最新 user；
                    # 把原始请求和工具结果再次放进最新的 user，避免上下文截断后
                    # 模型只看到一句“请继续”而看不到工具结果。
                    result_text = "\n\n".join(
                        f"【{name}】\n{text}" for name, text in round_results)
                    messages.append({"role": "user", "content":
                        f"用户请求：{user_query}\n\n"
                        f"工具执行结果：\n{result_text}\n\n"
                        "请根据以上工具结果继续完成用户请求。"})
                    continue
                text = "".join(parts)
                done = {"text": text}
                usage = _usage_event(turn_usage) if backend == "openai" else None
                if usage:
                    if turn_usage_missing:
                        usage["cache_hit"] = None
                    done["usage"] = usage
                    totals = update_session_usage(session_id, turn_usage, "agent")
                    if totals:
                        done["session_usage"] = totals
                if session_id:
                    history_messages = list(messages)
                    if text:
                        history_messages.append({"role": "assistant",
                                                 "content": text})
                    save_session_model_history(
                        session_id, history_messages,
                        {"last_time_context": time_context_at})
                self._sse("done", done)
                return
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            try:
                self._sse("error", {"error": f"{type(e).__name__}: {e}"})
            except OSError:
                pass

    def _sign(self, body):
        mode = "verify" if (body.get("mode") == "verify") else "compute"
        args = [sys.executable, os.path.join(ROOT, "code", "model_signature.py"),
                "--mode", mode, "--parent-pid", str(os.getpid())]
        one = (body.get("file") or "").strip()
        if one:
            args += ["--file", one]
        try:
            r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=300)
            self._json({"ok": r.returncode == 0, "mode": mode,
                        "out": (r.stdout or "")[-8000:], "err": (r.stderr or "")[-4000:]})
        except subprocess.TimeoutExpired:
            self._err(500, "签名任务超时")

    def _events(self):
        q = HUB.add()
        self._sse_start()
        try:
            self._sse("hello", {"ok": True})
            while True:
                try:
                    event, data = q.get(timeout=15)
                    self._sse(event, data)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.drop(q)


class LANHandler(Handler):
    """局域网只读对话页：只放行页面、模型清单和对话接口。

    训练 / 配置 / 签名 / 服务管理 / SSE 一概 403，避免把控制台暴露到内网。
    """

    server_version = "dlqq-webui-lan/1.0"
    GET_WHITELIST = ("/", "/chat.html", "/logo.ico", "/api/models", "/api/chat/configs")
    POST_WHITELIST = ("/api/chat",)

    def _auth(self, qs):
        return True                              # 内网对话页不带 token

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/vendor/"):
            self._static(path)
            return
        if path not in self.GET_WHITELIST:
            self._err(403, "局域网只开放对话页")
            return
        if path in ("/", "/chat.html"):
            self._send_file(os.path.join(WEB_DIR, "chat.html"),
                            "text/html; charset=utf-8")
        elif path == "/logo.ico":
            self._send_file(LOGO_PATH,
                            "image/x-icon")
        elif path == "/api/chat/configs":
            self._json({"ok": True, "configs": list_chat_configs(include_api=False)})
        else:
            self._json({"ok": True, "models": list_models()})

    def do_POST(self):
        if urlparse(self.path).path not in self.POST_WHITELIST:
            self._err(403, "局域网只开放对话页")
            return
        body = self._body()
        allowed = {c["name"] for c in list_chat_configs(include_api=False)}
        if (body.get("config") or "").strip() not in allowed:
            self._err(400, "配置不在清单里")
            return
        try:
            self._chat(body, {}, lan=True)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._err(500, f"{type(e).__name__}: {e}")
            except OSError:
                pass

    def _send_file(self, fp, ctype):
        try:
            with open(fp, "rb") as f:
                data = f.read()
        except OSError:
            self._err(404, "文件不存在")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


CHAT_LOCK = threading.RLock()
CHAT_TIMEOUT = 600
CHAT_WORKER = {"key": "", "child": None, "queue": None}


def _stop_worker():
    """停掉对话子进程（换配置、进程死掉、退出时都走这里）。"""
    w = CHAT_WORKER
    child = w.get("child")
    if child:
        child.send({"op": "quit"})
        child.stop()
    w.update(key="", child=None, queue=None)


def ensure_worker(key, py, config_name):
    """按配置保证有一个可用的对话子进程；调用方必须持有 CHAT_LOCK。"""
    w = CHAT_WORKER
    if w["child"] and w["child"].alive() and w["key"] == key:
        return w
    _stop_worker()
    q = queue.Queue()

    def on_line(line):
        try:
            q.put(json.loads(line))
        except ValueError:
            pass

    child = spawn([py, os.path.join(ROOT, "code", "chat_worker.py"),
                   "--config", config_name, "--parent-pid", str(os.getpid())],
                  f"chat-{config_name}.log", on_line=on_line, stdin_pipe=True)
    try:
        msg = q.get(timeout=120)
    except queue.Empty:
        child.stop()
        raise RuntimeError("对话进程启动超时（看 chat-*.log）")
    if msg.get("event") == "error":
        child.stop()
        raise RuntimeError(msg.get("error") or "对话进程启动失败")
    w.update(key=key, child=child, queue=q)
    return w


def detect_nvidia():
    """查询本机 NVIDIA 显卡和驱动；没有 nvidia-smi 时返回不可用。"""
    exe = "nvidia-smi"
    cmd = [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        line = (r.stdout or "").strip().splitlines()
        line = line[0].strip() if line else ""
        if not line:
            return {"nvidia": False, "name": "", "driver": "",
                    "error": (r.stderr or "nvidia-smi 无输出").strip()[-300:]}
        name, _, driver = line.rpartition(",")
        return {"nvidia": True, "name": name.strip(), "driver": driver.strip(),
                "error": ""}
    except (OSError, subprocess.SubprocessError) as e:
        return {"nvidia": False, "name": "", "driver": "",
                "error": f"{type(e).__name__}: {e}"}


def recommend_torch(gpu, info=None):
    """按显卡/驱动给出 torch 安装建议；版本可被 UI 文本框覆盖。"""
    info = info or {}
    installed = (info.get("torch_version") or "").split("+", 1)[0]
    base_version = installed or "2.14.1"

    def rec(index, reason):
        return {"index": index,
                "version": base_version,
                "install": f"{base_version}+{index}",
                "reason": reason}

    if not gpu.get("nvidia") and not info.get("cuda"):
        return rec("cpu", "未检测到 NVIDIA CUDA 显卡，推荐 CPU 版 torch")
    if info.get("cuda") and not gpu.get("nvidia"):
        return rec("cu126", "当前 torch 已检测到 CUDA，推荐 CUDA 12.6 版 torch")
    driver = (gpu.get("driver") or "").strip()
    m = re.match(r"(\d+)", driver)
    major = int(m.group(1)) if m else 0
    gpu_name = gpu.get("name") or info.get("gpu") or "NVIDIA GPU"
    if major >= 570:
        return rec("cu128", f"检测到 {gpu_name}（驱动 {driver}），推荐 CUDA 12.8 版 torch")
    if major >= 525:
        return rec("cu126", f"检测到 {gpu_name}（驱动 {driver}），推荐 CUDA 12.6 版 torch")
    return rec("cpu", f"检测到 {gpu_name}，但驱动 {driver or '未知'} 较旧；"
                      "建议更新驱动，或手动填写 CUDA 版本")


def latest_versions(info=None, gpu=None):
    """查询 PyPI / PyTorch 源，返回实际可选的 PyTorch 变体和最新版本。

    变体不是写死的 UI 分类：只有在该 PyTorch 源里确实查到 wheel 的变体才会返回。
    """
    info = info or {}
    gpu = gpu or {}
    cache = LATEST_CACHE.get("all")
    now = time.time()
    if cache and now - cache[0] < 1800:
        return cache[1]

    def get_json(url):
        req = urllib.request.Request(url, headers={"User-Agent": "dlqq-llm-webui/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def get_text(url):
        req = urllib.request.Request(url, headers={"User-Agent": "dlqq-llm-webui/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def pypi(name):
        data = get_json(f"https://pypi.org/pypi/{name}/json")
        return str((data.get("info") or {}).get("version") or "")

    def channel_label(channel):
        if channel == "cpu":
            return "CPU"
        if channel == "xpu":
            return "XPU"
        if re.fullmatch(r"cu\d+", channel):
            digits = channel[2:]
            return f"GPU / CUDA {digits[:-1]}.{digits[-1]}"
        return channel

    def channel_cuda(channel):
        if re.fullmatch(r"cu\d+", channel):
            digits = channel[2:]
            return f"{digits[:-1]}.{digits[-1]}"
        return ""

    def torch_channel(channel):
        html = unquote(get_text(f"https://download.pytorch.org/whl/{channel}/torch/"))
        versions = {}
        for mm in re.finditer(r"torch-([0-9][0-9A-Za-z.]*)(?:\+([0-9A-Za-z]+))?-", html):
            base, tag = mm.group(1), mm.group(2) or channel
            versions[f"{base}+{tag}"] = (base, tuple(int(x) for x in re.findall(r"\d+", base)))
        if not versions:
            return "", ""
        full = max(versions, key=lambda v: versions[v][1])
        return versions[full][0], full

    result = {"ok": False, "torch": "", "torch_base": "", "torch_variant": "",
              "torch_cuda": "", "torch_variants": [], "numpy": "", "transformers": "",
              "source": "network", "error": ""}
    errors = []

    # 已知 PyTorch wheel 源；只有真正返回 wheel 的源才会进入下拉框。
    channels = ["cpu", "cu126", "cu128", "cu130", "xpu"]
    torch_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            "numpy": pool.submit(pypi, "numpy"),
            "transformers": pool.submit(pypi, "transformers"),
        }
        for ch in channels:
            futures["torch:" + ch] = pool.submit(torch_channel, ch)
        for key, fut in futures.items():
            try:
                value = fut.result()
            except (OSError, ValueError, urllib.error.URLError) as e:
                errors.append(f"{key}: {type(e).__name__}")
                continue
            if key == "numpy":
                result["numpy"] = value
            elif key == "transformers":
                result["transformers"] = value
            else:
                channel = key.split(":", 1)[1]
                base, full = value
                if full:
                    torch_results[channel] = {"id": channel,
                                              "label": channel_label(channel),
                                              "cuda": channel_cuda(channel),
                                              "torch": base, "full": full}

    for ch in channels:
        if ch in torch_results:
            result["torch_variants"].append({k: v for k, v in torch_results[ch].items()
                                             if k != "full"})

    installed = (info.get("torch_version") or "").strip()
    m = re.search(r"\+(cpu|cu\d+|xpu)", installed)
    rec = recommend_torch(gpu, info)
    selected = m.group(1) if m else rec.get("index") or "cpu"
    if selected not in torch_results and torch_results:
        selected = next(iter(torch_results))
    if selected in torch_results:
        result["torch"] = torch_results[selected]["full"]
        result["torch_base"] = torch_results[selected]["torch"]
        result["torch_variant"] = selected
        result["torch_cuda"] = torch_results[selected]["cuda"]
    else:
        full = rec.get("install") or ""
        base = full.split("+", 1)[0] if full else ""
        result["torch"] = full
        result["torch_base"] = base
        result["torch_variant"] = selected
        result["torch_cuda"] = channel_cuda(selected)
        result["torch_variants"] = [{"id": selected,
                                     "label": channel_label(selected),
                                     "cuda": channel_cuda(selected),
                                     "torch": base}]
        result["source"] = "fallback"
    if not result["numpy"]:
        result["numpy"] = info.get("numpy_version") or ""
    if not result["transformers"]:
        result["transformers"] = info.get("transformers_version") or ""
    result["error"] = "；".join(errors)
    result["ok"] = bool(result["torch"] or result["numpy"] or result["transformers"])
    if result["ok"]:
        LATEST_CACHE["all"] = (now, result)
    return result


LATEST_CACHE = {}
PROBE_CACHE = {}


def probe_python(py, use_cache=True):
    """探测解释器：版本 / 位数 / torch / numpy / transformers / CUDA。

    结果按 (exe, mtime) 缓存 10 分钟；点“测试环境”时传 use_cache=False 拿实时结果。
    """
    try:
        key = (os.path.abspath(py), os.path.getmtime(py))
    except OSError:
        key = (py, 0)
    now = time.time()
    if use_cache:
        hit = PROBE_CACHE.get(key)
        if hit and now - hit[0] < 600:
            return hit[1]

    code = ("import json,sys,platform;"
            "import importlib.util;"
            "d={'version':sys.version.split()[0],'arch':platform.architecture()[0],"
            "'exe':sys.executable,"
            "'torch':importlib.util.find_spec('torch') is not None,"
            "'numpy':importlib.util.find_spec('numpy') is not None,"
            "'transformers':importlib.util.find_spec('transformers') is not None,"
            "'pip':importlib.util.find_spec('pip') is not None,"
            "'ensurepip':importlib.util.find_spec('ensurepip') is not None};"
            "print(json.dumps(d))")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    env.pop("PYTHONPATH", None)
    try:
        r = subprocess.run([py, "-s", "-P", "-c", code], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tried": [py]}
    lines = (r.stdout or "").strip().splitlines()
    if not lines:
        return {"ok": False, "error": (r.stderr or "探测没有输出").strip()[-500:],
                "tried": [py]}
    try:
        info = json.loads(lines[-1])
    except ValueError:
        return {"ok": False, "error": "解析探测输出失败", "tried": [py]}
    info["ok"] = True
    if info.get("torch"):
        try:
            r2 = subprocess.run(
                [py, "-s", "-P", "-c", "import torch;print(torch.__version__);"
                           "print(int(torch.cuda.is_available()));"
                           "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, env=env)
            lines2 = (r2.stdout or "").strip().splitlines()
            info["torch_version"] = lines2[0] if lines2 else ""
            info["cuda"] = bool(int(lines2[1])) if len(lines2) > 1 else False
            info["gpu"] = lines2[2] if len(lines2) > 2 else ""
        except (OSError, ValueError, subprocess.TimeoutExpired):
            info["torch_version"] = ""
            info["cuda"], info["gpu"] = False, ""
    else:
        info["torch_version"] = ""
        info["cuda"], info["gpu"] = False, ""
    info["numpy_version"] = ""
    info["transformers_version"] = ""
    try:
        r3 = subprocess.run(
            [py, "-s", "-P", "-c",
             "import importlib.metadata as m\n"
             "for n in ('numpy','transformers'):\n"
             "    try: print(n + '=' + m.version(n))\n"
             "    except Exception: print(n + '=')"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, env=env)
        for line in (r3.stdout or "").splitlines():
            if line.startswith("numpy="):
                info["numpy_version"] = line.split("=", 1)[1].strip()
            elif line.startswith("transformers="):
                info["transformers_version"] = line.split("=", 1)[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    if use_cache:
        PROBE_CACHE[key] = (now, info)
    return info


VERIFIED = {}


def _py_key(exe):
    return os.path.normcase(os.path.abspath(exe))


def verified_info(exe):
    return VERIFIED.get(_py_key(exe))


def remember_verified(exe, info):
    VERIFIED[_py_key(exe)] = {"exe": exe, "info": info, "ts": time.time()}


def port_in_use(port, host="127.0.0.1", timeout=0.3):
    """端口上已经有人在听？用来在启动服务前挡住重复实例。"""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()





def _lan_ipv4_from_adapters():
    """用 GetAdaptersInfo 找“像样的”局域网 IPv4：192.168.x.x、有网关、非虚拟网卡。"""
    if os.name != "nt":
        return ""
    from ctypes import wintypes

    class IP_ADDR_STRING(ctypes.Structure):
        pass

    IP_ADDR_STRING._fields_ = [
        ("Next", ctypes.POINTER(IP_ADDR_STRING)),
        ("IpAddress", ctypes.c_char * 16),
        ("IpMask", ctypes.c_char * 16),
        ("Context", wintypes.DWORD),
    ]

    class IP_ADAPTER_INFO(ctypes.Structure):
        pass

    IP_ADAPTER_INFO._fields_ = [
        ("Next", ctypes.POINTER(IP_ADAPTER_INFO)),
        ("ComboIndex", wintypes.DWORD),
        ("AdapterName", ctypes.c_char * 260),
        ("Description", ctypes.c_char * 132),
        ("AddressLength", wintypes.UINT),
        ("Address", ctypes.c_ubyte * 8),
        ("Index", wintypes.DWORD),
        ("Type", wintypes.UINT),
        ("DhcpEnabled", wintypes.UINT),
        ("CurrentIpAddress", ctypes.POINTER(IP_ADDR_STRING)),
        ("IpAddressList", IP_ADDR_STRING),
        ("GatewayList", IP_ADDR_STRING),
        ("DhcpServer", IP_ADDR_STRING),
        ("HaveWins", wintypes.BOOL),
        ("PrimaryWinsServer", IP_ADDR_STRING),
        ("SecondaryWinsServer", IP_ADDR_STRING),
        ("LeaseObtained", wintypes.DWORD),
        ("LeaseExpires", wintypes.DWORD),
    ]

    iphlpapi = ctypes.windll.iphlpapi
    size = wintypes.ULONG(0)
    iphlpapi.GetAdaptersInfo(None, ctypes.byref(size))      # 先问需要多大
    buf = ctypes.create_string_buffer(size.value)
    if iphlpapi.GetAdaptersInfo(buf, ctypes.byref(size)) != 0:
        return ""

    skip = ("vmware", "virtual", "hyper-v", "wsl", "loopback", "tap", "npcap", "vpn", "ppp")
    found = []
    node = ctypes.cast(buf, ctypes.POINTER(IP_ADAPTER_INFO))
    while node:
        a = node.contents
        desc = a.Description.decode("mbcs", "replace").lower()
        ip = a.IpAddressList.IpAddress.decode("ascii", "replace").rstrip("\x00")
        gw = a.GatewayList.IpAddress.decode("ascii", "replace").rstrip("\x00")
        if (ip.startswith("192.168.") and gw not in ("", "0.0.0.0")
                and not any(k in desc for k in skip)):
            found.append(ip)
        node = a.Next
    return found[0] if found else ""


def local_ip():
    """本机在局域网里的 IPv4。

    优先物理网卡的 192.168.x.x（虚拟网卡、VPN 会被跳过）；实在找不到
    才退回默认路由的出口地址。拿不到就返回 127.0.0.1。
    """
    try:
        ip = _lan_ipv4_from_adapters()
        if ip:
            return ip
    except (AttributeError, OSError, ValueError):
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # UDP connect 不发包，只为让系统选出口网卡
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()








class WebUIServer(ThreadingHTTPServer):
    """WebUI 主端口独占：端口被占用时不要静默复用。"""
    allow_reuse_address = False


def bind_port(host, want):
    """绑定 WebUI 端口；被占用时询问新端口，直接回车退出。"""
    while True:
        try:
            return WebUIServer((host, want), Handler), want
        except OSError as e:
            print(f"[错误] 端口 {want} 被占用：{e}")
            if not sys.stdin or not sys.stdin.isatty():
                print("[错误] 当前不是交互终端，无法询问新端口；"
                      "请用 --port 指定空闲端口，或关掉占用该端口的程序。")
                sys.exit(1)
            try:
                ans = input("       直接回车退出；或输入新的端口号：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[退出] 已取消启动。")
                sys.exit(1)
            if not ans:
                print("[退出] 已取消启动。")
                sys.exit(0)
            try:
                n = int(ans)
                if not (1 <= n <= 65535):
                    raise ValueError
            except ValueError:
                print(f"[错误] 端口号不合法：{ans}（应为 1–65535）")
                continue
            want = n


def main():
    global JOB_HANDLE
    ap = argparse.ArgumentParser(description="dlqq-llm 本地 WebUI")
    ap.add_argument("--port", type=int, default=0,
                    help="端口，默认 49134；被占用时会询问新端口")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认只绑本机")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--lan-port", type=int, default=49135,
                    help="局域网对话页端口，默认 49135")
    ap.add_argument("--lan-host", default="0.0.0.0",
                    help="局域网对话页监听地址，默认所有网卡")
    ap.add_argument("--no-lan", action="store_true", help="不启动局域网对话页")
    a = ap.parse_args()

    JOB_HANDLE = _make_job_object()
    STATE["token"] = secrets.token_urlsafe(12)
    want = a.port or 49134
    common.hide_dir(os.path.join(ROOT, ".ai_res"))
    ensure_cache_dir()
    os.makedirs(LOG_DIR, exist_ok=True)
    mcp_start_all()
    mcp_monitor_start()

    httpd, port = bind_port(a.host, want)
    url = f"http://{a.host}:{port}/?t={STATE['token']}"
    print("=" * 66)
    print("  dlqq-llm WebUI")
    print("=" * 66)
    print(f"[地址] {url}")

    lan = None
    if not a.no_lan:
        try:
            lan = ThreadingHTTPServer((a.lan_host, a.lan_port), LANHandler)
        except OSError as e:
            print(f"[警告] 局域网对话页没起来（{a.lan_host}:{a.lan_port}）：{e}")
        else:
            lan.daemon_threads = True
            threading.Thread(target=lan.serve_forever, daemon=True,
                             name="lan-chat").start()
            print(f"[内网] http://{local_ip()}:{a.lan_port}/   （只有对话页）")
            print("[提示] 首次可能需要在 Windows 防火墙里允许 Python 访问专用网络")

    print("[注意] 关掉本窗口 = 退出，所有子进程（训练/服务）一起结束")
    print("[提示] 关浏览器不影响运行，重新访问上面的地址即可")
    print("=" * 66, flush=True)

    httpd.daemon_threads = True
    if not a.no_browser:
        try:
            webbrowser.open(url)   # 先监听成功再开浏览器，避免打开时页面还没起来
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出] 收到 Ctrl+C")
    finally:
        stop_all()
        if lan:
            try:
                lan.server_close()
            except OSError:
                pass
        print("[退出] 子进程已清理")


if __name__ == "__main__":
    main()
