# -*- coding: utf-8 -*-
r"""给项目内置 Python 安装训练 / 对话依赖。

由 WebUI「环境」页启动，解释器就是 .runtime\python\python.exe 本身。
这个脚本只使用标准库：先确保 pip 可用，再执行 pip install。
"""
import argparse
import importlib.util
import re
import subprocess
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace",
                                     line_buffering=True)
    except (AttributeError, ValueError):
        pass

TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu130": "https://download.pytorch.org/whl/cu130",
    "xpu": "https://download.pytorch.org/whl/xpu",
}


def run(args):
    print("[环境] " + " ".join(args), flush=True)
    try:
        r = subprocess.run(args)
    except OSError as e:
        print(f"[错误] 启动失败：{e}", flush=True)
        return 1
    if r.returncode != 0:
        print(f"[错误] 命令退出码 {r.returncode}", flush=True)
    return r.returncode


def ensure_pip():
    if importlib.util.find_spec("pip") is not None:
        return 0
    print("[环境] 未找到 pip，使用 ensurepip 初始化", flush=True)
    if importlib.util.find_spec("ensurepip") is None:
        print("[错误] 项目 Python 缺少 ensurepip，请先运行 .runtime\\make_python_env.bat 重建",
              flush=True)
        return 1
    return run([sys.executable, "-m", "ensurepip", "--upgrade"])


def main():
    ap = argparse.ArgumentParser(description="安装训练和对话依赖")
    ap.add_argument("--torch-index", default="auto",
                    help="auto / cpu / cu126 / cu128 / cu130 / xpu，或自定义 PyTorch wheel 索引 URL")
    ap.add_argument("--torch-version", default="",
                    help="torch 版本，例如 2.14.1；留空 = 当前源最新版")
    ap.add_argument("--numpy-version", default="",
                    help="numpy 版本，例如 2.5.4；留空 = 最新版")
    ap.add_argument("--transformers-version", default="",
                    help="transformers 版本，例如 5.22.2；留空 = 最新版")
    ap.add_argument("--skip-torch", action="store_true",
                    help="跳过已安装的 torch")
    ap.add_argument("--skip-numpy", action="store_true",
                    help="跳过已安装的 numpy")
    ap.add_argument("--skip-transformers", action="store_true",
                    help="跳过已安装的 transformers")
    ap.add_argument("--no-cache-dir", action="store_true",
                    help="不保留 pip 下载缓存")
    a = ap.parse_args()

    torch_version = (a.torch_version or "").strip()
    numpy_version = (a.numpy_version or "").strip()
    transformers_version = (a.transformers_version or "").strip()
    for name, version in (("torch", torch_version),
                          ("numpy", numpy_version),
                          ("transformers", transformers_version)):
        if version and not re.fullmatch(r"[0-9][0-9A-Za-z.+!_-]*", version):
            print(f"[错误] {name} 版本不合法：{version}", flush=True)
            return 1
    torch_spec = f"torch=={torch_version}" if torch_version else "torch"
    numpy_spec = f"numpy=={numpy_version}" if numpy_version else "numpy"
    transformers_spec = (f"transformers=={transformers_version}"
                         if transformers_version else "transformers")

    rc = ensure_pip()
    if rc != 0:
        return rc

    pip = [sys.executable, "-m", "pip", "install", "--upgrade"]
    cache = ["--no-cache-dir"] if a.no_cache_dir else []

    index = (a.torch_index or "auto").strip()
    if index in TORCH_INDEX:
        index = TORCH_INDEX[index]

    if index and index != "auto":
        # PyTorch 必须先按指定源装，numpy / transformers 再走默认源
        if not a.skip_torch:
            rc = run(pip + cache + [torch_spec, "--index-url", index])
            if rc != 0:
                return rc
        deps = []
        if not a.skip_numpy:
            deps.append(numpy_spec)
        if not a.skip_transformers:
            deps.append(transformers_spec)
        if deps:
            rc = run(pip + cache + deps)
    else:
        packages = []
        if not a.skip_torch:
            packages.append(torch_spec)
        if not a.skip_numpy:
            packages.append(numpy_spec)
        if not a.skip_transformers:
            packages.append(transformers_spec)
        if packages:
            rc = run(pip + cache + packages)
        else:
            print("[环境] 全部依赖已安装，无需安装", flush=True)
            rc = 0

    return rc


if __name__ == "__main__":
    sys.exit(main())
