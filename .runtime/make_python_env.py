# -*- coding: utf-8 -*-
r"""把完整 CPython 安装裁剪成 WebUI 专用运行时 .runtime\python
================================================================
只保留标准库、ensurepip 和解释器本体；不带 site-packages（torch / numpy
一个都不进），也不带测试、文档、tkinter 和 Tcl/Tk。建好后由 .runtime\start.bat
用它跑 code\webui.py，这样用户机器上没装 Python 也能打开界面。
构建时会在 .runtime\python\licenses\ 内生成 Python / ensurepip 的许可证。
训练 / 对话依赖由 WebUI「环境」页通过 ensurepip + pip 安装到同一个项目
Python，不再使用用户自己的 Python。

用法（由同目录的 make_python_env.bat 调用；开发期工具，不进发布包）：
    python .runtime\make_python_env.py --src <Python安装目录> --out .runtime\python
    python .runtime\make_python_env.py --src <Python安装目录> --dry-run   # 只统计，不落盘

说明：
    · 源必须是完整 CPython 安装；给虚拟环境也行，会自动改用它的 base_prefix；
    · 目标已存在且非空时要 --force，或在终端里回答 y；
    · 只依赖标准库，任何 3.10+ 的 python 都能运行它。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace",
                                     line_buffering=True)
    except (AttributeError, ValueError):
        pass

# 工程根 = .runtime\ 的上一级；相对 --out 按它解析，不受当前工作目录影响
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 顶层只保留解释器本体和运行库；Doc / Scripts / include / tcl / 安装残留
# （比如 python_env.zip）一律不复制
ROOT_FILES = {"python.exe", "pythonw.exe", "python3.dll",
              "vcruntime140.dll", "vcruntime140_1.dll"}
RE_ROOT_DLL = re.compile(r"^python3\d+\.dll$", re.I)
RE_ROOT_ZIP = re.compile(r"^python3\d+\.zip$", re.I)

# Lib 下整目录排除（ensurepip 必须保留，WebUI 环境页用它安装训练依赖）
DIR_SKIP = {
    "Lib/site-packages", "Lib/test", "Lib/idlelib", "Lib/turtledemo",
    "Lib/tkinter",
}
# 文件级排除：Tcl/Tk 运行库
FILE_SKIP = {"_tkinter.pyd"}


def skip_file(low):
    """调试版、自由线程版、Tcl/Tk、pdb 全部不要。"""
    if low.endswith(".pdb") or low in FILE_SKIP:
        return True
    if "_d." in low:                       # *_d.pyd / *_d.dll / *_d.exe
        return True
    if re.search(r"\.cp\d+t", low):        # 自由线程版（cp314t-…）
        return True
    if "tcl" in low or re.match(r"^tk\d", low):   # tcl9tk90.dll / tk9tcl90.dll
        return True
    return False

# WebUI（webui.py + common.py + model_signature.py）实际用到的标准库
IMPORTS = ("argparse", "ctypes", "hashlib", "http.server", "json", "os", "queue",
           "re", "secrets", "shutil", "socket", "subprocess", "threading", "time",
           "traceback", "urllib.parse", "webbrowser", "ensurepip")


def run(py, code, timeout=120):
    return subprocess.run([py, "-c", code], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def keep_files(src):
    """产出要保留的文件：顶层白名单 + DLLs/ + Lib/（都过滤调试版与 tk）。"""
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        rel = "" if rel == "." else rel.replace(os.sep, "/")
        if rel == "":
            dirnames[:] = [d for d in dirnames if d in ("DLLs", "Lib")]
            for fn in sorted(filenames):
                low = fn.lower()
                if low in ROOT_FILES or RE_ROOT_DLL.match(low) or RE_ROOT_ZIP.match(low):
                    yield os.path.join(dirpath, fn)
            continue
        dirnames[:] = sorted(
            d for d in dirnames
            if d != "__pycache__"
            and (f"{rel}/{d}" if rel else d) not in DIR_SKIP)
        for fn in sorted(filenames):
            if skip_file(fn.lower()):
                continue
            yield os.path.join(dirpath, fn)


def remove_tree(path):
    """删除构建临时/旧运行时目录；遇到只读或占用文件时尽量重试。

    Windows 上复制出来的 DLL 有时会被残留句柄短暂占用，直接 rmtree 会让
    已经成功替换的新运行时被误判为构建失败；这里逐个文件清理，删不掉的
    只警告，不破坏已经就绪的新运行时。
    """
    if not os.path.isdir(path):
        return
    failed = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            fp = os.path.join(root, name)
            try:
                os.remove(fp)
            except OSError:
                try:
                    os.chmod(fp, 0o666)
                    os.remove(fp)
                except OSError as e:
                    failed.append(f"{fp}: {e}")
        for name in dirs:
            dp = os.path.join(root, name)
            try:
                os.rmdir(dp)
            except OSError as e:
                failed.append(f"{dp}: {e}")
    try:
        os.rmdir(path)
    except OSError as e:
        failed.append(f"{path}: {e}")
    if failed:
        print(f"[警告] 有 {len(failed)} 个旧文件暂时无法删除，请关闭占用后手动清理：{path}")


def collect_licenses(build, src):
    """把随包 Python 环境内的许可证文件一并放进新运行时。

    只处理 CPython 和 ensurepip 自带的 pip；torch / numpy / transformers
    等训练依赖由用户自行安装，不属于随包分发内容，不收集其许可证。
    仓库根目录的声明文件由人工维护，make 脚本不生成、不覆盖。
    """
    lic_dir = os.path.join(build, "licenses")
    os.makedirs(lic_dir, exist_ok=True)

    src_license = os.path.join(src, "LICENSE.txt")
    if os.path.isfile(src_license):
        shutil.copy2(src_license, os.path.join(build, "LICENSE.txt"))
        shutil.copy2(src_license, os.path.join(lic_dir, "PYTHON-LICENSE.txt"))

    # ensurepip 里的 pip wheel 也带许可证；从 wheel 内提取出来。
    bundled = os.path.join(build, "Lib", "ensurepip", "_bundled")
    if os.path.isdir(bundled):
        for wheel in sorted(os.listdir(bundled)):
            if not wheel.lower().endswith(".whl"):
                continue
            try:
                with zipfile.ZipFile(os.path.join(bundled, wheel)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        parts = info.filename.replace("\\", "/").split("/")
                        name = parts[-1]
                        parent = parts[-2].lower() if len(parts) > 1 else ""
                        low = name.lower()
                        if name.lower() in ("__init__.py", "_spdx.py"):
                            continue
                        if not (low.startswith(("license", "licence", "copying",
                                                "notice", "authors"))
                                or (parent in ("licenses", "license")
                                    and low.endswith((".txt", ".md", ".rst", ".terms")))):
                            continue
                        target = os.path.join(
                            lic_dir, f"{wheel.rsplit('.', 1)[0]}-{name}")
                        if not os.path.exists(target):
                            with open(target, "wb") as f:
                                f.write(zf.read(info))
            except (OSError, zipfile.BadZipFile, KeyError):
                continue

    count = len([n for n in os.listdir(lic_dir)
                 if os.path.isfile(os.path.join(lic_dir, n))])
    print(f"[许可] 已复制 {count} 个随包 Python 许可证文件到 {lic_dir}")


def main():
    ap = argparse.ArgumentParser(description="裁剪一份只带标准库的 WebUI 运行时")
    ap.add_argument("--src",
                    help="完整 CPython 安装目录（也可以直接给 python.exe）；不传则提示键盘输入")
    ap.add_argument("--out", default=".runtime/python",
                    help="输出目录，默认 .runtime/python")
    ap.add_argument("--force", action="store_true", help="目标已存在时直接覆盖")
    ap.add_argument("--autoyes", action="store_true",
                    help="自动回答 yes，供自动化调用；效果同 --force")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    a = ap.parse_args()

    src_arg = (a.src or "").strip()
    if not src_arg:
        try:
            src_arg = input("Python 安装目录（含 python.exe）: ").strip()
        except EOFError:
            src_arg = ""
        src_arg = src_arg.strip().strip('"').strip("'")
    if not src_arg:
        print("[错误] 没有提供 Python 安装目录")
        return 1

    src = os.path.abspath(src_arg)
    if os.path.isfile(src) and src.lower().endswith(".exe"):
        src = os.path.dirname(src)
    src_py = os.path.join(src, "python.exe")
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)

    print("=" * 64)
    print("  dlqq-llm WebUI 运行时（只含标准库）")
    print("=" * 64)
    print(f"[源] {src}")
    print(f"[输出] {out}")

    if not os.path.isfile(src_py):
        print(f"[错误] 这里没有 python.exe：{src}")
        return 1

    r = run(src_py, "import sys;print('%d.%d' % sys.version_info[:2]);print(sys.base_prefix)")
    if r.returncode != 0:
        print(f"[错误] 跑不了这个 python：{src_py}")
        print((r.stderr or "").strip()[-800:])
        return 1
    lines = (r.stdout or "").strip().splitlines()
    ver = lines[0] if lines else "?"
    base = lines[1] if len(lines) > 1 else src
    if tuple(int(x) for x in ver.split(".")) < (3, 10):
        print(f"[错误] 需要 Python 3.10+，这个是 {ver}")
        return 1
    if base and os.path.isdir(base) and os.path.abspath(base) != src:
        print(f"[提示] 这是虚拟环境，改用它的 base：{base}")
        src = os.path.abspath(base)
        src_py = os.path.join(src, "python.exe")

    files = list(keep_files(src))
    total = sum(os.path.getsize(p) for p in files)
    print(f"[计划] {len(files)} 个文件 / {total / 1048576:.1f} MB"
          "（已排除 site-packages、测试、tkinter、pip）")

    if a.dry_run:
        print("[dry-run] 没有写任何文件")
        return 0

    if os.path.isdir(out) and os.listdir(out):
        if not (a.force or a.autoyes):
            try:
                ans = input(f"[确认] {out} 已存在，覆盖？[y/N] ").strip().lower()
            except EOFError:
                ans = ""
            if ans not in ("y", "yes"):
                print("[取消] 什么都没改")
                return 1

    # 先构建到同级临时目录，成功后再替换；中途失败不会破坏现有运行时
    build = out + ".build"
    if os.path.isdir(build):
        remove_tree(build)
    os.makedirs(build, exist_ok=True)

    done = 0
    for fp in files:
        rel = os.path.relpath(fp, src)
        dst = os.path.join(build, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(fp, dst)
        done += 1
        if done % 200 == 0:
            print(f"  复制中 {done}/{len(files)} …")
    print(f"[复制] 完成 {done} 个文件")

    out_py = os.path.join(build, "python.exe")
    if not os.path.isfile(out_py):
        print("[错误] 复制完没有 python.exe，构建失败")
        return 1
    code = ("import sys;assert sys.version_info >= (3,10), 'needs 3.10+'\n"
            f"import {', '.join(IMPORTS)}\n"
            "print(sys.version.split()[0])")
    r = run(out_py, code)
    if r.returncode != 0:
        print("[错误] 新环境自检失败：")
        print((r.stderr or r.stdout or "").strip()[-1200:])
        return 1
    out_ver = (r.stdout or "").strip() or ver

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(build, "VERSION.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"Python {out_ver}\n")
        f.write(f"built {stamp}\n")
        f.write(f"files {done}, size {total / 1048576:.1f} MB\n")
        f.write(f"source {src}\n")

    # 随包分发的 Python 与 ensurepip 许可证，由 make 脚本复制；不收集用户装进
    # site-packages 的训练依赖。
    collect_licenses(build, src)

    # 原子替换：先把旧目录改名，再把新目录换上去；失败就恢复旧目录
    old = out + ".old"
    if os.path.isdir(old):
        remove_tree(old)
    if os.path.isdir(out):
        os.replace(out, old)
    try:
        os.replace(build, out)
    except Exception:
        if os.path.isdir(old):
            os.replace(old, out)
        raise
    if os.path.isdir(old):
        remove_tree(old)

    print()
    print("=" * 64)
    print(f"[完成] .runtime\\python 已就绪：Python {out_ver} / {done} 个文件 / "
          f"{total / 1048576:.1f} MB")
    print("       试跑：.runtime\\python\\python.exe code\\webui.py")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
