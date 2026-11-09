@echo off
REM ===================================================================
REM  dlqq-llm WebUI launcher (the only entry point)
REM
REM  Double-click: starts the local web UI and opens the browser.
REM  Closing THIS window stops everything: training, the bridge server,
REM  and any signature job started from the page.
REM  Closing the browser tab alone does NOT stop anything.
REM
REM  Runtime: always the bundled .runtime\python\python.exe. There is no
REM  PATH fallback; if it is missing, rebuild it with make_python_env.bat.
REM
REM  Optional arguments:
REM     start.bat --port 8080     preferred port (asks again if busy)
REM     start.bat --no-browser    do not open the browser automatically
REM     start.bat --no-lan        do not start the LAN chat page
REM
REM  NOTE: keep this file pure ASCII -- non-ASCII breaks the cmd parser.
REM ===================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

REM ---- parse optional nopause (anywhere, case-insensitive) -------------
set "NOPAUSE="
set "ARGS="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="nopause"  ( set "NOPAUSE=1" & shift & goto parse_args )
if /I "%~1"=="-nopause" ( set "NOPAUSE=1" & shift & goto parse_args )
if /I "%~1"=="/nopause" ( set "NOPAUSE=1" & shift & goto parse_args )
set ARGS=%ARGS% "%~1"
shift
goto parse_args
:args_done

set "SCRIPT=code\webui.py"
if not exist "%SCRIPT%" (
  echo [ERROR] not found: %SCRIPT%
  if not defined NOPAUSE pause
  exit /b 1
)

REM ---- 1) bundled runtime only ------------------------------------------
set "PY=.runtime\python\python.exe"
set "PYFLAGS=-s -P"
if not exist "%PY%" (
  echo [ERROR] .runtime\python\python.exe not found.
  echo         Re-extract the full package, or rebuild it with:
  echo         .runtime\make_python_env.bat
  if not defined NOPAUSE pause
  exit /b 1
)
goto run

:run
echo [python] %PY%
echo [start ] launching the web UI ...
echo.

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
"%PY%" %PYFLAGS% "%SCRIPT%" %ARGS%

set "RC=%ERRORLEVEL%"
echo.
echo [exit code %RC%]
if not defined NOPAUSE pause
exit /b %RC%
