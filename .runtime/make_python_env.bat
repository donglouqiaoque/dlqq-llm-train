@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

echo ============================================================
echo  dlqq-llm - build .runtime\python (WebUI runtime only)
echo  Standard library + ensurepip: no torch, no numpy.
echo  Training dependencies are installed from the WebUI Environment tab.
echo ============================================================
echo.

if not exist ".runtime\make_python_env.py" (
  echo [ERROR] .runtime\make_python_env.py not found.
  goto fail
)

REM ---- parse optional nopause / autoyes (anywhere, case-insensitive) ----
set "NOPAUSE="
set "AUTOTYES="
set "EXTRA="
set "SRC="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="nopause"  ( set "NOPAUSE=1" & shift & goto parse_args )
if /I "%~1"=="-nopause" ( set "NOPAUSE=1" & shift & goto parse_args )
if /I "%~1"=="/nopause" ( set "NOPAUSE=1" & shift & goto parse_args )
if /I "%~1"=="autoyes"  ( set "AUTOTYES=1" & set "EXTRA=--autoyes" & shift & goto parse_args )
if /I "%~1"=="-autoyes" ( set "AUTOTYES=1" & set "EXTRA=--autoyes" & shift & goto parse_args )
if /I "%~1"=="/autoyes" ( set "AUTOTYES=1" & set "EXTRA=--autoyes" & shift & goto parse_args )
if not defined SRC set "SRC=%~1"
shift
goto parse_args
:args_done

if not defined SRC (
  set /p "SRC=Python install dir (the folder that contains python.exe): "
)
if not defined SRC (
  echo [ERROR] no path given.
  goto fail
)
set SRC=%SRC:"=%

if exist "%SRC%\python.exe" goto run
if exist "%SRC%" (
  for %%I in ("%SRC%") do set "SRC=%%~dpI"
)
if not exist "%SRC%\python.exe" (
  echo [ERROR] python.exe not found: %SRC%
  goto fail
)

:run
"%SRC%\python.exe" ".runtime\make_python_env.py" --src "%SRC%" --out ".runtime\python" %EXTRA%
if errorlevel 1 goto fail

echo.
echo [OK] .runtime\python is ready.
echo      Test it: .runtime\python\python.exe code\webui.py
if not defined NOPAUSE pause
exit /b 0

:fail
echo.
echo [FAILED] build did not finish; existing files were left as-is.
if not defined NOPAUSE pause
exit /b 1
