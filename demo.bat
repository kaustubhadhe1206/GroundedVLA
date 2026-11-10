@echo off
rem Interactive browser demo (groundedvla\demo\app.py) with all caches kept on D: (plan Section 2.5).
rem   .\demo [model] [options]     e.g.  .\demo smolvla_libero --task T1     then open http://127.0.0.1:8765
rem   .\demo --help                list options
setlocal
if "%GVLA_ROOT%"=="" set "GVLA_ROOT=D:\groundedvla"
set "HF_HOME=%GVLA_ROOT%\cache\huggingface"
set "HF_LEROBOT_HOME=%GVLA_ROOT%\cache\lerobot"
set "TORCH_HOME=%GVLA_ROOT%\cache\torch"
set "PIP_CACHE_DIR=%GVLA_ROOT%\cache\pip"
set "WANDB_DIR=%GVLA_ROOT%\outputs\wandb"
set "WANDB_CACHE_DIR=%GVLA_ROOT%\cache\wandb"
set "WANDB_CONFIG_DIR=%GVLA_ROOT%\cache\wandb_config"
set "LIBERO_CONFIG_PATH=%GVLA_ROOT%\cache\libero_config"
set "TEMP=%GVLA_ROOT%\cache\tmp"
set "TMP=%GVLA_ROOT%\cache\tmp"
set "PYTHONPYCACHEPREFIX=%GVLA_ROOT%\cache\pycache"
set "MUJOCO_GL=glfw"
if "%GVLA_PYTHON%"=="" set "GVLA_PYTHON=%GVLA_ROOT%\envs\gvla\python.exe"
if not exist "%GVLA_PYTHON%" (
  echo Python env not found: %GVLA_PYTHON%  ^(set GVLA_PYTHON or see README "Local setup"^)
  exit /b 1
)
cd /d "%~dp0"
"%GVLA_PYTHON%" -m groundedvla.demo.app %*
