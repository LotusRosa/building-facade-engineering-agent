@echo off
cd /d "%~dp0"
if not defined FACADE_PYTHON set "FACADE_PYTHON=python"
"%FACADE_PYTHON%" -m facade_next serve --state runtime --port 8765 --open-browser
pause
