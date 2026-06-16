@echo off
setlocal

cd /d "%~dp0"

echo Starting VeriNova...
echo Local UI will be available at: http://127.0.0.1:8001
echo Press Ctrl+C to stop the server.
echo.

python web_server.py

if errorlevel 1 (
  echo.
  echo Python launcher failed. Trying the Windows Python launcher...
  echo.
  py web_server.py
)

endlocal
