@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-TriView2CAD.ps1" %*
if errorlevel 1 pause
