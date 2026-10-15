@echo off
call "T:\VisualStudio\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
cl /nologo /O2 /EHsc /LD HostBench.cpp /Fo:build\HostBench.obj /link /OUT:build\HostBench.dll /IMPLIB:build\HostBench.lib
exit /b %errorlevel%
