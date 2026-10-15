@echo off
setlocal
if not defined SOKURL_VSDEV set "SOKURL_VSDEV=T:\VisualStudio\Common7\Tools\VsDevCmd.bat"
if not defined SOKURL_CMAKE set "SOKURL_CMAKE=T:\VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
call "%SOKURL_VSDEV%" -arch=x86 -host_arch=x64
if errorlevel 1 exit /b 1
for %%I in ("%~dp0..") do set "PLATFORM=%%~fI"
"%SOKURL_CMAKE%" -S "%PLATFORM%\native\SokuRLBridge" -B "%PLATFORM%\build\bridge" -A Win32
if errorlevel 1 exit /b 1
"%SOKURL_CMAKE%" --build "%PLATFORM%\build\bridge" --config Release --parallel 4
exit /b %errorlevel%
