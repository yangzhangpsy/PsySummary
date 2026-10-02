@echo off
setlocal
cd /d "%~dp0"
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
    echo Inno Setup 6 was not found.
    exit /b 1
)
set "BUILD_DATE="
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set "BUILD_DATE=%%i"
if not defined BUILD_DATE exit /b 1
set "RELEASE_NAME=PsySummary%BUILD_DATE%Win"
if exist "..\%RELEASE_NAME%.exe" exit /b 1
if exist "..\%RELEASE_NAME%.zip" exit /b 1
set "PYINSTALLER_CONFIG_DIR=%~dp0build\pyinstaller-config"

echo Building PsySummary...
pyinstaller --clean --noconfirm PsySummary.spec
if errorlevel 1 exit /b 1
if not exist "dist\PsySummary\PsySummary.exe" exit /b 1

echo Creating installer...
"%ISCC%" "/F%RELEASE_NAME%" "%~dp0PsySummary.iss"
if errorlevel 1 exit /b 1
if not exist "..\%RELEASE_NAME%.exe" exit /b 1
pushd ".."
if errorlevel 1 exit /b 1
tar -caf "%RELEASE_NAME%.zip" "%RELEASE_NAME%.exe"
set "BUILD_RESULT=%errorlevel%"
popd
if not "%BUILD_RESULT%"=="0" exit /b %BUILD_RESULT%
echo Created: %~dp0..\%RELEASE_NAME%.zip
pause
