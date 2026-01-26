@echo off
REM Lyrebird VST Windows Build Script
REM
REM Usage:
REM   build-windows.bat          - Build the plugin using Docker
REM   build-windows.bat native   - Build natively (requires VS2022 + CMake)
REM   build-windows.bat shell    - Open interactive Docker shell
REM

setlocal

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

set IMAGE_NAME=lyrebird-vst-windows

if "%1"=="native" goto native_build
if "%1"=="shell" goto shell
if "%1"=="clean" goto clean
goto docker_build

:docker_build
echo Building Docker image (this may take a while on first run)...
docker build -t %IMAGE_NAME% -f Dockerfile.windows .
if errorlevel 1 (
    echo.
    echo ERROR: Docker build failed.
    echo Make sure Docker Desktop is running in Windows container mode.
    echo Right-click Docker icon ^> "Switch to Windows containers"
    exit /b 1
)

echo.
echo Building Lyrebird VST plugin in Docker...
docker run --rm -v %SCRIPT_DIR%..:/workspace %IMAGE_NAME%
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo Build complete!
echo VST3 plugin is at: vst\build\plugin\LyrebirdVST_artefacts\Release\VST3\
goto end

:native_build
echo Building natively with Visual Studio...
echo.

REM Clean old build cache if it exists (may have wrong paths from Docker/WSL)
if exist build\CMakeCache.txt (
    echo Cleaning old CMake cache...
    rmdir /s /q build
)

REM Check for cl.exe in PATH (Developer Command Prompt sets this up)
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found. Please run from Developer Command Prompt for VS.
    echo Open "Developer Command Prompt" or "Developer PowerShell" from Start menu.
    exit /b 1
)

REM Update submodules if git is available
where git >nul 2>&1
if not errorlevel 1 (
    git submodule update --init --recursive
) else (
    echo Note: git not found, skipping submodule update
)

REM Use Ninja generator if available (faster), otherwise let CMake auto-detect VS version
where ninja >nul 2>&1
if not errorlevel 1 (
    echo Using Ninja generator...
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build
) else (
    echo Using Visual Studio generator...
    cmake -B build -A x64
    cmake --build build --config Release
)

echo.
echo Build complete!
echo VST3 plugin is at: vst\build\plugin\LyrebirdVST_artefacts\Release\VST3\
goto end

:shell
echo Opening interactive Docker shell...
docker run --rm -it -v %SCRIPT_DIR%..:/workspace %IMAGE_NAME% powershell
goto end

:clean
echo Cleaning build artifacts...
if exist build rmdir /s /q build
echo Done.
goto end

:end
endlocal
