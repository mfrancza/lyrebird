@echo off
REM Lyrebird VST Build and Install Script
REM
REM Usage:
REM   build-and-install.bat           - Pull, build, and install to user VST3 folder
REM   build-and-install.bat system    - Pull, build, and install to system VST3 folder (requires Admin)
REM   build-and-install.bat nobuild   - Just install (skip pull and build)
REM

setlocal

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

REM Default VST3 install locations
set USER_VST3=%LOCALAPPDATA%\Programs\Common\VST3
set SYSTEM_VST3=C:\Program Files\Common Files\VST3

REM Determine install target
if "%1"=="system" (
    set VST3_DIR=%SYSTEM_VST3%
    echo Installing to system VST3 folder (requires Admin)
) else (
    set VST3_DIR=%USER_VST3%
    echo Installing to user VST3 folder
)

REM Skip build if requested
if "%1"=="nobuild" goto install

REM ===== PULL =====
echo.
echo === Pulling latest changes ===
where git >nul 2>&1
if errorlevel 1 (
    echo Warning: git not found, skipping pull
) else (
    git pull
    if errorlevel 1 (
        echo Warning: git pull failed, continuing with local files
    )
)

REM ===== BUILD =====
echo.
echo === Building Lyrebird VST ===

REM Clean old build cache if it exists from different environment
if exist build\CMakeCache.txt (
    findstr /C:"/workspace" build\CMakeCache.txt >nul 2>&1
    if not errorlevel 1 (
        echo Cleaning old CMake cache from Docker build...
        rmdir /s /q build
    )
)

REM Check for cl.exe in PATH
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
)

REM Build using Ninja if available, otherwise auto-detect VS
REM Disable JUCE's auto-install to avoid permission errors
where ninja >nul 2>&1
if not errorlevel 1 (
    echo Using Ninja generator...
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DJUCE_COPY_PLUGIN_AFTER_BUILD=OFF
    cmake --build build
) else (
    echo Using Visual Studio generator...
    cmake -B build -A x64 -DJUCE_COPY_PLUGIN_AFTER_BUILD=OFF
    cmake --build build --config Release
)

REM Check if VST3 was built (ignore JUCE's copy errors)
set VST3_CHECK=build\plugin\LyrebirdVST_artefacts\Release\VST3\Lyrebird.vst3
if not exist "%VST3_CHECK%" (
    set VST3_CHECK=build\plugin\LyrebirdVST_artefacts\VST3\Lyrebird.vst3
)
if not exist "%VST3_CHECK%" (
    echo.
    echo ERROR: Build failed - VST3 not found!
    exit /b 1
)
echo VST3 built successfully.

REM ===== INSTALL =====
:install
echo.
echo === Installing VST3 ===

REM Find the built VST3
set VST3_SOURCE=build\plugin\LyrebirdVST_artefacts\Release\VST3\Lyrebird.vst3
if not exist "%VST3_SOURCE%" (
    REM Try Ninja build location
    set VST3_SOURCE=build\plugin\LyrebirdVST_artefacts\VST3\Lyrebird.vst3
)
if not exist "%VST3_SOURCE%" (
    echo ERROR: Could not find built VST3 at expected location
    echo Expected: build\plugin\LyrebirdVST_artefacts\Release\VST3\Lyrebird.vst3
    exit /b 1
)

REM Create target directory if it doesn't exist
if not exist "%VST3_DIR%" (
    echo Creating VST3 directory: %VST3_DIR%
    mkdir "%VST3_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create VST3 directory. Try running as Administrator.
        exit /b 1
    )
)

REM Remove old version if it exists
if exist "%VST3_DIR%\Lyrebird.vst3" (
    echo Removing old version...
    rmdir /s /q "%VST3_DIR%\Lyrebird.vst3"
)

REM Copy new version
echo Copying VST3 to %VST3_DIR%...
xcopy /E /I /Y "%VST3_SOURCE%" "%VST3_DIR%\Lyrebird.vst3"
if errorlevel 1 (
    echo ERROR: Failed to copy VST3. Try running as Administrator for system install.
    exit /b 1
)

echo.
echo ========================================
echo  Lyrebird VST3 installed successfully!
echo ========================================
echo.
echo Location: %VST3_DIR%\Lyrebird.vst3
echo.
echo Next steps:
echo   1. Open your DAW (Reaper, etc.)
echo   2. Rescan VST3 plugins
echo   3. Add "Lyrebird" to a track
echo   4. Click "Load Model..." and select a .json model file
echo.

endlocal
