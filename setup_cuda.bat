@echo off
REM TorchFlat CUDA setup helper for Windows
REM Run this before using TorchFlat to configure the CUDA kernel compilation.

echo TorchFlat CUDA Setup
echo ====================
echo.

REM Auto-detect CUDA toolkit
set "FOUND_CUDA="
for /d %%d in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
    if exist "%%d\bin\nvcc.exe" (
        set "FOUND_CUDA=%%d"
    )
)

if defined FOUND_CUDA (
    echo Found CUDA toolkit: %FOUND_CUDA%
    set "CUDA_HOME=%FOUND_CUDA%"
    set "CUDA_PATH=%FOUND_CUDA%"
) else (
    echo ERROR: No CUDA toolkit found.
    echo Install from: https://developer.nvidia.com/cuda-downloads
    echo Pick: Windows ^> x86_64 ^> exe (local^)
    exit /b 1
)

REM Detect GPU architecture
echo.
echo Detecting GPU...
python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Could not detect GPU. Defaulting to compute 7.5 (Turing).
    set "TORCH_CUDA_ARCH_LIST=7.5"
) else (
    for /f %%a in ('python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')"') do set "TORCH_CUDA_ARCH_LIST=%%a"
)
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%

REM Check for cl.exe (C++ compiler)
echo.
where cl >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo WARNING: cl.exe not found in PATH.
    echo Searching for Visual Studio...
    set "FOUND_VCVARS="
    for %%v in (2025 2022 2019) do (
        for %%e in (Community Professional Enterprise BuildTools) do (
            if exist "C:\Program Files\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvarsall.bat" (
                set "FOUND_VCVARS=C:\Program Files\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvarsall.bat"
            )
            if exist "C:\Program Files (x86)\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvarsall.bat" (
                set "FOUND_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvarsall.bat"
            )
        )
    )
    if defined FOUND_VCVARS (
        echo Found: %FOUND_VCVARS%
        echo Loading MSVC environment...
        call "%FOUND_VCVARS%" x64 >nul 2>nul
    ) else (
        echo ERROR: No Visual Studio installation found.
        echo Install Visual Studio Build Tools with C++ workload:
        echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
        exit /b 1
    )
) else (
    echo cl.exe found in PATH.
)

REM Test kernel compilation
echo.
echo Testing kernel compilation...
python -c "from torchflat._kernel_loader import _get_umi_kernel; k = _get_umi_kernel(); print('Kernel:', 'LOADED' if k else 'FAILED')"
if %ERRORLEVEL% neq 0 (
    echo.
    echo Kernel compilation failed. Check the error above.
    exit /b 1
)

echo.
echo Setup complete! Environment variables set for this session:
echo   CUDA_HOME=%CUDA_HOME%
echo   CUDA_PATH=%CUDA_PATH%
echo   TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
echo.
echo To make these permanent, add them to System Environment Variables.
