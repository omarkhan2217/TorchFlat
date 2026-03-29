"""JIT-compile and load UMI CUDA/HIP kernels.

Provides two kernels:
  - ``masked_median``: O(n) median via quickselect (legacy, used by rolling_clip)
  - ``umi_median_mad``: O(n) median + MAD in single call (used by umi_detrend)

Kernels are compiled on first use and cached for subsequent imports.
Falls back gracefully if compilation fails (no GPU, no toolkit, etc.).
Set TORCHFLAT_NO_KERNEL=1 to disable kernels entirely.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger("torchflat")

_umi_kernel_module = None
_umi_kernel_load_attempted = False


def _short_path(p: str) -> str:
    """Get Windows 8.3 short path (avoids spaces breaking compiler args)."""
    if sys.platform != "win32":
        return p
    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 260)
    return buf.value or str(p)


# ---------------------------------------------------------------------------
# UMI median+MAD kernel (used by umi_detrend)
# ---------------------------------------------------------------------------

def _get_umi_kernel():
    """Load the UMI median+MAD kernel. Returns None if unavailable."""
    global _umi_kernel_module, _umi_kernel_load_attempted

    if _umi_kernel_load_attempted:
        return _umi_kernel_module

    _umi_kernel_load_attempted = True

    import torch

    if not torch.cuda.is_available():
        return None
    if os.environ.get("TORCHFLAT_NO_KERNEL", "0") == "1":
        logger.warning(
            "UMI kernel disabled by TORCHFLAT_NO_KERNEL=1. "
            "Using torch.sort fallback (6x slower). "
            "Unset the variable to enable the kernel."
        )
        return None

    csrc_dir = Path(__file__).parent / "csrc"
    build_dir = csrc_dir / "build"
    pyd_name = "torchflat_umi_ext"
    pyd_path = build_dir / f"{pyd_name}.pyd"

    # Ensure ROCm DLLs are findable (amdhip64.dll etc.)
    _add_rocm_dll_dirs()

    if pyd_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(pyd_name, str(pyd_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            logger.info("Loaded cached UMI kernel from %s", pyd_path)
            _umi_kernel_module = mod
            return _umi_kernel_module
        except Exception as e:
            logger.warning("Failed to load cached UMI kernel: %s", e)
            pyd_path.unlink(missing_ok=True)

    try:
        if torch.version.hip:
            _umi_kernel_module = _compile_hip_rocm72(
                csrc_dir, pyd_name,
                csrc_dir / "build" / "umi_kernel_hip.cpp",
                csrc_dir / "umi_ext.cpp",
            )
        else:
            _umi_kernel_module = _compile_cuda_umi(csrc_dir)
        if _umi_kernel_module is not None:
            logger.info("UMI kernel compiled and loaded successfully")
    except Exception as e:
        logger.warning("Failed to compile UMI kernel: %s", e)
        _umi_kernel_module = None

    return _umi_kernel_module


# ---------------------------------------------------------------------------
# Compilation backends
# ---------------------------------------------------------------------------

def _compile_hip_rocm72(csrc_dir: Path, pyd_name: str, hip_src: Path, ext_src: Path):
    """Compile a HIP kernel on Windows using ROCm 7.2 SDK from pip."""
    import torch

    build_dir = csrc_dir / "build"
    build_dir.mkdir(exist_ok=True)

    pyd_path = build_dir / f"{pyd_name}.pyd"

    rocm_sdk_path = _find_rocm72_sdk()
    if rocm_sdk_path is None:
        raise RuntimeError(
            "ROCm 7.2 SDK not found. Install it with:\n"
            "  pip install https://repo.radeon.com/rocm/windows/rocm-rel-7.2/"
            "rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl\n"
            "  pip install https://repo.radeon.com/rocm/windows/rocm-rel-7.2/"
            "rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl"
        )

    clang = str(rocm_sdk_path / "lib" / "llvm" / "bin" / "amdclang++.exe")
    if not os.path.exists(clang):
        raise RuntimeError(f"amdclang++ not found at {clang}")

    if not hip_src.exists():
        raise RuntimeError(f"Hipified kernel source not found at {hip_src}")

    rocm_sp = _short_path(str(rocm_sdk_path))
    device_lib = _short_path(str(rocm_sdk_path / "lib" / "llvm" / "amdgcn" / "bitcode"))

    _setup_msvc_env()

    torch_dir = Path(torch.__file__).parent
    python_inc = _short_path(sysconfig.get_path("include"))
    python_lib = _short_path(str(Path(sysconfig.get_path("stdlib")).parent / "libs"))
    torch_lib = _short_path(str(torch_dir / "lib"))

    inc = [
        f"-I{_short_path(str(torch_dir / 'include'))}",
        f"-I{_short_path(str(torch_dir / 'include' / 'torch' / 'csrc' / 'api' / 'include'))}",
        f"-I{python_inc}",
        f"-I{rocm_sp}/include",
    ]
    defs = [
        "-D__HIP_PLATFORM_AMD__",
        f"-DTORCH_EXTENSION_NAME={pyd_name}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
    ]

    # Step 1: Compile kernel (.cpp as HIP)
    kernel_obj = build_dir / f"{pyd_name}_kernel.o"
    logger.info("Compiling %s kernel with amdclang++...", pyd_name)
    _run_cmd([
        clang, "-O3", "-c", "-x", "hip",
        f"--rocm-path={rocm_sp}",
        f"--rocm-device-lib-path={device_lib}",
        "--offload-arch=gfx1200",
        *defs, *inc, "-std=c++17", "-w",
        str(hip_src), "-o", str(kernel_obj),
    ])

    # Step 2: Compile binding (.cpp as C++)
    ext_obj = build_dir / f"{pyd_name}_ext.o"
    logger.info("Compiling %s binding...", pyd_name)
    _run_cmd([
        clang, "-O3", "-c",
        *defs, *inc, "-std=c++17", "-w",
        str(ext_src), "-o", str(ext_obj),
    ])

    # Step 3: Link into .pyd
    logger.info("Linking %s...", pyd_name)
    _run_cmd([
        clang, "-shared",
        str(kernel_obj), str(ext_obj),
        f"-L{torch_lib}", "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10", "-lc10_hip",
        f"-L{python_lib}", f"-lpython{sys.version_info.major}{sys.version_info.minor}",
        f"-L{rocm_sp}/lib", "-lamdhip64",
        "-o", str(pyd_path),
    ])

    spec = importlib.util.spec_from_file_location(pyd_name, str(pyd_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def _compile_cuda_umi(csrc_dir: Path):
    """Compile UMI median+MAD CUDA kernel."""
    import torch.utils.cpp_extension as _ext
    return _ext.load(
        name="torchflat_umi_ext",
        sources=[
            str(csrc_dir / "umi_ext.cpp"),
            str(csrc_dir / "umi_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _add_rocm_dll_dirs():
    """Add ROCm SDK lib directories to DLL search path (Windows).

    Required so the compiled .pyd can find amdhip64.dll at load time.
    """
    if sys.platform != "win32":
        return
    sdk = _find_rocm72_sdk()
    if sdk is None:
        return
    lib_dir = sdk / "lib"
    bin_dir = sdk / "bin"
    for d in [lib_dir, bin_dir]:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
            # Also add to PATH as fallback
            if str(d) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def _find_rocm72_sdk() -> Path | None:
    """Find the ROCm 7.2 SDK installed via pip (rocm-sdk-core package)."""
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "_rocm_sdk_core",
        Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Python"
        / f"Python{sys.version_info.major}{sys.version_info.minor}"
        / "site-packages" / "_rocm_sdk_core",
    ]
    try:
        import _rocm_sdk_core
        candidates.insert(0, Path(_rocm_sdk_core.__file__).parent)
    except ImportError:
        pass

    for p in candidates:
        clang = p / "lib" / "llvm" / "bin" / "amdclang++.exe"
        if clang.exists():
            return p
    return None


def _setup_msvc_env():
    """Setup MSVC compiler environment on Windows."""
    if sys.platform != "win32":
        return
    if os.environ.get("VSINSTALLDIR"):
        return

    vcvars_candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    for vcvars in vcvars_candidates:
        if os.path.exists(vcvars):
            result = subprocess.run(
                f'cmd /c ""{vcvars}" x64 && set"',
                capture_output=True, text=True, shell=True,
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    os.environ[k] = v
            return

    logger.warning("MSVC not found, linking may fail")


def _run_cmd(cmd: list[str]):
    """Run a command, raising RuntimeError on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (return {result.returncode}):\n"
            f"  {' '.join(cmd[:5])}...\n"
            f"  {result.stderr[-500:]}"
        )
