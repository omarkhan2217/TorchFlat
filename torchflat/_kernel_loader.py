"""JIT-compile and load UMI CUDA/HIP kernels.

Provides two kernels:
  - ``umi_median_mad``: O(n) median + MAD via quickselect (diagnostics)
  - ``umi_detrend_direct``: fused median + upper-RMS + asymmetric bisquare
    from raw [B, L] arrays (used by umi_detrend)

Kernels are compiled on first use and cached for subsequent imports.
Falls back gracefully if compilation fails (no GPU, no toolkit, etc.).
Set TORCHFLAT_NO_KERNEL=1 to disable kernels entirely.
"""

from __future__ import annotations

import ctypes
import glob as _glob
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger("torchflat")

_umi_kernel_module = None
_umi_kernel_load_attempted = False


# ---------------------------------------------------------------------------
# Auto-detection of CUDA / ROCm toolkit paths
# ---------------------------------------------------------------------------

def _auto_detect_cuda_home() -> str | None:
    """Find CUDA toolkit install path, checking env vars then common locations."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(var)
        if val and Path(val).exists():
            return val

    if sys.platform == "win32":
        base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if Path(base).exists():
            versions = sorted(Path(base).iterdir(), reverse=True)
            for v in versions:
                if (v / "bin" / "nvcc.exe").exists():
                    return str(v)
    else:
        for candidate in ["/usr/local/cuda", "/opt/cuda"]:
            if Path(candidate).exists() and (Path(candidate) / "bin" / "nvcc").exists():
                return candidate
        # Check versioned paths like /usr/local/cuda-12.8
        for pattern in ["/usr/local/cuda-*"]:
            matches = sorted(_glob.glob(pattern), reverse=True)
            for m in matches:
                if (Path(m) / "bin" / "nvcc").exists():
                    return m
    return None


def _auto_detect_rocm_home() -> str | None:
    """Find ROCm install path, checking env vars then common locations."""
    for var in ("ROCM_HOME", "ROCM_PATH"):
        val = os.environ.get(var)
        if val and Path(val).exists():
            return val

    for candidate in ["/opt/rocm"]:
        if Path(candidate).exists():
            return candidate
    return None


def _check_compiler_available() -> tuple[bool, str]:
    """Check if a C++ compiler is available. Returns (available, message)."""
    if sys.platform == "win32":
        if shutil.which("cl") or os.environ.get("VSINSTALLDIR"):
            return True, "cl.exe found"
        return False, (
            "cl.exe not found. Install Visual Studio Build Tools:\n"
            "  https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            "  Or run from a 'Developer Command Prompt for VS'."
        )
    else:
        for cc in ("g++", "c++", "clang++"):
            if shutil.which(cc):
                return True, f"{cc} found"
        return False, (
            "No C++ compiler found. Install build tools:\n"
            "  Ubuntu/Debian: sudo apt install build-essential\n"
            "  Fedora: sudo dnf install gcc-c++\n"
            "  macOS: xcode-select --install"
        )


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

    # Ensure ROCm DLLs are findable before torch.cuda.is_available() check
    _add_rocm_dll_dirs()

    import torch

    if not torch.cuda.is_available():
        return None
    if os.environ.get("TORCHFLAT_NO_KERNEL", "0") == "1":
        logger.warning(
            "UMI kernel disabled by TORCHFLAT_NO_KERNEL=1. "
            "Using torch.sort fallback (20x slower, 44x more VRAM). "
            "Unset the variable to enable the kernel."
        )
        return None

    csrc_dir = Path(__file__).parent / "csrc"
    build_dir = csrc_dir / "build"
    pyd_name = "torchflat_umi_ext"
    ext = ".pyd" if sys.platform == "win32" else ".so"
    pyd_path = build_dir / f"{pyd_name}{ext}"

    # Auto-detect and set CUDA_HOME if not set (needed by PyTorch JIT)
    is_hip = getattr(torch.version, "hip", None)
    if not is_hip and not os.environ.get("CUDA_HOME"):
        cuda_home = _auto_detect_cuda_home()
        if cuda_home:
            os.environ["CUDA_HOME"] = cuda_home
            logger.info("Auto-detected CUDA_HOME: %s", cuda_home)
    if is_hip and not os.environ.get("ROCM_HOME"):
        rocm_home = _auto_detect_rocm_home()
        if rocm_home:
            os.environ["ROCM_HOME"] = rocm_home
            logger.info("Auto-detected ROCM_HOME: %s", rocm_home)

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

    # Pre-flight checks before attempting compilation
    if is_hip:
        # HIP uses amdclang++ from ROCm SDK, not system compiler
        rocm_sdk = _find_rocm72_sdk()
        if rocm_sdk is None:
            logger.warning(
                "UMI kernel compilation skipped: ROCm SDK not found.\n"
                "Install with: pip install rocm-sdk-core rocm-sdk-devel\n"
                "Using torch.sort fallback (20x slower, 44x more VRAM)."
            )
            return None
    else:
        # CUDA needs system C++ compiler + nvcc
        compiler_ok, compiler_msg = _check_compiler_available()
        if not compiler_ok:
            logger.warning(
                "UMI kernel compilation skipped: %s\n"
                "Using torch.sort fallback (20x slower, 44x more VRAM).",
                compiler_msg,
            )
            return None

        nvcc = shutil.which("nvcc")
        cuda_home = os.environ.get("CUDA_HOME")
        if not nvcc and not cuda_home:
            logger.warning(
                "UMI kernel compilation skipped: nvcc not found and CUDA_HOME not set.\n"
                "Install the CUDA toolkit: https://developer.nvidia.com/cuda-downloads\n"
                "Then set: CUDA_HOME=/path/to/cuda (or it will be auto-detected).\n"
                "Using torch.sort fallback (20x slower, 44x more VRAM)."
            )
            return None

    try:
        if is_hip:
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
        _umi_kernel_module = None
        err = str(e)
        if "nvcc" in err.lower() or "No CUDA" in err:
            logger.warning(
                "UMI kernel compilation failed: nvcc error.\n"
                "  CUDA_HOME=%s\n"
                "  Ensure CUDA toolkit is installed and nvcc is in PATH.\n"
                "  Download: https://developer.nvidia.com/cuda-downloads\n"
                "  Using torch.sort fallback (20x slower).",
                os.environ.get("CUDA_HOME", "(not set)"),
            )
        elif "cl.exe" in err.lower() or "cl" in err.lower() and "not found" in err.lower():
            logger.warning(
                "UMI kernel compilation failed: C++ compiler not found.\n"
                "  Run from a 'Developer Command Prompt for VS' or install\n"
                "  Visual Studio Build Tools (C++ workload).\n"
                "  Using torch.sort fallback (20x slower).",
            )
        else:
            logger.warning(
                "UMI kernel compilation failed: %s\n"
                "  Using torch.sort fallback (20x slower, 44x more VRAM).",
                err[-300:],
            )

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
