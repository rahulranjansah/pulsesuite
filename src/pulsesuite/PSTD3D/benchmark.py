"""
Benchmarking and run-summary for PulseSuite simulations.

Writes a ``run_summary.txt`` file inside the current working directory
(expected to be the timestamped run directory) containing:

* Wall-clock timing (total, init, time-loop)
* Hardware info (CPU, cores, RAM, GPU if present)
* Compute backend availability and dispatch chain
* Python / NumPy / Numba / CuPy version info
* Performance metrics (time per step, steps/s)
"""

import os
import platform
import sys
import time
from datetime import datetime


# ── Timer ────────────────────────────────────────────────────────────

_timers: dict = {}


def timer_start(label="total"):
    """Record the start time for *label*."""
    _timers[label] = {"start": time.perf_counter(), "end": None}


def timer_stop(label="total"):
    """Record the end time for *label*."""
    if label in _timers:
        _timers[label]["end"] = time.perf_counter()


def _elapsed(label):
    """Return elapsed seconds for *label*, or None."""
    t = _timers.get(label)
    if t and t["start"] is not None and t["end"] is not None:
        return t["end"] - t["start"]
    return None


# ── Per-function profiler ────────────────────────────────────────────

_profile_accum: dict = {}  # label -> {"total": float, "calls": int}


def profile_start():
    """Return a timestamp for use with :func:`profile_record`."""
    return time.perf_counter()


def profile_record(label, start_time):
    """Accumulate elapsed time since *start_time* under *label*."""
    elapsed = time.perf_counter() - start_time
    if label not in _profile_accum:
        _profile_accum[label] = {"total": 0.0, "calls": 0}
    _profile_accum[label]["total"] += elapsed
    _profile_accum[label]["calls"] += 1


def get_profile_data():
    """Return a copy of the accumulated profile data."""
    return dict(_profile_accum)


# ── Hardware detection ───────────────────────────────────────────────

def _cpu_info():
    """Return a short CPU description."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or "unknown"


def _gpu_info():
    """Try to detect GPU via nvidia-smi or CUDA."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, OSError):
        pass
    try:
        from numba import cuda
        if cuda.is_available():
            dev = cuda.get_current_device()
            return f"{dev.name} (compute {dev.compute_capability})"
    except Exception:
        pass
    return "none detected"


def _mem_info():
    """Return total RAM in GB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024 / 1024:.1f} GB"
    except (OSError, ValueError):
        pass
    return "unknown"


# ── Compute backend detection ────────────────────────────────────────

def _detect_backends():
    """
    Check which compute backends are installed and which the dispatch
    logic in SBEs.py / qwoptics.py will actually use.

    Returns a dict with availability flags, versions, and a plain-English
    description of the dispatch chain.
    """
    info = {
        "numba_installed": False,
        "numba_version": "not installed",
        "numba_jit_working": False,
        "cuda_available": False,
        "cupy_installed": False,
        "cupy_version": "not installed",
        "numpy_version": "unknown",
        "scipy_version": "unknown",
        "sbes_cuda": False,
        "qwoptics_cuda": False,
        "dispatch_chain": [],
        "active_backend": "unknown",
    }

    # NumPy
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    # SciPy
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    # Numba
    try:
        import numba
        info["numba_installed"] = True
        info["numba_version"] = numba.__version__
        # Verify JIT actually works (not just imported)
        try:
            from numba import jit as _jit
            @_jit(nopython=True)
            def _test_jit(x):
                return x + 1
            _test_jit(1)
            info["numba_jit_working"] = True
        except Exception:
            pass
    except ImportError:
        pass

    # CUDA via numba
    try:
        from numba import cuda
        if cuda.is_available():
            info["cuda_available"] = True
    except (ImportError, RuntimeError):
        pass

    # CuPy
    try:
        import cupy
        info["cupy_installed"] = True
        info["cupy_version"] = cupy.__version__
    except ImportError:
        pass

    # Check what SBEs.py and qwoptics.py detected at import time
    try:
        from pulsesuite.PSTD3D.SBEs import _HAS_CUDA as sbe_flag
        info["sbes_cuda"] = bool(sbe_flag)
    except (ImportError, AttributeError):
        pass

    try:
        from pulsesuite.PSTD3D.qwoptics import _HAS_CUDA as qw_flag
        info["qwoptics_cuda"] = bool(qw_flag)
    except (ImportError, AttributeError):
        pass

    # Determine the dispatch chain (matches the try/except logic in SBEs.py)
    # Pattern: if _HAS_CUDA -> try CUDA -> except -> try JIT -> except -> fallback
    #          else          -> try JIT -> except -> fallback
    chain = []
    if info["sbes_cuda"] or info["qwoptics_cuda"]:
        chain.append("CUDA (GPU)")
    if info["numba_jit_working"]:
        chain.append("Numba JIT (CPU parallel)")
    chain.append("NumPy fallback (CPU serial)")

    info["dispatch_chain"] = chain
    info["active_backend"] = chain[0] if chain else "unknown"

    return info


# ── Summary writer ───────────────────────────────────────────────────

def write_summary(filename="run_summary.txt", sim_params=None):
    """
    Write a human-readable run summary to *filename*.

    Parameters
    ----------
    filename : str
        Output file path (default: ``run_summary.txt`` in cwd).
    sim_params : dict, optional
        Simulation parameters to include (Nt, dt, Nr, etc.).
    """
    info = _detect_backends()

    lines = []
    lines.append("=" * 72)
    lines.append("  PULSESUITE RUN SUMMARY")
    lines.append("=" * 72)
    lines.append("")

    # Timestamp
    lines.append(f"Date/Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Working dir:     {os.getcwd()}")
    lines.append("")

    # ── Timing ──
    lines.append("--- Timing ---")
    for label in ["total", "init", "timeloop"]:
        elapsed = _elapsed(label)
        if elapsed is not None:
            mins, secs = divmod(elapsed, 60)
            hrs, mins = divmod(mins, 60)
            lines.append(
                f"  {label:12s}  {int(hrs):02d}:{int(mins):02d}:{secs:06.3f}"
                f"  ({elapsed:.3f} s)"
            )
    lines.append("")

    # ── Simulation params ──
    if sim_params:
        lines.append("--- Simulation Parameters ---")
        for k, v in sim_params.items():
            lines.append(f"  {k:16s}  {v}")
        lines.append("")

    # ── Hardware ──
    lines.append("--- Hardware ---")
    lines.append(f"  CPU:           {_cpu_info()}")
    lines.append(f"  Cores (os):    {os.cpu_count()}")
    lines.append(f"  RAM:           {_mem_info()}")
    lines.append(f"  GPU:           {_gpu_info()}")
    lines.append(f"  Platform:      {platform.platform()}")
    lines.append("")

    # ── Threading ──
    lines.append("--- Threading ---")
    numba_threads = os.environ.get("NUMBA_NUM_THREADS", "not set (defaults to all cores)")
    omp_threads = os.environ.get("OMP_NUM_THREADS", "not set")
    mkl_threads = os.environ.get("MKL_NUM_THREADS", "not set")
    lines.append(f"  NUMBA_NUM_THREADS:  {numba_threads}")
    lines.append(f"  OMP_NUM_THREADS:    {omp_threads}")
    lines.append(f"  MKL_NUM_THREADS:    {mkl_threads}")
    try:
        import numba
        lines.append(f"  Numba active threads:  {numba.config.NUMBA_NUM_THREADS}")
    except (ImportError, AttributeError):
        pass
    lines.append("")

    # ── Software versions ──
    lines.append("--- Software ---")
    lines.append(f"  Python:        {sys.version.split()[0]}")
    lines.append(f"  NumPy:         {info['numpy_version']}")
    lines.append(f"  SciPy:         {info['scipy_version']}")
    lines.append(f"  Numba:         {info['numba_version']}")
    lines.append(f"  CuPy:          {info['cupy_version']}")
    lines.append("")

    # ── Backend availability ──
    lines.append("--- Compute Backend Availability ---")
    lines.append(f"  Numba installed:     {'YES' if info['numba_installed'] else 'NO'}")
    lines.append(f"  Numba JIT working:   {'YES' if info['numba_jit_working'] else 'NO'}")
    lines.append(f"  CUDA available:      {'YES' if info['cuda_available'] else 'NO'}")
    lines.append(f"  CuPy installed:      {'YES' if info['cupy_installed'] else 'NO'}")
    lines.append(f"  SBEs CUDA flag:      {'YES' if info['sbes_cuda'] else 'NO'}")
    lines.append(f"  qwoptics CUDA flag:  {'YES' if info['qwoptics_cuda'] else 'NO'}")
    lines.append("")

    # ── Dispatch chain ──
    lines.append("--- Dispatch Chain (what actually runs) ---")
    lines.append(f"  Active backend:  {info['active_backend']}")
    lines.append("")
    lines.append("  Dispatch order (first success wins):")
    for i, backend in enumerate(info["dispatch_chain"], 1):
        marker = " <-- ACTIVE" if i == 1 else ""
        lines.append(f"    {i}. {backend}{marker}")
    lines.append("")

    if info["active_backend"].startswith("CUDA"):
        lines.append("  >> GPU is being used for heavy compute (dpdt, dCdt, dDdt, etc.)")
    elif info["active_backend"].startswith("Numba"):
        lines.append("  >> CPU parallel via Numba @jit(parallel=True, nopython=True)")
        lines.append("  >> Functions: dpdt, dCdt, dDdt, QWPolarization3, QWRho5, etc.")
    else:
        lines.append("  >> Plain NumPy loops - no JIT or GPU acceleration")
        lines.append("  >> Install numba for significant speedup: pip install numba")
    lines.append("")

    # ── Performance ──
    loop_time = _elapsed("timeloop")
    if sim_params and loop_time and "Nt" in sim_params:
        nt = int(sim_params["Nt"])
        if nt > 0:
            lines.append("--- Performance ---")
            lines.append(f"  Time per step:     {loop_time / nt * 1000:.3f} ms")
            lines.append(f"  Steps per second:  {nt / loop_time:.1f}")
            lines.append("")

    # ── Per-function profile breakdown ──
    pdata = get_profile_data()
    if pdata:
        total_time = _elapsed("total") or 1.0
        lines.append("--- Profile Breakdown ---")
        # Sort by total time, descending
        sorted_items = sorted(pdata.items(), key=lambda x: x[1]["total"],
                              reverse=True)
        profiled_total = sum(v["total"] for v in pdata.values())
        lines.append(f"  {'Function':<24s} {'Time (s)':>10s} {'%Total':>8s}"
                     f"  {'Calls':>8s}  {'Avg (ms)':>10s}")
        lines.append(f"  {'-'*24} {'-'*10} {'-'*8}  {'-'*8}  {'-'*10}")
        for label, vals in sorted_items:
            pct = 100.0 * vals["total"] / total_time
            avg_ms = vals["total"] / vals["calls"] * 1000 if vals["calls"] else 0
            lines.append(
                f"  {label:<24s} {vals['total']:>10.3f} {pct:>7.1f}%"
                f"  {vals['calls']:>8d}  {avg_ms:>10.3f}"
            )
        unaccounted = total_time - profiled_total
        if unaccounted > 0:
            pct = 100.0 * unaccounted / total_time
            lines.append(
                f"  {'(other/overhead)':<24s} {unaccounted:>10.3f} {pct:>7.1f}%"
            )
        lines.append("")

    lines.append("=" * 72)

    text = "\n".join(lines) + "\n"

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    # Also print to stdout
    print(text)

    return filename
