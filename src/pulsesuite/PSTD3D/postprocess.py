"""
Post-processing for SBE simulation output data.

Splits the multi-column time-trace files (info.XX.t.dat, EP.XX.t.dat)
into individual per-quantity files in dataQW/Wire/info/ for easy plotting.

This is the Python equivalent of the legacy scripts/output.sh.

Author: auto-generated from output.sh
"""

import os
import sys

import numpy as np


def _parse_real(token):
    """Parse a numeric token, handling complex strings by taking the real part."""
    if "j" in token or "J" in token:
        return complex(token).real
    return float(token)


def _fast_loadtxt(filepath):
    """Load a whitespace-delimited numeric file much faster than np.loadtxt."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return np.empty((0, 0))
    rows = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append([_parse_real(x) for x in stripped.split()])
    if not rows:
        return np.empty((0, 0))
    return np.array(rows)


# Column definitions for info.XX.t.dat (0-indexed after time column)
_INFO_COLUMNS = {
    "vde":    1,   # electron drift velocity
    "vdh":    2,   # hole drift velocity
    "rhoe":   3,   # electron density
    "rhoh":   4,   # hole density
    "enge":   5,   # electron energy
    "engh":   6,   # hole energy
    "tempe":  7,   # electron temperature
    "temph":  8,   # hole temperature
    "rmaxe":  9,   # max |rho_eh| electron
    "rmaxh":  10,  # max |rho_eh| hole
    "EDrift": 11,  # drift field
    "momentum": 12,  # total momentum
    "pe":     13,  # electron momentum
    "ph":     14,  # hole momentum
    "I0":     15,  # spontaneous emission
    "I0e":    16,  # electron I0
    "I0h":    17,  # hole I0
}

# Column definitions for EP.XX.t.dat (0-indexed after time column)
_EP_COLUMNS = {
    "Ex.t":   1,   # electric field x
    "Ey.t":   2,   # electric field y
    "Ez.t":   3,   # electric field z
    "Px.t":   4,   # polarization x
    "Py.t":   5,   # polarization y
    "Pz.t":   6,   # polarization z
}


def organize_output(base_dir="dataQW", wire=1):
    """
    Split multi-column time-trace files into individual per-quantity files.

    Reads info.XX.t.dat and EP.XX.t.dat and writes each column as a
    separate file in {base_dir}/Wire/info/ for easy plotting.

    Parameters
    ----------
    base_dir : str
        Base output directory (default: 'dataQW')
    wire : int
        Wire index (default: 1)
    """
    wire_str = f"{wire:02d}"
    info_dir = os.path.join(base_dir, "Wire", "info")
    os.makedirs(info_dir, exist_ok=True)

    # --- Process info.XX.t.dat ---
    info_file = os.path.join(base_dir, f"info.{wire_str}.t.dat")
    if os.path.isfile(info_file):
        sys.stdout.write(f"  Reading info.{wire_str}.t.dat... ")
        sys.stdout.flush()
        data = _fast_loadtxt(info_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        time_col = data[:, 0]

        for name, col_idx in _INFO_COLUMNS.items():
            if col_idx < data.shape[1]:
                outpath = os.path.join(info_dir, f"{name}.dat")
                np.savetxt(outpath, np.column_stack([time_col, data[:, col_idx]]),
                           fmt="%.6e")
        print(f"done -> {len(_INFO_COLUMNS)} files in {info_dir}/")
    else:
        print(f"  WARNING: {info_file} not found, skipping info columns")

    # --- Process EP.XX.t.dat ---
    ep_file = os.path.join(base_dir, f"EP.{wire_str}.t.dat")
    if os.path.isfile(ep_file):
        sys.stdout.write(f"  Reading EP.{wire_str}.t.dat... ")
        sys.stdout.flush()
        data = _fast_loadtxt(ep_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        time_col = data[:, 0]

        for name, col_idx in _EP_COLUMNS.items():
            if col_idx < data.shape[1]:
                outpath = os.path.join(info_dir, f"{name}.dat")
                np.savetxt(outpath, np.column_stack([time_col, data[:, col_idx]]),
                           fmt="%.6e")
        print(f"done -> {len(_EP_COLUMNS)} files in {info_dir}/")
    else:
        print(f"  WARNING: {ep_file} not found, skipping EP columns")

    # --- Derived: total energy vs time ---
    info_file_path = os.path.join(base_dir, f"info.{wire_str}.t.dat")
    if os.path.isfile(info_file_path):
        data = _fast_loadtxt(info_file_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] > 6:
            time_col = data[:, 0]
            total_energy = 2 * data[:, 5] + 2 * data[:, 6]  # 2*Eng_e + 2*Eng_h
            outpath = os.path.join(info_dir, "En.t.dat")
            np.savetxt(outpath, np.column_stack([time_col, total_energy]),
                       fmt="%.6e")
            print(f"  -> derived En.t.dat (total energy)")


def organize_all(base_dir="dataQW", max_wires=10):
    """
    Run organize_output for all wire indices that have data files.

    Parameters
    ----------
    base_dir : str
        Base output directory (default: 'dataQW')
    max_wires : int
        Maximum wire index to check (default: 10)
    """
    print("Post-processing simulation output...")
    found = False
    for w in range(1, max_wires + 1):
        wire_str = f"{w:02d}"
        info_file = os.path.join(base_dir, f"info.{wire_str}.t.dat")
        ep_file = os.path.join(base_dir, f"EP.{wire_str}.t.dat")
        if os.path.isfile(info_file) or os.path.isfile(ep_file):
            print(f"Wire {w}:")
            organize_output(base_dir, wire=w)
            found = True
    if not found:
        print(f"  No data files found in {base_dir}/")
    else:
        print("Done.")


if __name__ == "__main__":
    organize_all()
