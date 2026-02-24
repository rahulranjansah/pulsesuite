"""
Timestamped run-directory management.

Creates a ``runs/<YYYY-MM-DD_HH-MM-SS_testname>/`` directory for every
simulation run and ``os.chdir`` into it so that all relative output paths
(``dataQW/``, ``fields/``, ``output/``) land inside that directory.

A ``runs/latest`` symlink always points to the most recent run.

Input files (``params/``, ``DC.txt``) are copied into the run directory
so the simulation can find them **and** so there is a permanent record
of which parameters produced each data set.
"""

import os
import shutil
from datetime import datetime


def setup_run_directory(input_files=None, test_name=None):
    """
    Create a timestamped run directory and chdir into it.

    Parameters
    ----------
    input_files : list of str, optional
        Relative paths (files or directories) to copy into the run
        directory.  Defaults to ``["params", "DC.txt"]``.
    test_name : str, optional
        Short label appended to the timestamp so runs are easy to
        distinguish, e.g. ``"sbetestprop"`` gives
        ``runs/2026-02-24_09-30-00_sbetestprop/``.

    Returns
    -------
    str
        Absolute path of the new run directory.
    """
    if input_files is None:
        input_files = ["params", "DC.txt"]

    orig_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = f"{timestamp}_{test_name}" if test_name else timestamp
    run_dir = os.path.join("runs", folder)
    os.makedirs(run_dir, exist_ok=True)

    # Update runs/latest symlink
    latest_link = os.path.join("runs", "latest")
    if os.path.islink(latest_link):
        os.remove(latest_link)
    elif os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(folder, latest_link)

    # Copy input files into the run directory
    for src_name in input_files:
        src_path = os.path.join(orig_dir, src_name)
        dst_path = os.path.join(run_dir, src_name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        elif os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    abs_run_dir = os.path.abspath(run_dir)
    print(f"Run directory: {abs_run_dir}")
    os.chdir(run_dir)

    return abs_run_dir
