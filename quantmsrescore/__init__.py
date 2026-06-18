# =============================================================================
# Thread Control Configuration (MUST be done before importing heavy libraries)
# =============================================================================
# This module controls thread allocation for numerical libraries to prevent
# thread explosion in HPC/Slurm environments where multiple processes compete
# for resources.
#
# Problem: Libraries like PyTorch, TensorFlow, NumPy (via MKL/OpenBLAS) spawn
# threads equal to CPU count by default. With multiprocessing, this causes:
#   - processes × cpu_count threads competing for cpu_count cores
#   - Memory pressure from thread stacks
#   - Node OOM kills in Slurm when co-located with other jobs
#
# Solution: Set environment variables BEFORE importing heavy libraries.
# =============================================================================

import os
from typing import Optional, Tuple

# Default threads per worker process
_DEFAULT_THREADS_PER_PROCESS = 1


def calculate_optimal_parallelism(
    total_cpus: int,
    memory_gb: Optional[float] = None,
    memory_per_process_gb: float = 4.0,
) -> Tuple[int, int]:
    """
    Calculate optimal number of processes and threads per process.

    This function determines the best split between process-level and
    thread-level parallelism for a given number of CPUs.

    Parameters
    ----------
    total_cpus : int
        Total number of CPUs available (e.g., from Nextflow's $task.cpus).
    memory_gb : float, optional
        Available memory in GB. If provided, limits processes based on memory.
    memory_per_process_gb : float, optional
        Estimated memory per process in GB. Default is 4 GB.

    Returns
    -------
    Tuple[int, int]
        (n_processes, threads_per_process) where:
        - n_processes: Number of worker processes to spawn
        - threads_per_process: Number of threads each process should use

    Notes
    -----
    Strategy:
    - For multiprocessing workloads (MS2PIP, feature calculation), we prefer
      more processes with 1 thread each (better CPU utilization, less GIL contention)
    - Each process gets 1 thread to avoid thread explosion
    - Memory constraints may reduce the number of processes

    Example
    -------
    >>> calculate_optimal_parallelism(8)
    (8, 1)  # 8 processes, 1 thread each

    >>> calculate_optimal_parallelism(8, memory_gb=16.0, memory_per_process_gb=4.0)
    (4, 1)  # Limited by memory: 16GB / 4GB = 4 processes
    """
    if total_cpus < 1:
        total_cpus = 1

    # Default: use all CPUs as separate processes, 1 thread each
    n_processes = total_cpus
    threads_per_process = 1

    # Apply memory constraint if provided
    if memory_gb is not None and memory_per_process_gb > 0:
        memory_limited_processes = max(1, int(memory_gb / memory_per_process_gb))
        n_processes = min(n_processes, memory_limited_processes)

    return n_processes, threads_per_process


def configure_threading(
    n_threads: Optional[int] = None, verbose: bool = False, disable_gpu: bool = False
) -> None:
    """
    Configure thread counts for all numerical/ML libraries.

    This function MUST be called before importing numpy, torch, tensorflow,
    or any library that depends on them. It sets environment variables that
    control internal threading in these libraries.

    Parameters
    ----------
    n_threads : int, optional
        Number of threads per process. Defaults to 1 for HPC safety.
        Set higher only if you're sure about available resources.
    verbose : bool, optional
        If True, log the thread configuration.
    disable_gpu : bool, optional
        If True, disable GPU/CUDA by setting CUDA_VISIBLE_DEVICES="".
        This prevents CUDA initialization errors on nodes without GPU support.
        Default is False.

    Notes
    -----
    For Nextflow/Slurm/HPC environments, the recommended approach is:
        - Pass $task.cpus as --processes
        - Let the tool use 1 thread per process (default)
        - Total parallelism = number of processes
    """
    if n_threads is None:
        n_threads = _DEFAULT_THREADS_PER_PROCESS

    n_threads_str = str(n_threads)

    # Disable GPU BEFORE any TensorFlow/CUDA imports to prevent init errors
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # OpenMP (used by many scientific libraries)
    os.environ["OMP_NUM_THREADS"] = n_threads_str

    # Intel MKL (NumPy/SciPy on Intel, PyTorch default backend)
    os.environ["MKL_NUM_THREADS"] = n_threads_str

    # NumExpr (used by pandas for some operations)
    os.environ["NUMEXPR_MAX_THREADS"] = n_threads_str

    # OpenBLAS (NumPy/SciPy alternative backend)
    os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str

    # Apple Accelerate (macOS)
    os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str

    # BLIS (another BLAS implementation)
    os.environ["BLIS_NUM_THREADS"] = n_threads_str

    # TensorFlow specific
    os.environ["TF_NUM_INTEROP_THREADS"] = n_threads_str
    os.environ["TF_NUM_INTRAOP_THREADS"] = n_threads_str

    # Disable TensorFlow GPU memory pre-allocation (helps with shared nodes)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Reduce TensorFlow verbosity to suppress CUDA errors
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    if verbose:
        gpu_status = "disabled" if disable_gpu else "enabled"
        print(f"[quantms-rescoring] Thread configuration: {n_threads} thread(s) per process, GPU {gpu_status}")


def configure_torch_threads(n_threads: Optional[int] = None) -> None:
    """
    Configure PyTorch-specific thread settings.

    This should be called after PyTorch is imported, as it uses PyTorch's
    API directly rather than environment variables.

    Parameters
    ----------
    n_threads : int, optional
        Number of threads for PyTorch operations. Defaults to 1.
    """
    if n_threads is None:
        n_threads = _DEFAULT_THREADS_PER_PROCESS

    try:
        import torch
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
    except ImportError:
        pass  # PyTorch not installed
    except RuntimeError:
        # Threads already set (can only be set once)
        pass


def get_safe_process_count(requested: int, memory_per_process_gb: float = 4.0) -> int:
    """
    Calculate a safe number of processes based on available resources.

    Parameters
    ----------
    requested : int
        Requested number of processes (e.g., from --processes or $task.cpus).
    memory_per_process_gb : float, optional
        Estimated memory per process in GB. Default is 4 GB.

    Returns
    -------
    int
        Safe number of processes (min of requested and calculated safe limit).
    """
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    # Try to get available memory
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        memory_based_limit = max(1, int(available_memory_gb / memory_per_process_gb))
    except ImportError:
        memory_based_limit = cpu_count  # Fall back to CPU count if psutil unavailable

    safe_count = min(requested, cpu_count, memory_based_limit)

    return max(1, safe_count)


# =============================================================================
# Opt-in thread configuration via environment variable
# =============================================================================
# Set QUANTMS_HPC_MODE=1 to automatically apply HPC-safe thread limits at import.
# This ensures that subsequent imports of numpy, torch, etc. use limited threads
# and disables GPU to prevent CUDA initialization errors on CPU-only nodes.
#
# For explicit control (recommended), call configure_threading() directly:
#   from quantmsrescore import configure_threading, configure_torch_threads
#   configure_threading(n_threads=1, disable_gpu=True)
#   configure_torch_threads(n_threads=1)
#
# Note: The CLI commands (ms2rescore, transfer_learning) always apply thread
# limits regardless of this setting.
if os.environ.get("QUANTMS_HPC_MODE", "").lower() in ("1", "true", "yes"):
    configure_threading(n_threads=_DEFAULT_THREADS_PER_PROCESS, disable_gpu=True)

# =============================================================================
# Standard module initialization
# =============================================================================
from warnings import filterwarnings

# Suppress warnings about OPENMS_DATA_PATH
filterwarnings("ignore", message=".*OPENMS_DATA_PATH.*", category=UserWarning)

__version__ = "0.0.20"

__all__ = [
    "configure_threading",
    "configure_torch_threads",
    "calculate_optimal_parallelism",
    "get_safe_process_count",
    "__version__",
]
