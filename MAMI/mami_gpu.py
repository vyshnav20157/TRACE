"""GPU pinning for the MAMI scripts.

CUDA_VISIBLE_DEVICES must be set BEFORE `torch` is imported anywhere in the process (once
CUDA is initialized, it's too late to change which devices are visible). So this module has
zero third-party imports and must be the very first import in every MAMI entrypoint script,
ahead of `import torch`.

Change MAMI_GPU_ID below to pick which of the machine's GPUs MAMI scripts are allowed to
use. All MAMI scripts (skeleton builder, cap-gen, the three training backbones, eval) import
this module first, so they are restricted to a single GPU regardless of what else is running
on the machine's other GPU(s).

Set MAMI_GPU_ID = None to leave CUDA_VISIBLE_DEVICES untouched (all GPUs visible).
"""

import os

MAMI_GPU_ID = 0

if MAMI_GPU_ID is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(MAMI_GPU_ID)
