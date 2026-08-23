"""GPU pinning for the Memotion scripts.

CUDA_VISIBLE_DEVICES must be set BEFORE `torch` is imported anywhere in the process (once
CUDA is initialized, it's too late to change which devices are visible). So this module has
zero third-party imports and must be the very first import in every Memotion entrypoint
script, ahead of `import torch`.

Change MEMOTION_GPU_ID below to pick which of the machine's GPUs Memotion scripts are
allowed to use. All Memotion scripts (skeleton builder, cap-gen, the three training
backbones, eval) import this module first, so they are restricted to a single GPU
regardless of what else is running on the machine's other GPU(s).

Set MEMOTION_GPU_ID = None to leave CUDA_VISIBLE_DEVICES untouched (all GPUs visible).

Per-run override: an existing CUDA_VISIBLE_DEVICES in the environment always wins, so a
single run can be sent to another GPU without editing this file. This is what lets two
caption-generation tasks run concurrently on separate GPUs:

    CUDA_VISIBLE_DEVICES=0 python Memotion/memotion_cap_gen.py --prompt humour
    CUDA_VISIBLE_DEVICES=1 python Memotion/memotion_cap_gen.py --prompt offensive

`memotion_cap_gen.py` also accepts `--gpu N`, which sets the variable before this module is
imported and is equivalent to the above.
"""

import os

MEMOTION_GPU_ID = 1

if MEMOTION_GPU_ID is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(MEMOTION_GPU_ID)
