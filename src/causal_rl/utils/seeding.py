from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RandomStateSnapshot:
    python_state: str
    numpy_state: list[int]
    torch_seed: int


def set_seed(seed: int) -> RandomStateSnapshot:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy_state = [int(v) for v in np.random.randint(0, 2**31 - 1, size=16)]
    torch_seed = int(torch.initial_seed())
    py_state = repr(random.getstate())
    return RandomStateSnapshot(
        python_state=py_state[:128], numpy_state=numpy_state[:16], torch_seed=torch_seed
    )
