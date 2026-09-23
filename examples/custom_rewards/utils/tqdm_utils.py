from __future__ import annotations

from typing import Any, Iterable

import ray
from ray.experimental import tqdm_ray
from tqdm import tqdm


REMOTE_TQDM = ray.remote(tqdm_ray.tqdm)


def maybe_tqdm(iterable: Iterable[Any], use_tqdm: bool = True, **kwargs: Any) -> Iterable[Any]:
    return tqdm(iterable, **kwargs) if use_tqdm else iterable
