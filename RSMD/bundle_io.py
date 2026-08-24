from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to write parquet file: {path}. Install a parquet engine such as pyarrow."
        ) from exc


def write_sparse_npz(matrix: sp.spmatrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix.tocsr())


def ensure_bundle_dirs(bundle_root: Path) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "qc_tables").mkdir(parents=True, exist_ok=True)


def promote_bundle(tmp_bundle: Path, final_bundle: Path) -> None:
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    backup = final_bundle.parent / f"{final_bundle.name}.__bak__{int(time.time())}"

    if final_bundle.exists():
        if backup.exists():
            shutil.rmtree(backup)
        final_bundle.rename(backup)

    tmp_bundle.rename(final_bundle)

    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    arr = np.asarray(values)
    if arr.dtype.kind in {"U", "S", "O"}:
        payload = "\n".join(str(x) for x in arr.tolist()).encode("utf-8")
    else:
        payload = arr.tobytes()
    digest.update(payload)
    return digest.hexdigest()


def get_code_version(repo_root: Path, override: Optional[str]) -> str:
    if override:
        return override

    cmd = ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            commit = proc.stdout.strip()
            if commit:
                return commit
    except Exception:  # noqa: BLE001
        pass

    fallback = hashlib.sha1(f"{time.time()}-{os.getpid()}".encode("utf-8")).hexdigest()[:12]
    return f"local-{fallback}"
