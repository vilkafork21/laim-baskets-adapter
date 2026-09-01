"""SHA-256 файлов, канонического JSON и DataFrame.

Дайджест DataFrame — собственный простой алгоритм columns-values-sha256-v1:
имена и значения колонок хэшируются в фактическом порядке, значения — через
str() с пустой строкой вместо NaN/None. Полностью детерминирован и не зависит
от внутренностей pandas.
"""

import hashlib
import json
import math
from pathlib import Path

DATAFRAME_DIGEST_ALGORITHM = "columns-values-sha256-v1"

_SEP_NAME = b"\x00"
_SEP_VALUE = b"\x1f"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_json(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cell_repr(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def sha256_dataframe(df) -> str:
    digest = hashlib.sha256()
    for name in df.columns:
        digest.update(str(name).encode("utf-8") + _SEP_NAME)
        for value in df[name].tolist():
            digest.update(_cell_repr(value).encode("utf-8") + _SEP_VALUE)
    return digest.hexdigest()
