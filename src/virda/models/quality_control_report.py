from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityControlReport:
    report: dict[str, Any]
