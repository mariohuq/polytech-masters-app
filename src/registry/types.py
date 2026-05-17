from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SensorConfig:
    name: str
    low: float
    high: float

    def to_bounds_pair(self) -> tuple[float, float]:
        return (self.low, self.high)


@dataclass
class ModelMeta:
    name: str
    kind: str  # bounds | spikes | glitch
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    artifact_file: str = ""
    created_at: str = field(default_factory=_utc_now)
    trained_at: str | None = None
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegistryManifest:
    registry_id: str
    title: str = ""
    sensors: list[SensorConfig] = field(default_factory=list)
    models: list[ModelMeta] = field(default_factory=list)
    current_kind: str | None = None
    current_name: str | None = None
    created_at: str = field(default_factory=_utc_now)
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "title": self.title,
            "sensors": [asdict(s) for s in self.sensors],
            "models": [m.to_dict() for m in self.models if not m.deleted],
            "current": self.current_ref,
            "created_at": self.created_at,
            "deleted": self.deleted,
        }

    @property
    def current_ref(self) -> dict[str, str] | None:
        if self.current_kind and self.current_name:
            return {"kind": self.current_kind, "name": self.current_name}
        return None
