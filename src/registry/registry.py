from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from models import DetectorKind, UnifiedAnomalyDetector
from models.wrapper import TimeSeriesX


def _kind_str(kind: DetectorKind | str) -> str:
    return kind.value if isinstance(kind, DetectorKind) else DetectorKind(kind).value

from .types import ModelMeta, RegistryManifest, SensorConfig, _utc_now


class RegistryError(Exception):
    pass


class ModelRegistry:
    """Один экземпляр registry (каталог на диске)."""

    MANIFEST = "manifest.json"
    MODELS_DIR = "models"

    def __init__(self, root: Path, registry_id: str):
        self.root = Path(root)
        self.registry_id = registry_id
        self.path = self.root / registry_id
        self._models_path = self.path / self.MODELS_DIR

    @classmethod
    def create(
        cls,
        root: Path,
        sensors: list[SensorConfig],
        *,
        registry_id: str | None = None,
        title: str = "",
    ) -> ModelRegistry:
        if not sensors:
            raise RegistryError("Нужен хотя бы один датчик")
        names = [s.name for s in sensors]
        if len(names) != len(set(names)):
            raise RegistryError("Имена датчиков должны быть уникальными")

        rid = registry_id or str(uuid.uuid4())
        reg = cls(root, rid)
        if reg.path.exists():
            raise RegistryError(f"Registry {rid!r} уже существует")

        reg._models_path.mkdir(parents=True, exist_ok=True)
        manifest = RegistryManifest(registry_id=rid, title=title, sensors=sensors)
        reg._save_manifest(manifest)
        return reg

    @classmethod
    def open(cls, root: Path, registry_id: str) -> ModelRegistry:
        reg = cls(root, registry_id)
        if not reg._manifest_path.exists():
            raise RegistryError(f"Registry {registry_id!r} не найден")
        return reg

    @classmethod
    def list_registries(cls, root: Path) -> list[dict[str, Any]]:
        root = Path(root)
        if not root.exists():
            return []
        out: list[dict[str, Any]] = []
        for p in root.iterdir():
            if p.is_dir() and (p / cls.MANIFEST).exists():
                try:
                    m = cls.open(root, p.name).get_info()
                    if not m.get("deleted"):
                        out.append(m)
                except (RegistryError, json.JSONDecodeError):
                    continue
        return out

    @property
    def _manifest_path(self) -> Path:
        return self.path / self.MANIFEST

    def _load_manifest(self) -> RegistryManifest:
        raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        sensors = [SensorConfig(**s) for s in raw.get("sensors", [])]
        models = []
        for m in raw.get("models", []):
            models.append(
                ModelMeta(
                    name=m["name"],
                    kind=m["kind"],
                    hyperparameters=m.get("hyperparameters", {}),
                    model_id=m.get("model_id", ""),
                    artifact_file=m.get("artifact_file", ""),
                    created_at=m.get("created_at", _utc_now()),
                    trained_at=m.get("trained_at"),
                    deleted=m.get("deleted", False),
                )
            )
        return RegistryManifest(
            registry_id=raw["registry_id"],
            title=raw.get("title", ""),
            sensors=sensors,
            models=models,
            current_kind=raw.get("current_kind"),
            current_name=raw.get("current_name"),
            created_at=raw.get("created_at", _utc_now()),
            deleted=raw.get("deleted", False),
        )

    def _save_manifest(self, manifest: RegistryManifest) -> None:
        payload = {
            "registry_id": manifest.registry_id,
            "title": manifest.title,
            "sensors": [s.__dict__ for s in manifest.sensors],
            "models": [m.to_dict() for m in manifest.models],
            "current_kind": manifest.current_kind,
            "current_name": manifest.current_name,
            "created_at": manifest.created_at,
            "deleted": manifest.deleted,
        }
        self.path.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_info(self) -> dict[str, Any]:
        m = self._load_manifest()
        if m.deleted:
            raise RegistryError("Registry помечен удалённым")
        return m.to_dict()

    def mark_deleted(self) -> None:
        m = self._load_manifest()
        m.deleted = True
        self._save_manifest(m)

    def sensor_names(self) -> list[str]:
        return [s.name for s in self._load_manifest().sensors]

    def bounds_for_detector(self) -> dict[str, tuple[float, float]]:
        m = self._load_manifest()
        return {s.name: s.to_bounds_pair() for s in m.sensors}

    def _artifact_path(self, meta: ModelMeta) -> Path:
        return self._models_path / meta.artifact_file

    def _find_model(
        self, manifest: RegistryManifest, name: str, kind: str | None = None
    ) -> ModelMeta:
        for m in manifest.models:
            if m.deleted:
                continue
            if m.name != name:
                continue
            if kind is not None and m.kind != kind:
                continue
            return m
        raise RegistryError(f"Модель {name!r} не найдена")

    def add_model(
        self,
        name: str,
        kind: DetectorKind | str,
        hyperparameters: dict[str, Any] | None = None,
        *,
        fit_data: TimeSeriesX | None = None,
        set_current: bool = False,
    ) -> ModelMeta:
        """Новый файл модели: уникальное имя, тип (1), гиперпараметры для UI."""
        kind_str = _kind_str(kind)
        params = dict(hyperparameters or ())

        manifest = self._load_manifest()
        if manifest.deleted:
            raise RegistryError("Registry удалён")

        for m in manifest.models:
            if not m.deleted and m.name == name:
                raise RegistryError(f"Имя модели {name!r} уже занято")

        if kind_str == DetectorKind.BOUNDS.value:
            params.setdefault("bounds", self.bounds_for_detector())

        estimator = UnifiedAnomalyDetector(kind_str, estimator_params=params)
        if fit_data is not None:
            estimator.fit(fit_data)

        model_id = str(uuid.uuid4())
        artifact_file = f"{kind_str}__{name}__{model_id[:8]}.joblib"
        meta = ModelMeta(
            name=name,
            kind=kind_str,
            hyperparameters=params,
            model_id=model_id,
            artifact_file=artifact_file,
            trained_at=_utc_now() if fit_data is not None else None,
        )

        self._models_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, self._artifact_path(meta))

        manifest.models.append(meta)
        if set_current or manifest.current_name is None:
            manifest.current_kind = kind_str
            manifest.current_name = name
        self._save_manifest(manifest)
        return meta

    def delete_model(self, name: str, kind: str | None = None) -> None:
        manifest = self._load_manifest()
        meta = self._find_model(manifest, name, kind)
        meta.deleted = True
        path = self._artifact_path(meta)
        if path.exists():
            path.unlink()
        if manifest.current_name == name and (
            kind is None or manifest.current_kind == kind
        ):
            manifest.current_kind = None
            manifest.current_name = None
        self._save_manifest(manifest)

    def list_models(self) -> list[dict[str, Any]]:
        manifest = self._load_manifest()
        return [
            {
                "name": m.name,
                "kind": m.kind,
                "hyperparameters": m.hyperparameters,
                "model_id": m.model_id,
                "trained_at": m.trained_at,
                "created_at": m.created_at,
                "is_current": (
                    m.name == manifest.current_name and m.kind == manifest.current_kind
                ),
            }
            for m in manifest.models
            if not m.deleted
        ]

    def set_current(self, name: str, kind: DetectorKind | str) -> dict[str, str]:
        kind_str = _kind_str(kind)
        manifest = self._load_manifest()
        self._find_model(manifest, name, kind_str)
        manifest.current_kind = kind_str
        manifest.current_name = name
        self._save_manifest(manifest)
        return {"kind": kind_str, "name": name}

    def get_current(self) -> dict[str, Any] | None:
        manifest = self._load_manifest()
        if not manifest.current_name or not manifest.current_kind:
            return None
        meta = self._find_model(
            manifest, manifest.current_name, manifest.current_kind
        )
        return {
            "name": meta.name,
            "kind": meta.kind,
            "hyperparameters": meta.hyperparameters,
            "model_id": meta.model_id,
            "trained_at": meta.trained_at,
        }

    def load_current_estimator(self) -> UnifiedAnomalyDetector:
        manifest = self._load_manifest()
        if not manifest.current_name or not manifest.current_kind:
            raise RegistryError("Текущая модель не задана")
        meta = self._find_model(
            manifest, manifest.current_name, manifest.current_kind
        )
        path = self._artifact_path(meta)
        if not path.exists():
            raise RegistryError(f"Файл модели отсутствует: {path.name}")
        return joblib.load(path)

    def train_current(self, X: TimeSeriesX) -> dict[str, Any]:
        manifest = self._load_manifest()
        if not manifest.current_name or not manifest.current_kind:
            raise RegistryError("Текущая модель не задана")
        meta = self._find_model(
            manifest, manifest.current_name, manifest.current_kind
        )

        params = dict(meta.hyperparameters)
        if meta.kind == DetectorKind.BOUNDS.value:
            params["bounds"] = self.bounds_for_detector()

        estimator = UnifiedAnomalyDetector(meta.kind, estimator_params=params)
        estimator.fit(X)
        meta.trained_at = _utc_now()
        joblib.dump(estimator, self._artifact_path(meta))
        self._save_manifest(manifest)
        return {"name": meta.name, "kind": meta.kind, "trained_at": meta.trained_at}

    def predict(self, X: TimeSeriesX) -> dict[str, Any]:
        est = self.load_current_estimator()
        return {
            "predict": np.asarray(est.predict(X)).tolist(),
            "proba": np.asarray(est.predict_proba(X)).tolist(),
        }
