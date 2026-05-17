from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from registry import ModelRegistry, RegistryError, SensorConfig

REGISTRY_ROOT = Path(__file__).resolve().parents[1] / "data" / "registries"

router = APIRouter(prefix="/registry", tags=["registry"])


def _root() -> Path:
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    return REGISTRY_ROOT


class SensorIn(BaseModel):
    name: str
    low: float
    high: float


class CreateRegistryBody(BaseModel):
    title: str = ""
    registry_id: str | None = None
    sensors: list[SensorIn] = Field(..., min_length=1)


class AddModelBody(BaseModel):
    name: str = Field(..., description="Уникальное имя от пользователя")
    kind: str = Field(..., description="bounds | spikes | glitch")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    set_current: bool = False
    fit: list[list[float]] | None = Field(
        None, description="Опционально: матрица для fit при добавлении"
    )


class SetCurrentBody(BaseModel):
    name: str
    kind: str


class TrainBody(BaseModel):
    X: list[list[float]]


class InferBody(BaseModel):
    X: list[list[float]]


def _http(e: RegistryError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


@router.post("")
def create_registry(body: CreateRegistryBody) -> dict[str, Any]:
    sensors = [SensorConfig(s.name, s.low, s.high) for s in body.sensors]
    try:
        reg = ModelRegistry.create(
            _root(),
            sensors,
            registry_id=body.registry_id,
            title=body.title,
        )
        return reg.get_info()
    except RegistryError as e:
        raise _http(e) from e


@router.get("")
def list_registries() -> dict[str, Any]:
    return {"registries": ModelRegistry.list_registries(_root())}


@router.get("/{registry_id}")
def get_registry(registry_id: str) -> dict[str, Any]:
    try:
        return ModelRegistry.open(_root(), registry_id).get_info()
    except RegistryError as e:
        raise _http(e) from e


@router.delete("/{registry_id}")
def delete_registry(registry_id: str) -> dict[str, str]:
    try:
        ModelRegistry.open(_root(), registry_id).mark_deleted()
        return {"status": "deleted", "registry_id": registry_id}
    except RegistryError as e:
        raise _http(e) from e


@router.get("/{registry_id}/models")
def list_models(registry_id: str) -> dict[str, Any]:
    try:
        reg = ModelRegistry.open(_root(), registry_id)
        return {"models": reg.list_models(), "current": reg.get_current()}
    except RegistryError as e:
        raise _http(e) from e


@router.post("/{registry_id}/models")
def add_model(registry_id: str, body: AddModelBody) -> dict[str, Any]:
    try:
        reg = ModelRegistry.open(_root(), registry_id)
        X = np.asarray(body.fit, dtype=float) if body.fit is not None else None
        meta = reg.add_model(
            body.name,
            body.kind,
            body.hyperparameters,
            fit_data=X,
            set_current=body.set_current,
        )
        return meta.to_dict()
    except RegistryError as e:
        raise _http(e) from e


@router.delete("/{registry_id}/models/{name}")
def delete_model(registry_id: str, name: str, kind: str | None = None) -> dict[str, str]:
    try:
        ModelRegistry.open(_root(), registry_id).delete_model(name, kind)
        return {"status": "deleted", "name": name}
    except RegistryError as e:
        raise _http(e) from e


@router.put("/{registry_id}/models/current")
def set_current(registry_id: str, body: SetCurrentBody) -> dict[str, str]:
    try:
        return ModelRegistry.open(_root(), registry_id).set_current(body.name, body.kind)
    except RegistryError as e:
        raise _http(e) from e


@router.get("/{registry_id}/models/current")
def get_current(registry_id: str) -> dict[str, Any]:
    try:
        cur = ModelRegistry.open(_root(), registry_id).get_current()
        if cur is None:
            raise HTTPException(status_code=404, detail="Текущая модель не задана")
        return cur
    except RegistryError as e:
        raise _http(e) from e


@router.post("/{registry_id}/models/current/train")
def train_current(registry_id: str, body: TrainBody) -> dict[str, Any]:
    try:
        X = np.asarray(body.X, dtype=float)
        return ModelRegistry.open(_root(), registry_id).train_current(X)
    except RegistryError as e:
        raise _http(e) from e


@router.post("/{registry_id}/infer")
def infer(registry_id: str, body: InferBody) -> dict[str, Any]:
    try:
        X = np.asarray(body.X, dtype=float)
        return ModelRegistry.open(_root(), registry_id).predict(X)
    except RegistryError as e:
        raise _http(e) from e
