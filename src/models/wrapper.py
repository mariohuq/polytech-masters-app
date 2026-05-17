

from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator

from .bounds import BoundsDetector
from .glitch import ProbabilisticStuckSignalDetector
from .spikes import RobustRollingOutlierLookahead

# Строки — шаги времени t, столбцы — признаки (сенсоры).
TimeSeriesX: TypeAlias = npt.NDArray[np.floating] | pd.DataFrame


class DetectorKind(str, Enum):
    BOUNDS = "bounds"
    SPIKES = "spikes"
    GLITCH = "glitch"


# Реализация по умолчанию для каждого вида (можно переопределить через estimator_params).
_DEFAULT_ESTIMATORS: dict[DetectorKind, type[BaseEstimator]] = {
    DetectorKind.BOUNDS: BoundsDetector,
    DetectorKind.SPIKES: RobustRollingOutlierLookahead,
    DetectorKind.GLITCH: ProbabilisticStuckSignalDetector,
}


def list_models() -> list[str]:
    """Имена для API / query-параметра ``model``."""
    return [k.value for k in DetectorKind]


def _coerce_kind(model: DetectorKind | str) -> DetectorKind:
    if isinstance(model, DetectorKind):
        return model
    try:
        return DetectorKind(model)
    except ValueError as e:
        raise ValueError(
            f"Неизвестная модель {model!r}. Доступно: {list_models()}"
        ) from e


class UnifiedAnomalyDetector(BaseEstimator):
    """Обёртка над тремя детекторами: выбор через ``DetectorKind`` или строку ``bounds`` / ``spikes`` / ``glitch``.

  **Данные ``X``** — матрица ``(n_samples, n_features)``: ``ndarray`` (float) или ``DataFrame``.
  **Ответ** — ``predict`` / ``predict_proba`` формы ``(n_samples, n_features)``.

  Гиперпараметры конкретного бэкенда передаются в ``estimator_params`` или как ``**kwargs``
  (например ``bounds={...}`` для ``bounds``, ``window=15`` для ``spikes``).
    """

    def __init__(
        self,
        model: DetectorKind | str = DetectorKind.SPIKES,
        estimator_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.model = model
        merged: dict[str, Any] = dict(estimator_params or ())
        merged.update(kwargs)
        self.estimator_params = merged

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        model = self.model.value if isinstance(self.model, DetectorKind) else self.model
        return {"model": model, "estimator_params": dict(self.estimator_params)}

    def set_params(self, **params: Any) -> UnifiedAnomalyDetector:
        if not params:
            return self
        params = dict(params)
        if "model" in params:
            self.model = _coerce_kind(params.pop("model"))
        if "estimator_params" in params:
            self.estimator_params = dict(params.pop("estimator_params"))
        self.estimator_params.update(params)
        self._clear_estimator()
        return self

    @property
    def kind_(self) -> DetectorKind:
        """Вид детектора: после ``fit`` — зафиксированный, иначе из ``model``."""
        if hasattr(self, "kind__"):
            return self.kind__
        return _coerce_kind(self.model)

    def _clear_estimator(self) -> None:
        for attr in ("estimator_", "kind__"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _build(self) -> tuple[DetectorKind, BaseEstimator]:
        kind = _coerce_kind(self.model)
        cls = _DEFAULT_ESTIMATORS[kind]
        return kind, cls(**dict(self.estimator_params))

    def fit(self, X: TimeSeriesX, y: None = None) -> UnifiedAnomalyDetector:
        kind, est = self._build()
        if hasattr(est, "fit"):
            est.fit(X, y)
        self.kind__ = kind
        self.estimator_: BaseEstimator = est
        return self

    def predict(self, X: TimeSeriesX) -> npt.NDArray[np.int32]:
        self._require_estimator()
        est = self.estimator_
        if hasattr(est, "predict"):
            return np.asarray(est.predict(X), dtype=np.int32)
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            return (np.asarray(proba, dtype=float) >= 0.5).astype(np.int32)
        raise TypeError(f"{type(est).__name__}: нет predict и predict_proba")

    def predict_proba(self, X: TimeSeriesX) -> npt.NDArray[np.floating]:
        self._require_estimator()
        est = self.estimator_
        if hasattr(est, "predict_proba"):
            return np.asarray(est.predict_proba(X), dtype=float)
        if hasattr(est, "predict"):
            return np.asarray(est.predict(X), dtype=float)
        raise TypeError(f"{type(est).__name__}: нет predict_proba и predict")

    def _require_estimator(self) -> None:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Сначала вызовите fit(X).")
