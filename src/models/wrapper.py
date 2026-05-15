from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator


from .bounds import BoundsDetector
from .glitch import ProbabilisticStuckSignalDetector, StuckSignalDetector
from .spikes import (
    DetrendedRollingOutlier,
    RobustDetrendedRollingOutlier,
    RobustRollingOutlier,
    RobustRollingOutlierLookahead,
    SklearnWindowOutlier,
)


# Вход сигналов: строки — моменты времени (t = 0 … n_samples-1), столбцы — признаки (сенсоры / каналы).
# Рекомендуется float (NaN допустимы там, где это поддерживает конкретный бэкенд).
TimeSeriesX: TypeAlias = npt.NDArray[np.floating] | pd.DataFrame

REGISTRY: dict[str, type] = {
    "bounds": BoundsDetector,
    "stuck": StuckSignalDetector,
    "stuck_proba": ProbabilisticStuckSignalDetector,
    "robust_rolling": RobustRollingOutlier,
    "detrended_rolling": DetrendedRollingOutlier,
    "robust_detrended": RobustDetrendedRollingOutlier,
    "sklearn_window": SklearnWindowOutlier,
    "robust_rolling_lookahead": RobustRollingOutlierLookahead,
}


def list_models() -> list[str]:
    """Список допустимых значений аргумента ``UnifiedAnomalyDetector(model=...)``."""
    return sorted(REGISTRY.keys())


class UnifiedAnomalyDetector(BaseEstimator):
    """Обёртка над детекторами проекта в sklearn-стиле.

    **Данные ``X`` (и в ``fit``, и в ``predict`` / ``predict_proba``)**

    - **Смысл:** многомерный временной ряд без явной колонки времени: индекс строки
      — дискретный шаг ``t``, столбцы — независимые измеряемые величины
      (размерность ``n_features``).
    - **Форма:** ``(n_samples, n_features)``, где ``n_samples ≥ 1``, ``n_features ≥ 1``.
    - **Тип:** удобнее всего ``numpy.ndarray`` с вещественным ``dtype`` (например ``float64``)
      или ``pandas.DataFrame`` (числовые столбцы).
      Часть бэкендов внутри приводит ``ndarray`` к ``DataFrame``; для ``BoundsDetector``
      при ``fit`` на ``DataFrame`` словарь ``bounds`` может ключоваться **именами столбцов**,
      на ``ndarray`` — порядок границ соответствует **номеру столбца** (см. ``BoundsDetector``).
    - **Ответы:** ``predict`` и ``predict_proba`` возвращают ``ndarray`` формы
      ``(n_samples, n_features)`` — по одному значению на каждую ячейку ``X``
      (бинарная метка или вероятность «аномалии» в смысле конкретной модели).

    Выбор реализации — строка ``model``, гиперпараметры — ``estimator_params`` или ``**kwargs``.
    """

    def __init__(
        self,
        model: str = "robust_rolling",
        estimator_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.model = model
        merged: dict[str, Any] = dict(estimator_params or ())
        merged.update(kwargs)
        self.estimator_params = merged

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"model": self.model, "estimator_params": dict(self.estimator_params)}

    def set_params(self, **params: Any) -> UnifiedAnomalyDetector:
        if not params:
            return self
        params = dict(params)
        if "model" in params:
            self.model = params.pop("model")
        if "estimator_params" in params:
            self.estimator_params = dict(params.pop("estimator_params"))
        self.estimator_params.update(params)
        self._clear_estimator()
        return self

    def _clear_estimator(self) -> None:
        if hasattr(self, "estimator_"):
            del self.estimator_

    def _build(self) -> BaseEstimator:
        if self.model not in REGISTRY:
            raise ValueError(
                f"Неизвестная модель {self.model!r}. Доступно: {list_models()}"
            )
        cls = REGISTRY[self.model]
        return cls(**dict(self.estimator_params))

    def fit(self, X, y=None) -> UnifiedAnomalyDetector:
        est = self._build()
        if hasattr(est, "fit"):
            est.fit(X, y)
        self.estimator_: BaseEstimator = est
        return self

    def predict(self, X) -> np.ndarray:
        self._require_estimator()
        est = self.estimator_
        if hasattr(est, "predict"):
            pred = est.predict(X)
            return np.asarray(pred, dtype=np.int32)
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            return (np.asarray(proba, dtype=float) >= 0.5).astype(np.int32)
        raise TypeError(f"{type(est).__name__}: нет predict и predict_proba")

    def predict_proba(self, X) -> np.ndarray:
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
