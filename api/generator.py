from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int
    n_features: int
    step_sigma: float
    spike_prob: float
    spike_scale: float
    plateau_prob: float
    plateau_len: int


class MockSeriesGenerator:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        cfg = self._draw_config(rng)

        self.n_features = cfg.n_features
        self.step_sigma = cfg.step_sigma
        self.spike_prob = cfg.spike_prob
        self.spike_scale = cfg.spike_scale
        self.plateau_prob = cfg.plateau_prob
        self.plateau_len = cfg.plateau_len

        self._rng = rng
        self._state = self._rng.normal(size=self.n_features)
        self._plateau_left = 0
        self._plateau_value: np.ndarray | None = None
        self._step = 0

    @staticmethod
    def _draw_config(rng: np.random.Generator) -> GeneratorConfig:
        return GeneratorConfig(
            seed=-1,
            n_features=2 + int(rng.integers(0, 3)),
            step_sigma=float(rng.uniform(0.03, 0.08)),
            spike_prob=float(rng.uniform(0.01, 0.04)),
            spike_scale=float(rng.uniform(2.0, 4.0)),
            plateau_prob=float(rng.uniform(0.005, 0.02)),
            plateau_len=int(rng.integers(15, 40)),
        )

    @classmethod
    def config_from_seed(cls, seed: int) -> GeneratorConfig:
        rng = np.random.default_rng(seed)
        drawn = cls._draw_config(rng)
        return GeneratorConfig(
            seed=seed,
            n_features=drawn.n_features,
            step_sigma=drawn.step_sigma,
            spike_prob=drawn.spike_prob,
            spike_scale=drawn.spike_scale,
            plateau_prob=drawn.plateau_prob,
            plateau_len=drawn.plateau_len,
        )

    def describe(self) -> dict:
        return asdict(self.config_from_seed(self.seed))

    def next_row(self) -> np.ndarray:
        if self._plateau_left > 0 and self._plateau_value is not None:
            self._plateau_left -= 1
            self._step += 1
            return self._plateau_value.copy()

        step = self._rng.normal(scale=self.step_sigma, size=self.n_features)
        self._state = self._state + step

        if self._rng.random() < self.spike_prob:
            self._state = self._state + self._rng.normal(
                scale=self.spike_scale, size=self.n_features
            )

        if self._rng.random() < self.plateau_prob:
            self._plateau_value = self._state.copy()
            self._plateau_left = self.plateau_len - 1

        self._step += 1
        return self._state.copy()

    @property
    def step(self) -> int:
        return self._step
