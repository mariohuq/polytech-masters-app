from __future__ import annotations

import numpy as np


class MockSeriesGenerator:

    def __init__(
        self,
        n_features: int = 2,
        seed: int | None = None,
        step_sigma: float = 0.05,
        spike_prob: float = 0.02,
        spike_scale: float = 3.0,
        plateau_prob: float = 0.01,
        plateau_len: int = 25,
    ):
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)
        self.step_sigma = step_sigma
        self.spike_prob = spike_prob
        self.spike_scale = spike_scale
        self.plateau_prob = plateau_prob
        self.plateau_len = plateau_len
        self._state = self.rng.normal(size=n_features)
        self._plateau_left = 0
        self._plateau_value: np.ndarray | None = None

    def next_row(self) -> np.ndarray:
        if self._plateau_left > 0 and self._plateau_value is not None:
            self._plateau_left -= 1
            return self._plateau_value.copy()

        step = self.rng.normal(scale=self.step_sigma, size=self.n_features)
        self._state = self.state + step

        if self.rng.random() < self.spike_prob:
            self._state += self.rng.normal(
                scale=self.spike_scale, size=self.n_features
            )

        if self.rng.random() < self.plateau_prob:
            self._plateau_value = self._state.copy()
            self._plateau_left = self.plateau_len - 1

        return self._state.copy()

    @property
    def state(self) -> np.ndarray:
        return self._state
