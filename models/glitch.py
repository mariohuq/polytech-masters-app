import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted


class StuckSignalDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, window_size=10, threshold=1e-5, metric='std'):
        """
        Модель для поиска зависания сигнала в многомерных временных рядах.
        
        Parameters:
        -----------
        window_size : int, default=10
            Размер скользящего окна (в количестве точек).
        threshold : float, default=1e-5
            Порог, ниже которого значение метрики считается зависанием.
        metric : str, default='std'
            Метрика для оценки изменчивости: 'std' (стандартное отклонение) или 'mad' (абсолютное отклонение от медианы).
        """
        self.window_size = window_size
        self.threshold = threshold
        self.metric = metric
        
    def fit(self, X, y=None):
        """
        Для данного метода обучение без учителя не требуется, 
        но мы валидируем входные данные для соответствия стандартам sklearn.
        """
        X = check_array(X, ensure_2d=True, ensure_all_finite='allow-nan')
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        """
        Предсказывает аномалии (зависания) для каждой точки.
        
        Returns:
        --------
        y_pred : np.ndarray размера (n_samples, n_features)
            1 — сигнал завис (аномалия), 0 — сигнал в норме.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, ensure_2d=True, ensure_all_finite='allow-nan')
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Размерность признаков {X.shape[1]} не совпадает с обученной {self.n_features_in_}")
            
        n_samples, n_features = X.shape

        # Превращаем в DataFrame для удобного использования скользящего окна pandas
        df = pd.DataFrame(X)
        
        if self.metric == 'std':
            # Считаем скользящее стандартное отклонение
            rolling_metric = df.rolling(window=self.window_size, min_periods=1).std()
        elif self.metric == 'mad':
            # Считаем скользящее среднее абсолютное отклонение от медианы (MAD)
            def mad(x):
                median = np.nanmedian(x)
                return np.nanmedian(np.abs(x - median))
            rolling_metric = df.rolling(window=self.window_size, min_periods=1).apply(mad, raw=True)
        else:
            raise ValueError(f"Неизвестная метрика: {self.metric}. Используйте 'std' или 'mad'.")
            
        # Заполняем NaN в начале окна нулями, чтобы не триггерить ложные аномалии на старте
        rolling_metric.bfill(inplace=True)
        
        # Если изменчивость меньше порога — это зависание (1)
        
        return np.asarray(rolling_metric < self.threshold).astype(int)


class ProbabilisticStuckSignalDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, window_size=10, expected_noise=0.1, metric='std', decision_threshold=0.5):
        """
        Вероятностный одноклассовый классификатор зависания сигнала.
        
        Parameters:
        -----------
        window_size : int, default=10
            Размер скользящего окна.
        expected_noise : float, default=0.1
            Ожидаемый уровень шума (std) в НОРМАЛЬНОМ состоянии.
            Если текущий шум падает сильно ниже этого значения, вероятность аномалии растет к 1.
        metric : str, default='std'
            Метрика изменчивости ('std' или 'mad').
        decision_threshold : float, default=0.5
            Порог вероятности для метода predict() (выше какого уровня считать аномалией).
        """
        self.window_size = window_size
        self.expected_noise = expected_noise
        self.metric = metric
        self.decision_threshold = decision_threshold
        
    def fit(self, X, y=None):
        """
        Если expected_noise не задан пользователем, мы можем автоматически 
        посчитать его как медианную изменчивость по всему сигналу (предполагая, что большую часть времени он в норме).
        """
        X = check_array(X, ensure_2d=True, ensure_all_finite='allow-nan')
        self.n_features_in_ = X.shape[1]
        
        # Автоматический подбор ожидаемого шума, если передан 0 или None
        if not self.expected_noise:
            if self.metric == 'std':
                self.expected_noise_map_ = np.nanstd(X, axis=0)
            else:
                median = np.nanmedian(X, axis=0)
                self.expected_noise_map_ = np.nanmedian(np.abs(X - median), axis=0)
        else:
            # Если передано одно число, тиражируем его на все фичи
            self.expected_noise_map_ = np.full(self.n_features_in_, self.expected_noise)
            
        self.is_fitted_ = True
        return self
        
    def predict_proba(self, X):
        """
        Вычисляет вероятность зависания для каждого отсчета и каждой компоненты.
        
        Returns:
        --------
        proba : np.ndarray размера (n_samples, n_features)
            Значения от 0.0 (сигнал активно шумит) до 1.0 (сигнал абсолютно статичен).
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, ensure_2d=True, ensure_all_finite='allow-nan')
        
        df = pd.DataFrame(X)
        if self.metric == 'std':
            rolling_metric = df.rolling(window=self.window_size, min_periods=1).std()
        elif self.metric == 'mad':
            def mad(x):
                m = np.nanmedian(x)
                return np.nanmedian(np.abs(x - m))
            rolling_metric = df.rolling(window=self.window_size, min_periods=1).apply(mad, raw=True)
            
        rolling_metric.bfill(inplace=True)
        V = rolling_metric.values # Матрица текущей изменчивости
        
        # --- Магическая логика перевода в вероятность ---
        # Мы хотим, чтобы при V -> 0 вероятность стремилась к 1.
        # А при V >= expected_noise вероятность стремилась к 0.
        # Используем экспоненциальное затухание: P = exp(- k * V)
        # Подберем коэффициент k так, чтобы при V = expected_noise вероятность была близка к 0.01
        
        k = -np.log(0.01) / (self.expected_noise_map_ + 1e-9)
        probabilities = np.exp(-k * V)
        
        return np.clip(probabilities, 0.0, 1.0)
        
    def predict(self, X):
        """
        Бинарный прогноз на основе порога вероятности decision_threshold.
        """
        proba = self.predict_proba(X)
        return (proba > self.decision_threshold).astype(int)