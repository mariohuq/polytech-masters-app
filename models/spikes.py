import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from scipy.special import expit
from sklearn.linear_model import HuberRegressor

class RobustRollingOutlier(BaseEstimator, OutlierMixin):
    def __init__(self, window=10, sigma_cutoff=3.0, sensitivity=1.0):
        self.window = window
        self.sigma_cutoff = sigma_cutoff
        self.sensitivity = sensitivity # Higher = more sensitive to small spikes

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # 1. Calculate Rolling Median (Robust center)
        rolling_median = (
            X.rolling(window=self.window, center=True)
            .median()
            .ffill()
            .bfill()
        )
        # 2. Calculate Rolling MAD (Robust scale)
        # MAD = median(|x - median|)
        rolling_mad = (
            (X - rolling_median).abs()
            .rolling(window=self.window, center=True)
            .median()
            .ffill()
            .bfill()
        )
        
        # 3. Robust Z-Score (Modified Z-score)
        # 0.6745 scales MAD to be consistent with Std Dev for normal distributions
        modified_z = 0.6745 * (X - rolling_median) / (rolling_mad + 1e-6)
        
        # 3. Probability via Sigmoid using hyperparameters
        # We use self.sigma_cutoff to define the 50% probability point
        # We use self.sensitivity to define the steepness
        proba = expit(self.sensitivity * (np.abs(modified_z) - self.sigma_cutoff))
        return np.nan_to_num(proba.values, nan=0.0)


class DetrendedRollingOutlier(BaseEstimator, OutlierMixin):
    def __init__(self, window=30, degree=1, sigma_cutoff=3.0, sensitivity=1.5):
        self.window = window
        self.degree = degree
        self.sigma_cutoff = sigma_cutoff
        self.sensitivity = sensitivity

    def fit(self, X, y=None):
        return self

    def _get_local_residual(self, segment):
        """Fits a polynomial to the segment and returns the residual of the last point."""
        assert len(segment) > 0
        segment = np.asarray(segment)
        x = np.arange(len(segment))
        coeffs = np.polyfit(x, segment, self.degree)
        poly_func = np.poly1d(coeffs)
        # Residual = Actual value - Predicted trend value at the last point
        return segment[-1] - poly_func(x[-1])

    def predict_proba(self, X):
        df = pd.DataFrame(X)
        n_samples, n_dims = df.shape
        
        # 1. Calculate Rolling Residuals (Detrending)
        # This removes the local d-poly trend for each dimension
        residuals = df.apply(lambda col: col.rolling(
            window=self.window, min_periods=self.degree + 2
        ).apply(self._get_local_residual))
        
        # Fill initial NaNs caused by the window startup
        residuals = residuals.fillna(0)
        
        # 2. Robust Scaling of Residuals
        # We use MAD on the residuals to find the 'typical' noise level after detrending
        rolling_mad = residuals.abs().rolling(
            window=self.window, min_periods=1
        ).median().fillna(1e-6)
        
        # 3. Modified Z-Score on the detrended signal
        modified_z = 0.6745 * residuals / (rolling_mad + 1e-6)
        
        # 4. Probability via Expit
        proba = expit(self.sensitivity * (np.abs(modified_z) - self.sigma_cutoff))
        
        return proba.values


class RobustDetrendedRollingOutlier(BaseEstimator, OutlierMixin):
    def __init__(self, window=30, degree=1, sigma_cutoff=3.0, sensitivity=1.5, epsilon=1.35):
        self.window = window
        self.degree = degree
        self.sigma_cutoff = sigma_cutoff
        self.sensitivity = sensitivity
        self.epsilon = epsilon  # Huber hyperparameter: lower = more robust to window outliers

    def fit(self, X, y=None):
        return self

    def _get_robust_local_residual(self, segment):
        """Fits a robust Huber polynomial regression to the segment and returns the last residual."""
        n = len(segment)
        x = np.arange(n)
        segment = np.asarray(segment)
        
        # 1. Create Polynomial Features manually: [x, x^2, x^3...]
        # shape: (n_samples, degree)
        X_poly = np.vander(x, self.degree + 1)[:, :-1] 
        
        # 2. Use HuberRegressor instead of Least Squares (polyfit)
        # alpha=0 disables regularization so it behaves purely like an un-penalized fit
        huber = HuberRegressor(epsilon=self.epsilon, alpha=0.1, max_iter=200)
        
        try:
            huber.fit(X_poly, segment)
            # Predict the value at the very last point in the window
            pred_last = huber.predict(X_poly[-1, :].reshape(1, -1))[0]
            return segment[-1] - pred_last
        except Exception:
            # Fallback if Huber fails to converge on a bad window segment
            return segment[-1] - np.median(segment)

    def predict_proba(self, X):
        df = pd.DataFrame(X)
        
        # 1. Calculate Rolling Residuals using the Robust Huber engine
        # Requires at least enough points to fit the polynomial degrees
        residuals = df.apply(lambda col: col.rolling(
            window=self.window, min_periods=self.degree + 2
        ).apply(self._get_robust_local_residual))
        
        residuals = residuals.fillna(0)
        
        # 2. Robust Scaling of the resulting residuals
        rolling_mad = residuals.abs().rolling(
            window=self.window, min_periods=1
        ).median().fillna(1e-6)
        
        # 3. Modified Z-Score
        modified_z = 0.6745 * residuals / (rolling_mad + 1e-6)
        
        # 4. Probability via Expit (Sigmoid)
        proba = expit(self.sensitivity * (np.abs(modified_z) - self.sigma_cutoff))
        
        return proba.values


from sklearn.neighbors import LocalOutlierFactor

class SklearnWindowOutlier(BaseEstimator, OutlierMixin):
    def __init__(self, window=30, contamination=0.05, sensitivity=2.0):
        self.window = window
        self.contamination = contamination
        self.sensitivity = sensitivity

    def fit(self, X, y=None):
        # Natively, LOF is an unsupervised detector; fit is still empty
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        n_samples, n_dims = X_arr.shape
        
        # Initialize an empty array for probabilities
        proba_matrix = np.zeros((n_samples, n_dims))
        
        # Create a time feature to serve as our "anchor" for the temporal window
        time_feature = np.arange(n_samples).reshape(-1, 1)
        
        # Process each dimension completely independently
        for d in range(n_dims):
            # Combine [Time, Signal_Value] so the model measures local distance in time AND value
            # We scale the time feature so the 'window' hyperparameter limits the neighbors
            X_dim = np.hstack([time_feature, X_arr[:, [d]]])
            
            # n_neighbors behaves similarly to your time window constraint
            lof = LocalOutlierFactor(
                n_neighbors=self.window, 
                contamination=self.contamination,
                novelty=False
            )
            
            # Fit and predict the data points against themselves
            lof.fit_predict(X_dim)
            
            # negative_outlier_factor_: closer to -1 is normal, much lower than -1 is an outlier
            # We invert it so higher numbers mean more anomalous
            raw_scores = -lof.negative_outlier_factor_
            
            # Standardize the scores around the expected normal factor (which is 1.0)
            # A score of 1.5 means the point is 1.5x less dense than its neighbors
            standardized_scores = (raw_scores - 1.0)
            
            # Squash into probabilities using the sigmoid function
            # Subtracting 0.5 centers the 50% probability mark around a score of 1.5
            proba_matrix[:, d] = expit(self.sensitivity * (standardized_scores - 0.5))
            
        return proba_matrix

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)



class RobustRollingOutlierLookahead(BaseEstimator, OutlierMixin):
    def __init__(self, window=11, sigma_cutoff=3.0, sensitivity=1.0):
        # window лучше брать нечетным (например, 11), тогда центр — это ровно текущая точка,
        # а window // 2 (например, 5) — это количество точек «в будущем» для анализа.
        self.window = window if window % 2 != 0 else window + 1
        self.sigma_cutoff = sigma_cutoff
        self.sensitivity = sensitivity 

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        half_w = self.window // 2

        # 1. Локальный уровень определяем по центрированному окну (смотрим вперед и назад)
        rolling_median = (
            X.rolling(window=self.window, center=True, min_periods=1)
            .median()
            .ffill().bfill()
        )
        
        # 2. ХИТРОСТЬ: Чтобы смена режима не раздувала MAD, считаем локальную вариацию 
        # через первую разность (дифференциалы). На ступеньке будет только ОДНО большое значение,
        # которое медиана внутри окна просто проигнорирует!
        abs_diffs = X.diff().abs()
        rolling_mad = (
            abs_diffs.rolling(window=self.window, center=True, min_periods=1)
            .median()
            .ffill().bfill()
        )
        
        # Пересчитываем MAD в эквивалент стандартного отклонения для разностей
        # (для нормального распределения разностей константа немного другая, ~0.95)
        robust_scale = rolling_mad / 0.9542
        
        # Защитный epsilon от "нулевых" полок
        global_std = X.std()[:len(X.columns)]
        epsilon = np.maximum(1e-6, 0.01 * global_std.values)
        robust_scale = robust_scale + epsilon

        # 3. Считаем Modified Z-Score
        # Отклонение текущей точки от центрированной медианы
        deviation = X - rolling_median
        modified_z = 0.6745 * deviation / robust_scale

        # 4. ФИЛЬТР БУДУЩЕГО (Логика подавления смены режима):
        # Смотрим на точки вперед. Если это смена режима, то точки в будущем (t+1, t+2...)
        # тоже будут находиться близко к новому rolling_median, то есть их deviation будет МАЛЕНЬКИМ.
        # Если это одиночный выброс, то будущие точки вернутся на старый уровень, и их отклонение
        # от ТЕКУЩЕЙ медианы (которая на короткое время станет промежуточной) будет другим.
        
        # Но проще и надежнее сделать так: если следующие (half_w) точек ТОЖЕ имеют 
        # знак отклонения того же направления и остаются на новом уровне, мы штрафуем z-score.
        
        # Реализуем через проверку: "упало" ли значение обратно?
        # Если последующие точки подтверждают новый уровень, зануляем z-score для текущей точки.
        for col in X.columns:
            # Сдвигаем ряд назад, чтобы заглянуть в будущее для каждой точки
            future_values = np.column_stack([X[col].shift(-i).values for i in range(1, half_w + 1)])
            future_medians = np.column_stack([rolling_median[col].shift(-i).values for i in range(1, half_w + 1)])
            
            # Если будущие точки стабильно близки к будущим медианам — значит режим закрепился!
            # И значит текущий скачок — это просто фронт новой ступени.
            future_deviations = np.abs(future_values - future_medians)
            
            # Трэшхолд стабильности будущего (будущие точки не должны быть аномальными относительно новой медианы)
            future_is_stable = np.all(future_deviations < (robust_scale[col].values[:, None] * self.sigma_cutoff), axis=1)
            
            # Если режим сменился (стабилен в будущем), и это длится дольше чем просто импульс:
            # Мы смотрим, не вернулся ли ряд назад. Если не вернулся — гасим Z-score текущей точки.
            is_step = future_is_stable & (np.abs(X[col] - rolling_median[col]) > robust_scale[col] * self.sigma_cutoff)
            
            # Дополнительная проверка: удерживается ли уровень в будущем (не вернулись ли назад к исходной точке)
            # Если будущие точки равны (плюс-минус) текущей точке X_t, значит X_t — это начало плато!
            future_close_to_current = np.all(np.abs(future_values - X[col].values[:, None]) < (robust_scale[col].values[:, None] * self.sigma_cutoff), axis=1)
            
            # Если это подтвержденная ступень — обнуляем для нее Z-score
            modified_z.loc[future_close_to_current & is_step, col] = 0.0

        # 5. Probability via Sigmoid
        proba = expit(self.sensitivity * (np.abs(modified_z) - self.sigma_cutoff))
        return np.nan_to_num(proba.values, nan=0.0)