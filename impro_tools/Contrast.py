import numpy as np
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import find_peaks
from typing import Tuple, Optional, List


class Contrast:
    """
    Utility class for Michelson and MTF contrast calculations.
    """
    ZERO_DIV_CONST: float = 1e-13


    @classmethod
    def get_Michelson(cls, y: np.ndarray) -> float:
        """
        Compute the Michelson contrast for a 1D signal.

        Parameters
        ----------
        y : np.ndarray
            The intensity profile (e.g. grayscale values).

        Returns
        -------
        float
            The Michelson contrast, or 0.0 if conditions are not met.
        """
        # Find peaks (maxima) and minima
        maxima, _ = find_peaks(y, prominence=5)
        minima, _ = find_peaks(255 - y, prominence=5)
        line_indices = np.concatenate((maxima, minima))
        line_indices.sort()

        cont_sum = cls.ZERO_DIV_CONST
        contrast = 0.0

        # A simple heuristic: only compute if exactly 5 key points found
        if len(line_indices) == 5:
            max_diff = 0.0
            for idx in range(len(line_indices) - 1):
                diff = abs(y[line_indices[idx]] - y[line_indices[idx + 1]])
                if diff > max_diff:
                    max_diff = diff
                    cont_sum = abs(y[line_indices[idx]] + y[line_indices[idx + 1]])
            contrast = max_diff / cont_sum

        return contrast
    


    @classmethod
    def get_MTF(cls, profile: np.ndarray, pixel_pitch: float = 0.25967
    ) -> Tuple[Optional[np.ndarray], List[float], Optional[float], Optional[float], np.ndarray]:
        """
        Compute the Modulation Transfer Function (MTF) constrast from an edge profile.

        This method fits a tanh to the edge profile, computes the line spread function (LSF),
        performs a Fourier transform, and returns the normalized MTF along with MTF50 and MTF10.

        Parameters
        ----------
        profile : np.ndarray
            1D array of intensity values along an edge.
        pixel_pitch : float, optional
            Spacing between samples (e.g. in mm), by default 0.25967.

        Returns
        -------
        Tuple[
            Optional[np.ndarray],  # Frequencies (lp / mm) or None if failed
             np.ndarray,           # Normalized MTF values
            Optional[float],       # MTF50 (frequency where MTF = 0.5)
            Optional[float],       # MTF10 (frequency where MTF = 0.1)
            np.ndarray              # Fitted edge profile (tanh fit)
        ]
        """
        if len(profile) < 10:
            return None, np.array([]), None, None, np.array([])

        x_data = np.arange(profile.shape[0])
        y_data = profile

        # Fit tanh to the profile
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                params, _ = curve_fit(
                    cls.tanh_func, x_data, y_data, p0=[2.0, 1.5, len(x_data) / 2, np.mean(y_data)]
                )
        except (RuntimeError, ValueError):
            return None, np.array([]), None, None, np.array([])

        y_fit = cls.tanh_func(x_data, *params)

        # Compute derivative (LSF) from fit
        dy = np.gradient(y_fit)
        slope_start = abs(dy[0])
        slope_end = abs(dy[-1])
        slope_middle = np.max(np.abs(dy))

        # Ensure shape is approximately a tanh (flat tails, steep center)
        if not (slope_start < 0.2 * slope_middle and slope_end < 0.2 * slope_middle):
            return None, np.array([]), None, None, y_fit

        amplitude = np.max(y_fit) - np.min(y_fit)
        if amplitude < 0.5:
            return None, np.array([]), None, None, y_fit

        # Compute Line Spread Function (LSF)
        lsf = np.diff(y_fit)
        window = np.hanning(lsf.size)
        lsf_windowed = lsf * window

        # Fourier transform to get MTF
        mtf = np.abs(np.fft.fft(lsf_windowed))

        # Frequency axis
        N = lsf.size
        freq = np.fft.fftfreq(N, d=pixel_pitch)
        f_pos = freq[: N // 2]
        mtf_pos = mtf[: N // 2]

        # Normalize
        mtf_norm = mtf_pos / (mtf_pos[0] + cls.ZERO_DIV_CONST)

        # Compute MTF50 and MTF10
        mtf50 = float(np.interp(0.5, mtf_norm[::-1], f_pos[::-1]))
        mtf10 = float(np.interp(0.1, mtf_norm[::-1], f_pos[::-1]))

        return f_pos, mtf_norm, mtf50, mtf10, y_fit
    


    @staticmethod
    def tanh_func(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """
        Hyperbolic tangent function used to fit an edge profile.

        f(x) = a * tanh(b * (x - c)) + d

        Parameters
        ----------
        x : np.ndarray
            Independent variable (e.g. pixel positions).
        a, b, c, d : float
            Parameters of the tanh function.

        Returns
        -------
        np.ndarray
            Computed y-values.
        """
        return a * np.tanh(b * (x - c)) + d
    


    @staticmethod
    def get_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Root Mean Square Error (RMSE) between true and predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The RMSE.
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
