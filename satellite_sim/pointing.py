from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class PointingOptions:
    dt: float = 0.01
    duration: float = 3600.0
    tau_jitter: float = 0.3
    tau_drift: float = 300.0
    bias_std_deg: float = 0.012
    jitter_std_deg: float = 0.004
    win_sec: float = 300.0
    mpe_limit_deg: float = 0.04
    rpe_limit_deg: float = 0.028
    use_guiding: bool = True
    guiding_tau: float = 2000.0
    seed: int | None = 7


@dataclass
class PointingStats:
    t: np.ndarray
    angle_x_deg: np.ndarray
    angle_y_deg: np.ndarray
    rate_x_dps: np.ndarray
    rate_y_dps: np.ndarray
    mpe_p99_deg: float
    rpe_p99_deg: float
    global_p99_deg: float
    options: PointingOptions


def _ar1_unit_rms(n: int, dt: float, tau: float, rng: np.random.Generator) -> np.ndarray:
    a = np.exp(-dt / tau)
    s = np.sqrt(max(0.0, 1.0 - a * a))
    e = rng.normal(size=n)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = a * x[i - 1] + s * e[i]
    return x


def _moving_mean_valid(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, min(win, x.size))
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="valid")


def _moving_rms_valid(x: np.ndarray, win: int) -> np.ndarray:
    return np.sqrt(_moving_mean_valid(x * x, win))


def _windowed_relative_error_rms(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, min(win, x.size))
    kernel = np.ones(win, dtype=float) / win
    sum_x = np.convolve(x, kernel, mode="valid")
    sum_x2 = np.convolve(x * x, kernel, mode="valid")
    variance = np.maximum(0.0, sum_x2 - sum_x * sum_x)
    return np.sqrt(variance)


def _guided(angle_x: np.ndarray, angle_y: np.ndarray, dt: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.exp(-dt / tau)
    mx = np.zeros_like(angle_x)
    my = np.zeros_like(angle_y)
    for k in range(1, angle_x.size):
        mx[k] = alpha * mx[k - 1] + (1.0 - alpha) * angle_x[k - 1]
        my[k] = alpha * my[k - 1] + (1.0 - alpha) * angle_y[k - 1]
    return angle_x - mx, angle_y - my


def simulate_pointing(options: PointingOptions) -> PointingStats:
    rng = np.random.default_rng(options.seed)
    t = np.arange(0.0, options.duration + 0.5 * options.dt, options.dt)
    n = t.size

    angle_x_raw = (
        options.bias_std_deg * _ar1_unit_rms(n, options.dt, options.tau_drift, rng)
        + options.jitter_std_deg * _ar1_unit_rms(n, options.dt, options.tau_jitter, rng)
    )
    angle_y_raw = (
        options.bias_std_deg * _ar1_unit_rms(n, options.dt, options.tau_drift, rng)
        + options.jitter_std_deg * _ar1_unit_rms(n, options.dt, options.tau_jitter, rng)
    )

    if options.use_guiding:
        angle_x, angle_y = _guided(angle_x_raw, angle_y_raw, options.dt, options.guiding_tau)
    else:
        angle_x, angle_y = angle_x_raw, angle_y_raw

    rate_x = np.gradient(angle_x, options.dt)
    rate_y = np.gradient(angle_y, options.dt)
    win = max(1, round(options.win_sec / options.dt))

    mpe_series = np.abs(_moving_mean_valid(angle_x, win))
    rpe_series = _windowed_relative_error_rms(angle_x, win)

    return PointingStats(
        t=t,
        angle_x_deg=angle_x,
        angle_y_deg=angle_y,
        rate_x_dps=rate_x,
        rate_y_dps=rate_y,
        mpe_p99_deg=float(np.percentile(mpe_series, 99)),
        rpe_p99_deg=float(np.percentile(rpe_series, 99)),
        global_p99_deg=float(np.percentile(np.abs(angle_x), 99)),
        options=options,
    )


def rpe_p99_from_series(angle_deg: np.ndarray, dt: float, win_sec: float) -> float:
    win = max(1, round(win_sec / dt))
    rpe_series = _windowed_relative_error_rms(angle_deg, win)
    return float(np.percentile(rpe_series, 99))


def save_pointing_npz(stats: PointingStats, file: str | BinaryIO) -> None:
    opts = stats.options
    rpe_p99_deg = getattr(stats, "rpe_p99_deg", rpe_p99_from_series(stats.angle_x_deg, opts.dt, opts.win_sec))
    np.savez_compressed(
        file,
        t=stats.t,
        angle_x_deg=stats.angle_x_deg,
        angle_y_deg=stats.angle_y_deg,
        rate_x_dps=stats.rate_x_dps,
        rate_y_dps=stats.rate_y_dps,
        metrics=np.array([stats.mpe_p99_deg, rpe_p99_deg, stats.global_p99_deg]),
        options=np.array(
            [
                opts.dt,
                opts.duration,
                opts.tau_jitter,
                opts.tau_drift,
                opts.bias_std_deg,
                opts.jitter_std_deg,
                opts.win_sec,
                opts.mpe_limit_deg,
                opts.rpe_limit_deg,
                float(opts.use_guiding),
                opts.guiding_tau,
                -1 if opts.seed is None else opts.seed,
            ],
            dtype=float,
        ),
    )


def load_pointing_npz(file: str | BinaryIO) -> PointingStats:
    data = np.load(file)
    raw = data["options"]
    opts = PointingOptions(
        dt=float(raw[0]),
        duration=float(raw[1]),
        tau_jitter=float(raw[2]),
        tau_drift=float(raw[3]),
        bias_std_deg=float(raw[4]),
        jitter_std_deg=float(raw[5]),
        win_sec=float(raw[6]),
        mpe_limit_deg=float(raw[7]),
        rpe_limit_deg=float(raw[8]),
        use_guiding=bool(raw[9]),
        guiding_tau=float(raw[10]),
        seed=None if int(raw[11]) < 0 else int(raw[11]),
    )
    metrics = data["metrics"]
    return PointingStats(
        t=data["t"],
        angle_x_deg=data["angle_x_deg"],
        angle_y_deg=data["angle_y_deg"],
        rate_x_dps=data["rate_x_dps"],
        rate_y_dps=data["rate_y_dps"],
        mpe_p99_deg=float(metrics[0]),
        rpe_p99_deg=float(metrics[1]),
        global_p99_deg=float(metrics[2]),
        options=opts,
    )
