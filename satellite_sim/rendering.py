from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO

import numpy as np
from PIL import Image

from satellite_sim.catalog import CatalogOptions, StarCatalog, query_catalog, small_angle_offsets
from satellite_sim.pointing import PointingStats
from satellite_sim.visuals import frame_to_image


@dataclass(frozen=True)
class ImagingOptions:
    aperture_m: float = 0.20
    f_number: float = 8.0
    airy_fwhm_um: float = 11.0
    pixel_pitch_um: float = 4.6
    nx: int = 1600
    ny: int = 1200
    qe: float = 0.80
    read_noise: float = 0.9
    sky_mag: float = 22.0


@dataclass(frozen=True)
class TargetOptions:
    ra_deg: float = 84.0
    dec_deg: float = -5.0
    band: str = "V"
    mag_limit: float = 18.0
    catalog: CatalogOptions = CatalogOptions()


@dataclass(frozen=True)
class RenderOptions:
    frame_exposure: float = 0.1
    half_subframe: int = 240
    start_time: float = 0.0
    stretch: str = "percentile"
    percentile_low: float = 1.0
    percentile_high: float = 99.7
    fixed_clim: tuple[float, float] = (-2.0, 35.0)


@dataclass(frozen=True)
class GifOptions:
    duration: float = 3.0
    fps: float = 10.0
    max_frames: int = 60


@dataclass
class SceneMeta:
    stars: StarCatalog
    catalog_name: str
    plate_arcsec_per_pix: float
    background_e_per_pix_per_s: float
    x0_pix: np.ndarray
    y0_pix: np.ndarray
    e_star_per_s: np.ndarray
    sigma_pix: float
    half_win: int


def vband_phi0() -> float:
    return 995.5 * 840.0 * 1e4


def vmag_to_photons_per_m2s(mag: np.ndarray) -> np.ndarray:
    return vband_phi0() * np.power(10.0, -0.4 * mag)


def build_scene(imaging: ImagingOptions, target: TargetOptions) -> SceneMeta:
    focal_m = imaging.f_number * imaging.aperture_m
    plate = 206265.0 * (imaging.pixel_pitch_um * 1e-6) / focal_m
    fov_x_deg = imaging.nx * plate / 3600.0
    fov_y_deg = imaging.ny * plate / 3600.0
    radius_deg = 0.5 * float(np.hypot(fov_x_deg, fov_y_deg)) + 0.01

    stars = query_catalog(target.ra_deg, target.dec_deg, radius_deg, target.mag_limit, target.band, target.catalog)

    area_m2 = np.pi * (imaging.aperture_m / 2.0) ** 2
    e_star_per_s = vmag_to_photons_per_m2s(stars.mag) * area_m2 * imaging.qe

    dx_as, dy_as = small_angle_offsets(target.ra_deg, target.dec_deg, stars.ra_deg, stars.dec_deg)
    x0_pix = imaging.nx / 2.0 + dx_as / plate
    y0_pix = imaging.ny / 2.0 + dy_as / plate

    fwhm_pix = imaging.airy_fwhm_um / imaging.pixel_pitch_um
    sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half_win = max(3, int(np.ceil(5.0 * sigma_pix)))

    phi_bg_as2 = vband_phi0() * 10.0 ** (-0.4 * imaging.sky_mag)
    pix_as2 = plate * plate
    bg_rate = phi_bg_as2 * pix_as2 * area_m2 * imaging.qe

    return SceneMeta(stars, stars.name, plate, bg_rate, x0_pix, y0_pix, e_star_per_s, sigma_pix, half_win)


def target_star_index(scene: SceneMeta, imaging: ImagingOptions) -> int | None:
    if scene.x0_pix.size == 0:
        return None
    dx = scene.x0_pix - imaging.nx / 2.0
    dy = scene.y0_pix - imaging.ny / 2.0
    return int(np.argmin(dx * dx + dy * dy))


def _crop_bounds(imaging: ImagingOptions, half_subframe: int) -> tuple[np.ndarray, np.ndarray]:
    cx = int(round(imaging.nx / 2.0))
    cy = int(round(imaging.ny / 2.0))
    ix = np.arange(max(0, cx - half_subframe), min(imaging.nx, cx + half_subframe + 1))
    iy = np.arange(max(0, cy - half_subframe), min(imaging.ny, cy + half_subframe + 1))
    return ix, iy


def _render_expected_cropped(
    ix_crop: np.ndarray,
    iy_crop: np.ndarray,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    e_star: np.ndarray,
    sigma_pix: float,
    half_win: int,
) -> np.ndarray:
    img = np.zeros((iy_crop.size, ix_crop.size), dtype=float)
    xlo, xhi = int(ix_crop[0]), int(ix_crop[-1])
    ylo, yhi = int(iy_crop[0]), int(iy_crop[-1])

    for x0, y0, e0 in zip(x_pix, y_pix, e_star, strict=False):
        if e0 <= 0:
            continue
        ixg_lo = max(int(np.floor(x0 - half_win)), xlo)
        ixg_hi = min(int(np.ceil(x0 + half_win)), xhi)
        iyg_lo = max(int(np.floor(y0 - half_win)), ylo)
        iyg_hi = min(int(np.ceil(y0 + half_win)), yhi)
        if ixg_hi < ixg_lo or iyg_hi < iyg_lo:
            continue

        ixg = np.arange(ixg_lo, ixg_hi + 1)
        iyg = np.arange(iyg_lo, iyg_hi + 1)
        xx, yy = np.meshgrid(ixg, iyg)
        g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma_pix * sigma_pix))
        total = g.sum()
        if total <= 0:
            continue
        g /= total
        img[np.ix_(iyg - ylo, ixg - xlo)] += e0 * g
    return img


def _shortest_pointing_timescale(stats: PointingStats) -> float:
    candidates = [
        stats.options.tau_jitter,
        stats.options.tau_drift,
    ]
    if stats.options.use_guiding:
        candidates.append(stats.options.guiding_tau)
    positive = [float(v) for v in candidates if v > 0]
    return min(positive) if positive else float(np.median(np.diff(stats.t)))


def _exposure_segments(
    stats: PointingStats,
    start_time: float,
    exposure: float,
    scene: SceneMeta,
    max_motion_pix: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointing_dt = float(np.median(np.diff(stats.t)))
    base_dt = min(pointing_dt, _shortest_pointing_timescale(stats) / 20.0)
    base_dt = max(base_dt, max(exposure / 2000.0, 1e-5))

    t0 = float(np.clip(start_time, stats.t[0], stats.t[-1]))
    t1 = float(np.clip(start_time + exposure, t0, stats.t[-1]))
    if t1 <= t0:
        t1 = min(stats.t[-1], t0 + pointing_dt)
    if t1 <= t0:
        t1 = t0

    n_base = max(1, int(np.ceil((t1 - t0) / base_dt)))
    edges = np.linspace(t0, t1, n_base + 1)
    x_edges = np.interp(edges, stats.t, stats.angle_x_deg) * 3600.0 / scene.plate_arcsec_per_pix
    y_edges = np.interp(edges, stats.t, stats.angle_y_deg) * 3600.0 / scene.plate_arcsec_per_pix

    sample_times: list[float] = []
    weights: list[float] = []
    for i in range(n_base):
        seg_dt = float(edges[i + 1] - edges[i])
        motion = float(np.hypot(x_edges[i + 1] - x_edges[i], y_edges[i + 1] - y_edges[i]))
        n_split = max(1, int(np.ceil(motion / max_motion_pix)))
        n_split = min(n_split, 200)
        sub_dt = seg_dt / n_split
        for j in range(n_split):
            sample_times.append(float(edges[i] + (j + 0.5) * sub_dt))
            weights.append(sub_dt)

    if not sample_times:
        sample_times = [t0]
        weights = [0.0]

    sample_times_arr = np.array(sample_times, dtype=float)
    weights_arr = np.array(weights, dtype=float)
    dx_pix = np.interp(sample_times_arr, stats.t, stats.angle_x_deg) * 3600.0 / scene.plate_arcsec_per_pix
    dy_pix = np.interp(sample_times_arr, stats.t, stats.angle_y_deg) * 3600.0 / scene.plate_arcsec_per_pix
    return dx_pix, dy_pix, weights_arr


def _render_frame_integrated(
    stats: PointingStats,
    imaging: ImagingOptions,
    render: RenderOptions,
    scene: SceneMeta,
    frame_start_time: float,
    rng: np.random.Generator,
) -> np.ndarray:
    ix_crop, iy_crop = _crop_bounds(imaging, render.half_subframe)
    frame_e = np.zeros((iy_crop.size, ix_crop.size), dtype=float)
    dx_pix, dy_pix, weights = _exposure_segments(stats, frame_start_time, render.frame_exposure, scene)
    for dx, dy, weight in zip(dx_pix, dy_pix, weights, strict=False):
        frame_e += _render_expected_cropped(
            ix_crop,
            iy_crop,
            scene.x0_pix + dx,
            scene.y0_pix + dy,
            scene.e_star_per_s * weight,
            scene.sigma_pix,
            scene.half_win,
        )
    frame_e += scene.background_e_per_pix_per_s * float(np.sum(weights))
    noisy = rng.poisson(np.maximum(frame_e, 0.0)).astype(float)
    noisy += imaging.read_noise * rng.normal(size=noisy.shape)
    return noisy


def render_preview_frame(
    stats: PointingStats,
    imaging: ImagingOptions,
    target: TargetOptions,
    render: RenderOptions,
) -> tuple[np.ndarray, SceneMeta]:
    scene = build_scene(imaging, target)
    rng = np.random.default_rng(12345)
    return _render_frame_integrated(stats, imaging, render, scene, render.start_time, rng), scene


def render_frame_sequence(
    stats: PointingStats,
    imaging: ImagingOptions,
    target: TargetOptions,
    render: RenderOptions,
    duration: float,
    max_frames: int,
) -> tuple[np.ndarray, SceneMeta]:
    scene = build_scene(imaging, target)
    frames = render_frame_sequence_from_scene(stats, imaging, render, scene, duration, max_frames)
    return frames, scene


def render_frame_sequence_from_scene(
    stats: PointingStats,
    imaging: ImagingOptions,
    render: RenderOptions,
    scene: SceneMeta,
    duration: float,
    max_frames: int,
    *,
    start_time: float | None = None,
    target_delta_mag: float = 0.0,
    target_index: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    requested_frames = max(1, int(round(duration / render.frame_exposure)))
    n_frames = min(requested_frames, max(1, int(max_frames)))
    render_start = render.start_time if start_time is None else float(start_time)
    available = max(0.0, float(stats.t[-1] - render_start))
    max_possible = max(1, int(np.floor(available / render.frame_exposure)))
    n_frames = min(n_frames, max_possible)

    render_scene = scene
    if target_delta_mag != 0.0:
        idx = target_star_index(scene, imaging) if target_index is None else target_index
        if idx is not None and 0 <= idx < scene.e_star_per_s.size:
            e_star_per_s = scene.e_star_per_s.copy()
            e_star_per_s[idx] *= 10.0 ** (-0.4 * float(target_delta_mag))
            render_scene = replace(scene, e_star_per_s=e_star_per_s)

    if rng is None:
        rng = np.random.default_rng(12345)
    frames = []
    for k in range(n_frames):
        frame_start = render_start + k * render.frame_exposure
        frames.append(_render_frame_integrated(stats, imaging, render, render_scene, frame_start, rng))
    return np.asarray(frames, dtype=float)


def make_starfield_gif(
    stats: PointingStats,
    imaging: ImagingOptions,
    target: TargetOptions,
    render: RenderOptions,
    gif: GifOptions,
) -> tuple[bytes, SceneMeta]:
    frame_array, scene = render_frame_sequence(stats, imaging, target, render, gif.duration, gif.max_frames)
    return frames_to_gif(frame_array, render, gif.fps), scene


def frames_to_gif(frame_array: np.ndarray, render: RenderOptions, fps: float) -> bytes:
    frames: list[Image.Image] = []
    fixed_limits: tuple[float, float] | None = None
    for frame in frame_array:
        if render.stretch == "fixed":
            fixed_limits = render.fixed_clim
        elif fixed_limits is None:
            fixed_limits = (float(np.percentile(frame, render.percentile_low)), float(np.percentile(frame, render.percentile_high)))
        frames.append(frame_to_image(frame, render, fixed_limits).convert("P", palette=Image.Palette.ADAPTIVE))

    buf = BytesIO()
    duration_ms = int(round(1000.0 / max(fps, 1e-6)))
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return buf.getvalue()
