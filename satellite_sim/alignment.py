from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from satellite_sim.visuals import add_arcsec_scale_bar, frame_to_image


@dataclass
class AlignmentResult:
    original_frames: np.ndarray
    aligned_frames: np.ndarray
    stack: np.ndarray
    centroids_xy: np.ndarray
    shifts_xy: np.ndarray
    found: np.ndarray


def gif_bytes_to_frames(data: bytes, max_frames: int | None = None) -> np.ndarray:
    frames: list[np.ndarray] = []
    with Image.open(BytesIO(data)) as img:
        n = getattr(img, "n_frames", 1)
        limit = n if max_frames is None else min(n, int(max_frames))
        for i in range(limit):
            img.seek(i)
            gray = img.convert("L")
            frames.append(np.asarray(gray, dtype=float))
    if not frames:
        raise ValueError("No frames found in GIF.")
    return np.stack(frames, axis=0)


def robust_centroid(
    frame: np.ndarray,
    center_xy: tuple[float, float] | None,
    box_size: int,
    threshold_sigma: float,
) -> tuple[float, float, bool]:
    h, w = frame.shape
    if center_xy is None:
        y0, x0 = np.unravel_index(int(np.argmax(frame)), frame.shape)
        cx, cy = float(x0), float(y0)
    else:
        cx, cy = center_xy

    half = max(2, int(box_size // 2))
    x_min = max(0, int(round(cx)) - half)
    x_max = min(w, int(round(cx)) + half + 1)
    y_min = max(0, int(round(cy)) - half)
    y_max = min(h, int(round(cy)) + half + 1)
    patch = frame[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return cx, cy, False

    bg = float(np.percentile(patch, 30.0))
    resid = patch - bg
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = max(1e-6, 1.4826 * mad)
    weights = np.maximum(patch - (bg + threshold_sigma * sigma), 0.0)

    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0:
        yy, xx = np.unravel_index(int(np.argmax(patch)), patch.shape)
        return float(x_min + xx), float(y_min + yy), False

    yy, xx = np.indices(patch.shape)
    total = float(np.sum(weights))
    x_cent = x_min + float(np.sum(xx * weights) / total)
    y_cent = y_min + float(np.sum(yy * weights) / total)
    return x_cent, y_cent, True


def shift_integer(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    sx = int(round(dx))
    sy = int(round(dy))
    h, w = frame.shape
    out = np.zeros_like(frame)

    src_x0 = max(0, -sx)
    src_x1 = min(w, w - sx)
    dst_x0 = max(0, sx)
    dst_x1 = min(w, w + sx)
    src_y0 = max(0, -sy)
    src_y1 = min(h, h - sy)
    dst_y0 = max(0, sy)
    dst_y1 = min(h, h + sy)

    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


def shift_bilinear(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = frame.shape
    yy, xx = np.indices((h, w), dtype=float)
    src_x = xx - dx
    src_y = yy - dy
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    out = np.zeros_like(frame, dtype=float)
    if not np.any(valid):
        return out

    wx = src_x - x0
    wy = src_y - y0
    v00 = frame[y0[valid], x0[valid]]
    v10 = frame[y0[valid], x1[valid]]
    v01 = frame[y1[valid], x0[valid]]
    v11 = frame[y1[valid], x1[valid]]
    out[valid] = (
        (1.0 - wx[valid]) * (1.0 - wy[valid]) * v00
        + wx[valid] * (1.0 - wy[valid]) * v10
        + (1.0 - wx[valid]) * wy[valid] * v01
        + wx[valid] * wy[valid] * v11
    )
    return out


def stack_frames(frames: np.ndarray, method: str) -> np.ndarray:
    if method == "sum":
        return np.sum(frames, axis=0)
    if method == "max":
        return np.max(frames, axis=0)
    return np.mean(frames, axis=0)


def align_target_centroid(
    frames: np.ndarray,
    reference_index: int,
    search_box_size: int,
    threshold_sigma: float,
    shift_mode: str,
    stack_method: str,
) -> AlignmentResult:
    if frames.ndim != 3:
        raise ValueError("frames must have shape (n_frames, height, width)")
    n, h, w = frames.shape
    if n == 0:
        raise ValueError("at least one frame is required")

    ref_idx = int(np.clip(reference_index, 0, n - 1))
    ref_centroid = robust_centroid(frames[ref_idx], None, search_box_size, threshold_sigma)[:2]
    target_xy = ((w - 1) / 2.0, (h - 1) / 2.0)

    centroids = np.zeros((n, 2), dtype=float)
    shifts = np.zeros((n, 2), dtype=float)
    found = np.zeros(n, dtype=bool)
    aligned = np.zeros_like(frames, dtype=float)
    previous = ref_centroid

    shift_fn = shift_integer if shift_mode == "integer" else shift_bilinear
    for i in range(n):
        center_hint = ref_centroid if i == ref_idx else previous
        cx, cy, ok = robust_centroid(frames[i], center_hint, search_box_size, threshold_sigma)
        centroids[i] = (cx, cy)
        found[i] = ok
        previous = (cx, cy)
        dx = target_xy[0] - cx
        dy = target_xy[1] - cy
        shifts[i] = (dx, dy)
        aligned[i] = shift_fn(frames[i], dx, dy)

    return AlignmentResult(
        original_frames=frames,
        aligned_frames=aligned,
        stack=stack_frames(aligned, stack_method),
        centroids_xy=centroids,
        shifts_xy=shifts,
        found=found,
    )


def overlay_centroid_png(
    frame: np.ndarray,
    centroid_xy: tuple[float, float],
    render,
    plate_arcsec_per_pix: float | None = None,
) -> bytes:
    img = frame_to_image(frame, render)
    if plate_arcsec_per_pix is not None:
        img = add_arcsec_scale_bar(img, plate_arcsec_per_pix)
    rgb = img.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    x, y = centroid_xy
    r = 9
    draw.line([(x - r, y), (x + r, y)], fill=(255, 70, 40), width=2)
    draw.line([(x, y - r), (x, y + r)], fill=(255, 70, 40), width=2)
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 70, 40), width=2)
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()


def frame_png(frame: np.ndarray, render, plate_arcsec_per_pix: float | None = None) -> bytes:
    img = frame_to_image(frame, render)
    if plate_arcsec_per_pix is not None:
        img = add_arcsec_scale_bar(img, plate_arcsec_per_pix)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def frames_to_gif_bytes(frames: np.ndarray, render, fps: float, plate_arcsec_per_pix: float | None = None) -> bytes:
    images: list[Image.Image] = []
    limits: tuple[float, float] | None = None
    for frame in frames:
        if limits is None and render.stretch == "percentile":
            limits = (float(np.percentile(frame, render.percentile_low)), float(np.percentile(frame, render.percentile_high)))
        elif render.stretch == "fixed":
            limits = render.fixed_clim
        img = frame_to_image(frame, render, limits)
        if plate_arcsec_per_pix is not None:
            img = add_arcsec_scale_bar(img, plate_arcsec_per_pix)
        images.append(img.convert("P", palette=Image.Palette.ADAPTIVE))

    buf = BytesIO()
    duration_ms = int(round(1000.0 / max(fps, 1e-6)))
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return buf.getvalue()
