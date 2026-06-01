from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from satellite_sim.pointing import PointingStats


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _scale(values: np.ndarray, lo: float, hi: float, pixels_lo: int, pixels_hi: int, invert: bool = False) -> np.ndarray:
    if hi == lo:
        hi = lo + 1.0
    p = pixels_lo + (values - lo) * (pixels_hi - pixels_lo) / (hi - lo)
    if invert:
        p = pixels_hi - (p - pixels_lo)
    return p


def _nice_ticks(lo: float, hi: float, count: int = 7) -> np.ndarray:
    if hi <= lo:
        return np.array([lo])
    raw_step = (hi - lo) / max(1, count - 1)
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1.5:
        nice_step = 1.0 * magnitude
    elif residual <= 3.0:
        nice_step = 2.0 * magnitude
    elif residual <= 7.0:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude
    start = np.ceil(lo / nice_step) * nice_step
    stop = np.floor(hi / nice_step) * nice_step
    ticks = np.arange(start, stop + 0.5 * nice_step, nice_step)
    if ticks.size < 3:
        ticks = np.linspace(lo, hi, count)
    return ticks


def _fmt_tick(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _draw_y_label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    label = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((4, 4), text, fill=(5, 10, 18), font=font)
    rotated = label.rotate(90, expand=True)
    draw.bitmap(xy, rotated, fill=(5, 10, 18))


def trajectory_png(stats: PointingStats, width: int = 1100, height: int = 620) -> bytes:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(18)
    title_font = _font(26)
    margin_l, margin_r, margin_t, margin_b = 90, 35, 70, 70
    x = stats.angle_x_deg * 3600.0
    y = stats.angle_y_deg * 3600.0
    extent = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0)
    extent *= 1.08
    xpix = _scale(x, -extent, extent, margin_l, width - margin_r)
    ypix = _scale(y, -extent, extent, margin_t, height - margin_b, invert=True)

    x_ticks = _nice_ticks(-extent, extent, 9)
    y_ticks = _nice_ticks(-extent, extent, 9)
    for tick in x_ticks:
        gx = int(_scale(np.array([tick]), -extent, extent, margin_l, width - margin_r)[0])
        draw.line([(gx, margin_t), (gx, height - margin_b)], fill=(220, 225, 230))
        draw.line([(gx, height - margin_b), (gx, height - margin_b + 5)], fill=(40, 45, 50))
        label = _fmt_tick(tick)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((gx - (box[2] - box[0]) // 2, height - margin_b + 10), label, fill=(60, 65, 70), font=font)
    for tick in y_ticks:
        gy = int(_scale(np.array([tick]), -extent, extent, margin_t, height - margin_b, invert=True)[0])
        draw.line([(margin_l, gy), (width - margin_r, gy)], fill=(220, 225, 230))
        draw.line([(margin_l - 5, gy), (margin_l, gy)], fill=(40, 45, 50))
        label = _fmt_tick(tick)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((margin_l - 12 - (box[2] - box[0]), gy - (box[3] - box[1]) // 2), label, fill=(60, 65, 70), font=font)
    draw.rectangle([margin_l, margin_t, width - margin_r, height - margin_b], outline=(40, 45, 50))

    step = max(1, len(xpix) // 12000)
    pts = list(zip(xpix[::step].astype(int), ypix[::step].astype(int), strict=False))
    if len(pts) > 1:
        draw.line(pts, fill=(0, 125, 185), width=1)
    draw.ellipse([xpix[0] - 5, ypix[0] - 5, xpix[0] + 5, ypix[0] + 5], fill=(35, 210, 55))
    draw.ellipse([xpix[-1] - 5, ypix[-1] - 5, xpix[-1] + 5, ypix[-1] + 5], fill=(220, 20, 25))
    draw.text((margin_l, 20), "Pointing trajectory", fill=(5, 10, 18), font=title_font)
    draw.text((width // 2 - 55, height - 42), "X (arcsec)", fill=(5, 10, 18), font=font)
    _draw_y_label(draw, "Y (arcsec)", (24, height // 2 - 55), font)
    return _png_bytes(img)


def timeseries_png(stats: PointingStats, width: int = 1100, height: int = 340) -> bytes:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(16)
    title_font = _font(22)
    margin_l, margin_r, margin_t, margin_b = 90, 35, 55, 55
    t = stats.t
    x = stats.angle_x_deg * 3600.0
    y = stats.angle_y_deg * 3600.0
    extent = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0) * 1.08
    tpix = _scale(t, float(t[0]), float(t[-1]), margin_l, width - margin_r)
    xpix = _scale(x, -extent, extent, margin_t, height - margin_b, invert=True)
    ypix = _scale(y, -extent, extent, margin_t, height - margin_b, invert=True)

    x_ticks = _nice_ticks(float(t[0]), float(t[-1]), 7)
    y_ticks = _nice_ticks(-extent, extent, 7)
    for tick in y_ticks:
        gy = int(_scale(np.array([tick]), -extent, extent, margin_t, height - margin_b, invert=True)[0])
        draw.line([(margin_l, gy), (width - margin_r, gy)], fill=(225, 228, 232))
        draw.line([(margin_l - 5, gy), (margin_l, gy)], fill=(40, 45, 50))
        label = _fmt_tick(tick)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((margin_l - 12 - (box[2] - box[0]), gy - (box[3] - box[1]) // 2), label, fill=(60, 65, 70), font=font)
    for tick in x_ticks:
        gx = int(_scale(np.array([tick]), float(t[0]), float(t[-1]), margin_l, width - margin_r)[0])
        draw.line([(gx, margin_t), (gx, height - margin_b)], fill=(235, 237, 240))
        draw.line([(gx, height - margin_b), (gx, height - margin_b + 5)], fill=(40, 45, 50))
        label = _fmt_tick(tick)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((gx - (box[2] - box[0]) // 2, height - margin_b + 10), label, fill=(60, 65, 70), font=font)
    draw.rectangle([margin_l, margin_t, width - margin_r, height - margin_b], outline=(40, 45, 50))
    step = max(1, len(tpix) // 3000)
    draw.line(list(zip(tpix[::step].astype(int), xpix[::step].astype(int), strict=False)), fill=(0, 120, 190), width=2)
    draw.line(list(zip(tpix[::step].astype(int), ypix[::step].astype(int), strict=False)), fill=(210, 80, 45), width=2)
    draw.text((margin_l, 18), "Pointing error over time", fill=(5, 10, 18), font=title_font)
    draw.text((margin_l + 10, margin_t + 8), "X", fill=(0, 120, 190), font=font)
    draw.text((margin_l + 42, margin_t + 8), "Y", fill=(210, 80, 45), font=font)
    draw.text((width // 2 - 45, height - 38), "Time (s)", fill=(5, 10, 18), font=font)
    _draw_y_label(draw, "arcsec", (30, height // 2 - 34), font)
    return _png_bytes(img)


def frame_to_image(frame: np.ndarray, render, limits: tuple[float, float] | None = None) -> Image.Image:
    if limits is None:
        if render.stretch == "fixed":
            lo, hi = render.fixed_clim
        else:
            lo = float(np.percentile(frame, render.percentile_low))
            hi = float(np.percentile(frame, render.percentile_high))
    else:
        lo, hi = limits
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
    return Image.fromarray(np.uint8(scaled * 255.0), mode="L")


def _nice_scale_length_arcsec(width_arcsec: float) -> float:
    target = max(width_arcsec / 5.0, 1e-9)
    magnitude = 10.0 ** np.floor(np.log10(target))
    residual = target / magnitude
    if residual < 1.5:
        nice = 1.0
    elif residual < 3.5:
        nice = 2.0
    elif residual < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


def add_arcsec_scale_bar(img: Image.Image, plate_arcsec_per_pix: float) -> Image.Image:
    if plate_arcsec_per_pix <= 0:
        return img
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    width, height = out.size
    scale_arcsec = _nice_scale_length_arcsec(width * plate_arcsec_per_pix)
    bar_px = int(round(scale_arcsec / plate_arcsec_per_pix))
    max_bar_px = max(12, width // 3)
    if bar_px > max_bar_px:
        bar_px = max_bar_px
        scale_arcsec = bar_px * plate_arcsec_per_pix
    bar_px = max(8, bar_px)

    font = _font(max(12, min(18, width // 32)))
    label = f"{_fmt_tick(scale_arcsec)} arcsec"
    box = draw.textbbox((0, 0), label, font=font)
    label_w = box[2] - box[0]
    label_h = box[3] - box[1]

    pad = max(10, width // 45)
    x0 = pad
    y0 = height - pad - label_h - 13
    x1 = x0 + bar_px
    y1 = y0 + 8
    bg = [x0 - 7, y0 - 7, max(x1, x0 + label_w) + 7, y0 + label_h + 24]
    draw.rounded_rectangle(bg, radius=4, fill=(0, 0, 0), outline=(235, 235, 235))
    draw.line([(x0, y1), (x1, y1)], fill=(255, 255, 255), width=3)
    draw.line([(x0, y1 - 5), (x0, y1 + 5)], fill=(255, 255, 255), width=2)
    draw.line([(x1, y1 - 5), (x1, y1 + 5)], fill=(255, 255, 255), width=2)
    draw.text((x0, y1 + 7), label, fill=(255, 255, 255), font=font)
    return out


def image_to_png_bytes(frame: np.ndarray, render, plate_arcsec_per_pix: float | None = None) -> bytes:
    img = frame_to_image(frame, render)
    if plate_arcsec_per_pix is not None:
        img = add_arcsec_scale_bar(img, plate_arcsec_per_pix)
    return _png_bytes(img)
