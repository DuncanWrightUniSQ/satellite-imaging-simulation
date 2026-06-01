from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from satellite_sim.alignment import align_target_centroid
from satellite_sim.pointing import PointingStats
from satellite_sim.rendering import (
    ImagingOptions,
    RenderOptions,
    SceneMeta,
    render_frame_sequence_from_scene,
    target_star_index,
)


@dataclass
class TimeseriesProduct:
    preview_stacks: np.ndarray
    metadata_csv: str
    zip_bytes: bytes | None
    output_dir: str | None
    n_outputs: int
    target_index: int | None


def parse_delta_magnitudes(text: str) -> list[float]:
    values: list[float] = []
    for raw in text.replace(",", " ").split():
        token = raw.strip()
        if not token or token.startswith("#"):
            continue
        values.append(float(token))
    return values


def float_tiff_bytes(frame: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.float32), mode="F").save(buf, format="TIFF", compression="tiff_lzw")
    return buf.getvalue()


def run_astronomical_timeseries(
    stats: PointingStats,
    imaging: ImagingOptions,
    render: RenderOptions,
    scene: SceneMeta,
    *,
    total_duration: float,
    frames_per_stack: int,
    max_outputs: int,
    delta_magnitudes: list[float] | None,
    search_box_size: int,
    threshold_sigma: float,
    shift_mode: str,
    stack_method: str,
    output_dir: str | None,
    make_zip: bool,
    preview_limit: int = 12,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TimeseriesProduct:
    frames_per_stack = max(1, int(frames_per_stack))
    cadence = frames_per_stack * float(render.frame_exposure)
    n_by_duration = max(1, int(np.floor(float(total_duration) / cadence + 1e-9)))
    n_outputs = min(n_by_duration, max(1, int(max_outputs)))

    if delta_magnitudes:
        if len(delta_magnitudes) != n_outputs:
            raise ValueError(f"Delta magnitude list has {len(delta_magnitudes)} values, but this run will create {n_outputs} outputs.")
        deltas = delta_magnitudes
    else:
        deltas = [0.0] * n_outputs

    out_path: Path | None = None
    if output_dir:
        out_path = Path(output_dir).expanduser()
        out_path.mkdir(parents=True, exist_ok=True)

    target_idx = target_star_index(scene, imaging)
    rng = np.random.default_rng(98765)
    preview: list[np.ndarray] = []
    rows = ["index,start_time_s,mid_time_s,delta_mag,centroid_found_fraction,mean_shift_x_px,mean_shift_y_px,file"]

    zip_buf = BytesIO() if make_zip else None
    zip_file = ZipFile(zip_buf, "w", compression=ZIP_DEFLATED) if zip_buf is not None else None

    try:
        for i in range(n_outputs):
            start_time = render.start_time + i * cadence
            frames = render_frame_sequence_from_scene(
                stats,
                imaging,
                render,
                scene,
                cadence,
                frames_per_stack,
                start_time=start_time,
                target_delta_mag=deltas[i],
                target_index=target_idx,
                rng=rng,
            )
            result = align_target_centroid(
                frames,
                reference_index=0,
                search_box_size=search_box_size,
                threshold_sigma=threshold_sigma,
                shift_mode=shift_mode,
                stack_method=stack_method,
            )
            stack = result.stack.astype(np.float32, copy=False)
            filename = f"astro_stack_{i:05d}.tiff"
            tiff_data = float_tiff_bytes(stack)

            if out_path is not None:
                (out_path / filename).write_bytes(tiff_data)
            if zip_file is not None:
                zip_file.writestr(filename, tiff_data)
            if len(preview) < preview_limit:
                preview.append(stack.copy())

            rows.append(
                ",".join(
                    [
                        str(i),
                        f"{start_time:.9g}",
                        f"{start_time + 0.5 * cadence:.9g}",
                        f"{deltas[i]:.9g}",
                        f"{float(np.mean(result.found)):.6g}",
                        f"{float(np.mean(result.shifts_xy[:, 0])):.9g}",
                        f"{float(np.mean(result.shifts_xy[:, 1])):.9g}",
                        filename,
                    ]
                )
            )
            if progress_callback is not None:
                progress_callback(i + 1, n_outputs)

        metadata_csv = "\n".join(rows) + "\n"
        if out_path is not None:
            (out_path / "astro_timeseries_metadata.csv").write_text(metadata_csv, encoding="utf-8")
        if zip_file is not None:
            zip_file.writestr("astro_timeseries_metadata.csv", metadata_csv)
    finally:
        if zip_file is not None:
            zip_file.close()

    preview_array = np.stack(preview, axis=0) if preview else np.empty((0, 0, 0), dtype=np.float32)
    return TimeseriesProduct(
        preview_stacks=preview_array,
        metadata_csv=metadata_csv,
        zip_bytes=None if zip_buf is None else zip_buf.getvalue(),
        output_dir=str(out_path) if out_path is not None else None,
        n_outputs=n_outputs,
        target_index=target_idx,
    )
