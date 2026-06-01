from __future__ import annotations

from dataclasses import replace as dc_replace
from io import BytesIO
import pandas as pd
import streamlit as st

from satellite_sim.alignment import align_target_centroid, frame_png, frames_to_gif_bytes, gif_bytes_to_frames, overlay_centroid_png
from satellite_sim.catalog import CatalogOptions
from satellite_sim.pointing import PointingOptions, load_pointing_npz, rpe_p99_from_series, save_pointing_npz, simulate_pointing
from satellite_sim.rendering import (
    GifOptions,
    ImagingOptions,
    RenderOptions,
    TargetOptions,
    build_scene,
    frames_to_gif,
    render_frame_sequence,
    render_preview_frame,
)
from satellite_sim.timeseries import parse_delta_magnitudes, run_astronomical_timeseries
from satellite_sim.visuals import image_to_png_bytes, trajectory_png, timeseries_png


st.set_page_config(page_title="Satellite Imaging Simulator", page_icon="*", layout="wide")


def _npz_download(stats) -> bytes:
    buf = BytesIO()
    save_pointing_npz(stats, buf)
    return buf.getvalue()


def _metrics_table(stats) -> pd.DataFrame:
    rpe_p99_deg = _current_rpe_p99_deg(stats)
    return pd.DataFrame(
        {
            "Metric": ["99% MPE", "99% RPE", "99% |angle| global"],
            "Value (deg)": [stats.mpe_p99_deg, rpe_p99_deg, stats.global_p99_deg],
            "Value (arcsec)": [stats.mpe_p99_deg * 3600.0, rpe_p99_deg * 3600.0, stats.global_p99_deg * 3600.0],
        }
    )


def _current_rpe_p99_deg(stats) -> float:
    if hasattr(stats, "rpe_p99_deg"):
        return stats.rpe_p99_deg
    rpe = rpe_p99_from_series(stats.angle_x_deg, stats.options.dt, stats.options.win_sec)
    stats.rpe_p99_deg = rpe
    return rpe


def _deg_caption(value: float) -> str:
    return f"{value:.5f} deg = {value * 3600.0:.2f} arcsec"


def _parse_angle_parts(value: str, *, is_ra: bool) -> float:
    cleaned = value.strip().replace(":", " ")
    parts = [p for p in cleaned.split() if p]
    if len(parts) == 1:
        angle = float(parts[0])
        return angle
    if len(parts) != 3:
        kind = "RA" if is_ra else "Dec"
        raise ValueError(f"{kind} must be decimal degrees or three fields like HH MM SS.SS / DD MM SS.SS.")

    first_s, minute_s, second_s = parts
    sign = -1.0 if first_s.startswith("-") else 1.0
    first = abs(float(first_s))
    minutes = float(minute_s)
    seconds = float(second_s)
    if minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
        raise ValueError("Minutes and seconds must be in the range [0, 60).")
    angle = sign * (first + minutes / 60.0 + seconds / 3600.0)
    if is_ra:
        angle *= 15.0
    return angle


def _parse_ra(value: str) -> float:
    ra = _parse_angle_parts(value, is_ra=True)
    if not 0.0 <= ra < 360.0:
        raise ValueError("RA must be in [0, 360) degrees, or [0, 24) hours.")
    return ra


def _parse_dec(value: str) -> float:
    dec = _parse_angle_parts(value, is_ra=False)
    if not -90.0 <= dec <= 90.0:
        raise ValueError("Dec must be in [-90, 90] degrees.")
    return dec


def _band_code(label: str) -> str:
    return label.split(" ", 1)[0]


def _plate_scale_arcsec_per_pix(aperture_m: float, f_number: float, pixel_pitch_um: float) -> float:
    focal_m = aperture_m * f_number
    return 206265.0 * (pixel_pitch_um * 1e-6) / focal_m


def _rayleigh_arcsec(aperture_m: float, wavelength_nm: float = 550.0) -> float:
    return 1.22 * wavelength_nm * 1e-9 / aperture_m * 206265.0


METRIC_WINDOW_HELP = (
    "The rolling time interval used when computing the reported pointing metrics. "
    "MPE is the 99th percentile of the absolute rolling mean pointing error over this window. "
    "RPE is the 99th percentile of the rolling RMS residual after subtracting that same window mean."
)


st.title("Satellite Pointing and Starfield Simulator")

with st.sidebar:
    st.header("Workflow")
    mode = st.radio("Pointing source", ["Simulate new pointing", "Load previous pointing"], index=0)

tabs = st.tabs(
    [
        "Pointing",
        "Imaging Payload",
        "Target Field",
        "Render & Download",
        "Alignment & Stacking",
        "Astronomical Timeseries",
    ]
)

if "pointing_stats" not in st.session_state:
    st.session_state.pointing_stats = None
if "last_gif" not in st.session_state:
    st.session_state.last_gif = None
if "last_frame_png" not in st.session_state:
    st.session_state.last_frame_png = None
if "alignment_result" not in st.session_state:
    st.session_state.alignment_result = None
if "alignment_meta" not in st.session_state:
    st.session_state.alignment_meta = None
if "aligned_gif" not in st.session_state:
    st.session_state.aligned_gif = None
if "aligned_stack_png" not in st.session_state:
    st.session_state.aligned_stack_png = None
if "last_render_frames" not in st.session_state:
    st.session_state.last_render_frames = None
if "last_render_meta" not in st.session_state:
    st.session_state.last_render_meta = None
if "astro_product" not in st.session_state:
    st.session_state.astro_product = None
if "astro_preview_gif" not in st.session_state:
    st.session_state.astro_preview_gif = None

with tabs[0]:
    left, right = st.columns([0.34, 0.66], gap="large")
    with left:
        st.subheader("Pointing Inputs")
        if mode == "Simulate new pointing":
            dt = st.number_input(
                "Sample period dt (s)",
                min_value=0.001,
                max_value=10.0,
                value=0.02,
                step=0.01,
                format="%.4f",
                help="Simulation time step. Smaller values resolve faster jitter but increase run time.",
            )
            duration = st.number_input("Simulation total duration (s)", min_value=1.0, max_value=86400.0, value=300.0, step=60.0)
            jitter_std = st.number_input(
                "Jitter (short-term) RMS amplitude (deg)",
                min_value=0.0,
                value=0.001,
                step=0.001,
                format="%.5f",
                help="RMS amplitude of the fast pointing component.",
            )
            st.caption(_deg_caption(jitter_std))
            tau_jitter = st.number_input(
                "Jitter (short-term) correlation time (s)",
                min_value=0.001,
                value=0.3,
                step=0.1,
                help="Correlation time of the fast pointing component.",
            )
            bias_std = st.number_input(
                "Medium-term drift RMS amplitude (deg)",
                min_value=0.0,
                value=0.020,
                step=0.001,
                format="%.5f",
                help="RMS amplitude of the slower drift component. This is the former bias RMS input.",
            )
            st.caption(_deg_caption(bias_std))
            tau_drift = st.number_input(
                "Medium-term drift correlation time (s)",
                min_value=0.001,
                value=300.0,
                step=50.0,
                help="Correlation time of the slower drift component.",
            )
            win_sec = st.number_input("Metric window (s)", min_value=0.1, value=300.0, step=30.0, help=METRIC_WINDOW_HELP)
            use_guiding = st.toggle(
                "Use long-term guiding / slow recentering",
                value=True,
                help="Subtracts a very slow exponential moving average from the raw pointing series.",
            )
            guiding_tau = st.number_input(
                "Long-term guiding time constant (s)",
                min_value=0.001,
                value=3000.0,
                step=250.0,
                help="Time constant for the slow recentering model. Larger values make guiding correction slower.",
            )
            seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=7, step=1)

            if st.button("Run Pointing Simulation", type="primary", width="stretch"):
                opts = PointingOptions(
                    dt=dt,
                    duration=duration,
                    tau_jitter=tau_jitter,
                    tau_drift=tau_drift,
                    bias_std_deg=bias_std,
                    jitter_std_deg=jitter_std,
                    win_sec=win_sec,
                    use_guiding=use_guiding,
                    guiding_tau=guiding_tau,
                    seed=int(seed),
                )
                st.session_state.pointing_stats = simulate_pointing(opts)
                st.session_state.last_gif = None
        else:
            uploaded = st.file_uploader("Load a saved pointing `.npz`", type=["npz"])
            if uploaded is not None and st.button("Load Pointing", type="primary", width="stretch"):
                st.session_state.pointing_stats = load_pointing_npz(uploaded)
                st.session_state.last_gif = None

    with right:
        stats = st.session_state.pointing_stats
        if stats is None:
            st.info("Run or load a pointing simulation to populate the trajectory and metrics.")
        else:
            rpe_p99_deg = _current_rpe_p99_deg(stats)
            m1, m2, m3 = st.columns(3)
            m1.metric("99% MPE", f"{stats.mpe_p99_deg:.5f} deg", f"{stats.mpe_p99_deg * 3600.0:.2f} arcsec")
            m2.metric("99% RPE", f"{rpe_p99_deg:.5f} deg", f"{rpe_p99_deg * 3600.0:.2f} arcsec")
            m3.metric("99% global", f"{stats.global_p99_deg:.5f} deg", f"{stats.global_p99_deg * 3600.0:.2f} arcsec")
            st.caption(
                "MPE is the rolling window mean pointing error. RPE is the rolling residual about that window mean, "
                "so it is an angular pointing metric, not an angular velocity."
            )

            st.image(trajectory_png(stats), caption="Pointing trajectory in arcsec", width="stretch")
            st.image(timeseries_png(stats), caption="Pointing error over time", width="stretch")
            st.dataframe(_metrics_table(stats), hide_index=True, width="stretch")
            st.download_button(
                "Download Pointing Simulation",
                data=_npz_download(stats),
                file_name="pointing_simulation.npz",
                mime="application/octet-stream",
                width="stretch",
            )

with tabs[1]:
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.subheader("Telescope")
        aperture_m = st.number_input("Aperture (m)", min_value=0.01, value=0.20, step=0.01)
        f_number = st.number_input("F-number", min_value=0.5, value=8.0, step=0.5)
        airy_fwhm_um = st.number_input("PSF FWHM (um)", min_value=0.1, value=11.0, step=0.5)
    with c2:
        st.subheader("Sensor")
        nx = st.number_input("Sensor width Nx (px)", min_value=16, value=1600, step=128)
        ny = st.number_input("Sensor height Ny (px)", min_value=16, value=1200, step=128)
        pixel_pitch_um = st.number_input("Pixel pitch (um)", min_value=0.1, value=4.6, step=0.1)
    with c3:
        st.subheader("Detector")
        qe = st.slider("Quantum efficiency", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
        read_noise = st.number_input("Read noise (e- RMS)", min_value=0.0, value=0.9, step=0.1)
        sky_mag = st.number_input("Sky brightness (mag/arcsec^2)", min_value=5.0, max_value=30.0, value=22.0, step=0.1)
    focal_length_m = aperture_m * f_number
    plate_scale = _plate_scale_arcsec_per_pix(aperture_m, f_number, pixel_pitch_um)
    rayleigh = _rayleigh_arcsec(aperture_m)
    m1, m2, m3 = st.columns(3)
    m1.metric("Focal length", f"{focal_length_m:.3f} m")
    m2.metric("Plate scale", f"{plate_scale:.3f} arcsec/px")
    m3.metric("Rayleigh resolution", f"{rayleigh:.3f} arcsec")
    st.caption("Rayleigh resolution is calculated at 550 nm using 1.22 lambda / aperture.")

with tabs[2]:
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.subheader("Field Center")
        ra_input = st.text_input("RA (deg or HH MM SS.SS)", value="84.0", help="Use decimal degrees, or hours minutes seconds with spaces or colons.")
        dec_input = st.text_input("Dec (deg or DD MM SS.SS)", value="-5.0", help="Use decimal degrees, or degrees arcminutes arcseconds with spaces or colons.")
        target_valid = True
        try:
            ra_deg = _parse_ra(ra_input)
            dec_deg = _parse_dec(dec_input)
            st.caption(
                f"Parsed target: RA {ra_deg:.8f} deg ({ra_deg * 3600.0:.2f} arcsec), "
                f"Dec {dec_deg:.8f} deg ({dec_deg * 3600.0:.2f} arcsec)"
            )
        except ValueError as exc:
            target_valid = False
            ra_deg = 84.0
            dec_deg = -5.0
            st.error(str(exc))
    with c2:
        st.subheader("Catalog")
        catalog_source = st.selectbox(
            "Star source",
            ["Auto: APASS then Gaia", "APASS", "Gaia", "Synthetic offline"],
            index=0,
            help="These catalogs will be used to determine stars that fall on the image; resolved objects such as galaxies are not considered.",
        )
        band_label = st.selectbox("Band", ["V (Johnson-Cousins visual band)", "G (Gaia optical broad band)"], index=0)
        band = _band_code(band_label)
        mag_limit = st.number_input(
            "Magnitude limit",
            min_value=1.0,
            max_value=25.0,
            value=18.0,
            step=0.5,
            help="Stars fainter than this will not be considered in rendering the image.",
        )
    with c3:
        st.subheader("Synthetic Fallback")
        synthetic_count = st.number_input(
            "Synthetic star count",
            min_value=1,
            max_value=5000,
            value=350,
            step=50,
            help="Number of artificial stars generated when using Synthetic offline or when online catalog queries fail.",
        )
        synthetic_seed = st.number_input(
            "Synthetic seed",
            min_value=0,
            max_value=2_147_483_647,
            value=11,
            step=1,
            help="Random seed for the synthetic fallback field, allowing repeatable demo fields.",
        )

with tabs[3]:
    stats = st.session_state.pointing_stats
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.subheader("Frame")
        frame_exposure = st.number_input("Frame exposure (s)", min_value=0.001, value=0.1, step=0.05, format="%.4f")
        half_subframe = st.number_input("Half subframe (px)", min_value=8, max_value=2000, value=240, step=16)
        start_time = st.number_input("Start time in pointing run (s)", min_value=0.0, value=0.0, step=1.0)
    with c2:
        st.subheader("GIF")
        gif_duration = st.number_input("GIF simulated duration (s)", min_value=0.001, value=3.0, step=1.0)
        gif_fps = st.number_input("GIF playback FPS", min_value=1.0, max_value=60.0, value=10.0, step=1.0)
        max_frames = st.number_input("Maximum GIF frames", min_value=1, max_value=300, value=60, step=5)
    with c3:
        st.subheader("Display")
        stretch = st.selectbox("Grayscale stretch", ["percentile", "fixed"], index=0)
        p_low = st.number_input("Low percentile", min_value=0.0, max_value=99.0, value=1.0, step=0.5)
        p_high = st.number_input("High percentile", min_value=1.0, max_value=100.0, value=99.7, step=0.1)
        fixed_low = st.number_input("Fixed black level (e-)", value=-2.0, step=1.0)
        fixed_high = st.number_input("Fixed white level (e-)", value=35.0, step=1.0)

    imaging = ImagingOptions(
        aperture_m=aperture_m,
        f_number=f_number,
        airy_fwhm_um=airy_fwhm_um,
        pixel_pitch_um=pixel_pitch_um,
        nx=int(nx),
        ny=int(ny),
        qe=qe,
        read_noise=read_noise,
        sky_mag=sky_mag,
    )
    target = TargetOptions(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        band=band,
        mag_limit=mag_limit,
        catalog=CatalogOptions(source=catalog_source, synthetic_count=int(synthetic_count), synthetic_seed=int(synthetic_seed)),
    )
    render_opts = RenderOptions(
        frame_exposure=frame_exposure,
        half_subframe=int(half_subframe),
        start_time=start_time,
        stretch=stretch,
        percentile_low=p_low,
        percentile_high=p_high,
        fixed_clim=(fixed_low, fixed_high),
    )
    gif_opts = GifOptions(duration=gif_duration, fps=gif_fps, max_frames=int(max_frames))

    if not target_valid:
        st.info("Enter a valid target RA/Dec before rendering a preview frame or GIF.")
    elif stats is None:
        st.info("Run or load pointing first, then render a preview frame or GIF.")
    else:
        left, right = st.columns([0.38, 0.62], gap="large")
        with left:
            preview_clicked = st.button("Render Preview Frame", type="primary", width="stretch")
            gif_clicked = st.button("Make GIF", width="stretch")
        with right:
            if preview_clicked:
                with st.spinner("Rendering grayscale preview frame..."):
                    frame, meta = render_preview_frame(stats, imaging, target, render_opts)
                    st.session_state.last_frame_png = image_to_png_bytes(frame, render_opts, meta.plate_arcsec_per_pix)
                    st.session_state.last_frame_meta = meta
            if gif_clicked:
                with st.spinner("Rendering GIF frames..."):
                    frames_for_gif, meta = render_frame_sequence(stats, imaging, target, render_opts, gif_duration, int(max_frames))
                    gif_bytes = frames_to_gif(frames_for_gif, render_opts, gif_opts.fps)
                    st.session_state.last_gif = gif_bytes
                    st.session_state.last_gif_meta = meta
                    st.session_state.last_render_frames = frames_for_gif
                    st.session_state.last_render_meta = meta

            if st.session_state.last_frame_png:
                st.image(st.session_state.last_frame_png, caption="Latest grayscale preview", width="stretch")
                meta = st.session_state.get("last_frame_meta")
                if meta:
                    st.caption(f"{meta.catalog_name}; {len(meta.stars)} stars; plate scale {meta.plate_arcsec_per_pix:.3f} arcsec/px")
                st.download_button("Download Preview PNG", st.session_state.last_frame_png, "starfield_preview.png", "image/png")

            if st.session_state.last_gif:
                st.image(st.session_state.last_gif, caption="Latest GIF", width="stretch")
                meta = st.session_state.get("last_gif_meta")
                if meta:
                    st.caption(f"{meta.catalog_name}; {len(meta.stars)} stars; plate scale {meta.plate_arcsec_per_pix:.3f} arcsec/px")
                st.download_button("Download GIF", st.session_state.last_gif, "starfield_pointing.gif", "image/gif")

with tabs[4]:
    stats = st.session_state.pointing_stats
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.subheader("Input Sequence")
        input_source = st.radio(
            "Alignment input",
            ["Render from current simulation", "Use current rendered sequence", "Upload GIF"],
            index=0,
            key="align_input_source",
        )
        uploaded_align_gif = None
        if input_source == "Upload GIF":
            uploaded_align_gif = st.file_uploader("Upload GIF for alignment", type=["gif"], key="align_gif_upload")
        align_duration = st.number_input("Alignment sequence duration (s)", min_value=0.001, value=1.0, step=0.5, key="align_duration")
        align_max_frames = st.number_input("Maximum alignment frames", min_value=1, max_value=200, value=20, step=1, key="align_max_frames")
        align_fps = st.number_input("Aligned GIF FPS", min_value=1.0, max_value=60.0, value=10.0, step=1.0, key="align_fps")
    with c2:
        st.subheader("Centroid Tracking")
        align_mode = st.selectbox("Alignment mode", ["Target centroid (x/y shift)", "Multi-star rotation (coming later)"], index=0, key="align_mode")
        search_box_size = st.number_input("Centroid search box (px)", min_value=8, max_value=1000, value=96, step=8, key="align_search_box")
        threshold_sigma = st.number_input("Centroid threshold (sigma)", min_value=0.0, max_value=20.0, value=3.0, step=0.5, key="align_threshold_sigma")
    with c3:
        st.subheader("Alignment Output")
        reference_index = st.number_input("Reference frame index", min_value=0, max_value=199, value=0, step=1, key="align_reference_index")
        shift_mode_label = st.selectbox("Shift mode", ["Subpixel bilinear", "Integer pixels, fastest"], index=0, key="align_shift_mode")
        stack_method_label = st.selectbox("Stack method", ["mean", "sum", "max"], index=0, key="align_stack_method")

    can_align_without_sim = input_source == "Upload GIF" and uploaded_align_gif is not None
    can_align_current_sequence = input_source == "Use current rendered sequence" and st.session_state.last_render_frames is not None

    if input_source == "Render from current simulation" and not target_valid:
        st.info("Enter a valid target RA/Dec before rendering an alignment sequence.")
    elif input_source == "Render from current simulation" and stats is None:
        st.info("Run or load pointing first, then render and align an image sequence.")
    elif input_source == "Use current rendered sequence" and not can_align_current_sequence:
        st.info("Render a GIF in the Render & Download tab first, then this option will reuse that generated frame sequence.")
    elif input_source == "Upload GIF" and not can_align_without_sim:
        st.info("Upload a GIF to align and stack its frames.")
    elif align_mode != "Target centroid (x/y shift)":
        st.info("Multi-star translation/rotation is planned next. The first implemented mode is target-centroid x/y alignment.")
    else:
        if st.button("Render & Align Sequence", type="primary", width="stretch"):
            with st.spinner("Rendering sequence and aligning target centroid..."):
                if input_source == "Upload GIF":
                    frames = gif_bytes_to_frames(uploaded_align_gif.getvalue(), int(align_max_frames))
                    meta = None
                    plate_scale_for_display = None
                elif input_source == "Use current rendered sequence":
                    frames = st.session_state.last_render_frames[: int(align_max_frames)]
                    meta = st.session_state.last_render_meta
                    plate_scale_for_display = meta.plate_arcsec_per_pix if meta is not None else None
                else:
                    frames, meta = render_frame_sequence(stats, imaging, target, render_opts, align_duration, int(align_max_frames))
                    plate_scale_for_display = meta.plate_arcsec_per_pix
                result = align_target_centroid(
                    frames,
                    reference_index=int(reference_index),
                    search_box_size=int(search_box_size),
                    threshold_sigma=float(threshold_sigma),
                    shift_mode="integer" if shift_mode_label.startswith("Integer") else "bilinear",
                    stack_method=stack_method_label,
                )
                st.session_state.alignment_result = result
                st.session_state.alignment_meta = meta
                st.session_state.alignment_plate_scale = plate_scale_for_display
                st.session_state.aligned_gif = frames_to_gif_bytes(
                    result.aligned_frames,
                    render_opts,
                    fps=align_fps,
                    plate_arcsec_per_pix=plate_scale_for_display,
                )
                st.session_state.aligned_stack_png = frame_png(result.stack, render_opts, plate_scale_for_display)

        result = st.session_state.alignment_result
        plate_scale_for_display = st.session_state.get("alignment_plate_scale")
        if result is not None:
            n_frames = result.original_frames.shape[0]
            frame_idx = st.slider("Frame", min_value=0, max_value=n_frames - 1, value=0, step=1)
            row = {
                "frame": frame_idx,
                "centroid_x": result.centroids_xy[frame_idx, 0],
                "centroid_y": result.centroids_xy[frame_idx, 1],
                "shift_x": result.shifts_xy[frame_idx, 0],
                "shift_y": result.shifts_xy[frame_idx, 1],
                "centroid_found": bool(result.found[frame_idx]),
            }
            st.dataframe(pd.DataFrame([row]), hide_index=True, width="stretch")

            left, mid, right = st.columns(3, gap="large")
            with left:
                st.image(
                    overlay_centroid_png(
                        result.original_frames[frame_idx],
                        tuple(result.centroids_xy[frame_idx]),
                        render_opts,
                        plate_scale_for_display,
                    ),
                    caption="Original frame with centroid",
                    width="stretch",
                )
            with mid:
                st.image(
                    frame_png(result.aligned_frames[frame_idx], render_opts, plate_scale_for_display),
                    caption="Aligned frame",
                    width="stretch",
                )
            with right:
                st.image(st.session_state.aligned_stack_png, caption=f"{stack_method_label.title()} stack", width="stretch")

            cdl, cdm = st.columns(2)
            with cdl:
                st.download_button("Download Aligned GIF", st.session_state.aligned_gif, "aligned_sequence.gif", "image/gif")
            with cdm:
                st.download_button("Download Stacked PNG", st.session_state.aligned_stack_png, "aligned_stack.png", "image/png")

with tabs[5]:
    stats = st.session_state.pointing_stats
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.subheader("Long Run")
        astro_total_duration = st.number_input(
            "Astronomical timeseries duration (s)",
            min_value=0.001,
            max_value=86400.0,
            value=1800.0,
            step=600.0,
            key="astro_total_duration",
        )
        astro_frames_per_stack = st.number_input(
            "Simulated frames per stacked image",
            min_value=1,
            max_value=1000,
            value=20,
            step=1,
            key="astro_frames_per_stack",
            help="Each output image is made by rendering this many short-exposure frames, aligning them, and stacking them.",
        )
        astro_max_outputs = st.number_input(
            "Maximum output stacked images",
            min_value=1,
            max_value=5000,
            value=100,
            step=10,
            key="astro_max_outputs",
        )
    with c2:
        st.subheader("Target Variability")
        astro_delta_text = st.text_area(
            "Delta magnitudes",
            value="",
            height=120,
            key="astro_delta_text",
            help="Optional whitespace- or comma-separated list. Positive values make the target star fainter. If supplied, the list length must equal the output image count.",
        )
        astro_delta_file = st.file_uploader("Upload delta magnitudes CSV/TXT", type=["csv", "txt"], key="astro_delta_file")
        astro_fresh_pointing = st.toggle(
            "Simulate fresh long pointing for this run",
            value=True,
            key="astro_fresh_pointing",
            help="Uses the current pointing parameters as a template, but extends the run to cover this whole astronomical timeseries.",
        )
    with c3:
        st.subheader("Output")
        astro_output_dir = st.text_input(
            "Output directory path",
            value="",
            key="astro_output_dir",
            help="Optional local/server directory. For large runs this avoids holding every output in browser memory.",
        )
        astro_make_zip = st.toggle(
            "Create downloadable ZIP",
            value=True,
            key="astro_make_zip",
            help="Convenient for small to medium runs. For very large products, prefer writing to an output directory.",
        )
        astro_shift_mode_label = st.selectbox(
            "Timeseries shift mode",
            ["Subpixel bilinear", "Integer pixels, fastest"],
            index=0,
            key="astro_shift_mode",
        )

    astro_cadence = float(frame_exposure) * int(astro_frames_per_stack)
    astro_expected_outputs = min(
        max(1, int(astro_max_outputs)),
        max(1, int((astro_total_duration / max(astro_cadence, 1e-12)) + 1e-9)),
    )
    st.caption(
        f"Cadence: {astro_cadence:.4g} s per stacked image; this run will produce {astro_expected_outputs} stacked outputs. "
        "The target star is the catalog star nearest the requested field center."
    )

    delta_source = ""
    if astro_delta_file is not None:
        delta_source = astro_delta_file.getvalue().decode("utf-8", errors="replace")
    elif astro_delta_text.strip():
        delta_source = astro_delta_text

    if delta_source:
        try:
            parsed_delta_preview = parse_delta_magnitudes(delta_source)
            st.caption(f"Parsed {len(parsed_delta_preview)} delta-magnitude values.")
        except ValueError as exc:
            st.error(f"Could not parse delta magnitudes: {exc}")

    if not target_valid:
        st.info("Enter a valid target RA/Dec before running an astronomical timeseries.")
    elif stats is None:
        st.info("Run or load a pointing simulation first. The timeseries tab uses it as the pointing model template.")
    else:
        if astro_expected_outputs > 300 and astro_make_zip:
            st.warning("A ZIP with hundreds or thousands of float TIFFs may be large. For big runs, use an output directory and turn the ZIP off.")

        if st.button("Run Astronomical Timeseries", type="primary", width="stretch"):
            try:
                deltas = parse_delta_magnitudes(delta_source) if delta_source else None
                required_end = float(start_time) + float(astro_total_duration) + float(frame_exposure)
                run_stats = stats
                if astro_fresh_pointing:
                    long_opts = dc_replace(stats.options, duration=required_end)
                    run_stats = simulate_pointing(long_opts)
                elif float(stats.t[-1]) < required_end:
                    st.error(
                        f"The current pointing simulation ends at {float(stats.t[-1]):.2f} s, "
                        f"but this timeseries needs {required_end:.2f} s. Enable fresh long pointing or run a longer pointing simulation."
                    )
                    st.stop()

                with st.spinner("Rendering, aligning, stacking, and writing the astronomical timeseries..."):
                    scene = build_scene(imaging, target)
                    progress = st.progress(0, text="Starting astronomical timeseries...")
                    product = run_astronomical_timeseries(
                        run_stats,
                        imaging,
                        render_opts,
                        scene,
                        total_duration=float(astro_total_duration),
                        frames_per_stack=int(astro_frames_per_stack),
                        max_outputs=int(astro_max_outputs),
                        delta_magnitudes=deltas,
                        search_box_size=int(search_box_size),
                        threshold_sigma=float(threshold_sigma),
                        shift_mode="integer" if astro_shift_mode_label.startswith("Integer") else "bilinear",
                        stack_method=stack_method_label,
                        output_dir=astro_output_dir.strip() or None,
                        make_zip=bool(astro_make_zip),
                        progress_callback=lambda done, total: progress.progress(
                            done / total,
                            text=f"Processed {done}/{total} stacked outputs",
                        ),
                    )
                    progress.empty()
                    st.session_state.astro_product = product
                    st.session_state.astro_preview_gif = (
                        frames_to_gif_bytes(product.preview_stacks, render_opts, fps=4.0, plate_arcsec_per_pix=scene.plate_arcsec_per_pix)
                        if product.preview_stacks.size
                        else None
                    )
            except ValueError as exc:
                st.error(str(exc))

        product = st.session_state.astro_product
        if product is not None:
            st.success(f"Created {product.n_outputs} stacked astronomical-timeseries images.")
            if product.output_dir:
                st.caption(f"Wrote float TIFF images and metadata to `{product.output_dir}`.")
            if product.target_index is None:
                st.warning("No catalog stars were available, so no target-star delta magnitude could be applied.")

            if product.preview_stacks.size:
                idx = st.slider("Preview stacked image", 0, product.preview_stacks.shape[0] - 1, 0, key="astro_preview_idx")
                st.image(frame_png(product.preview_stacks[idx], render_opts), caption="Preview stack", width="stretch")
                if st.session_state.astro_preview_gif:
                    st.image(st.session_state.astro_preview_gif, caption="Preview of first stacks", width="stretch")

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download Metadata CSV",
                    product.metadata_csv,
                    "astro_timeseries_metadata.csv",
                    "text/csv",
                    width="stretch",
                )
            with dl2:
                if product.zip_bytes is not None:
                    st.download_button(
                        "Download Float TIFF ZIP",
                        product.zip_bytes,
                        "astro_timeseries_float_tiffs.zip",
                        "application/zip",
                        width="stretch",
                    )
