from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CatalogOptions:
    source: str = "Auto: APASS then Gaia"
    synthetic_count: int = 350
    synthetic_seed: int = 11


@dataclass
class StarCatalog:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray
    name: str

    def __len__(self) -> int:
        return int(self.mag.size)


def small_angle_offsets(ra0_deg: float, dec0_deg: float, ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dra = (ra_deg - ra0_deg) * np.cos(np.deg2rad(dec0_deg))
    ddec = dec_deg - dec0_deg
    return dra * 3600.0, ddec * 3600.0


def query_catalog(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    mag_limit: float,
    band: str,
    options: CatalogOptions,
) -> StarCatalog:
    source = options.source.lower()
    errors: list[str] = []

    if "synthetic" not in source:
        if "apass" in source or "auto" in source:
            try:
                stars = _query_apass(ra_deg, dec_deg, radius_deg, mag_limit)
                if len(stars):
                    return stars
            except Exception as exc:  # noqa: BLE001 - fallback is part of the app behavior.
                errors.append(f"APASS: {exc}")
        if "gaia" in source or "auto" in source:
            try:
                stars = _query_gaia(ra_deg, dec_deg, radius_deg, mag_limit)
                if len(stars):
                    return stars
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Gaia: {exc}")

    synthetic = synthetic_catalog(ra_deg, dec_deg, radius_deg, mag_limit, options.synthetic_count, options.synthetic_seed)
    if errors:
        synthetic.name += " (catalog fallback)"
    return synthetic


def synthetic_catalog(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    mag_limit: float,
    count: int,
    seed: int,
) -> StarCatalog:
    rng = np.random.default_rng(seed)
    r = radius_deg * np.sqrt(rng.random(count))
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    dra = (r * np.cos(theta)) / max(0.05, np.cos(np.deg2rad(dec_deg)))
    ddec = r * np.sin(theta)
    # More faint stars than bright stars, roughly matching a demo field.
    min_mag = max(4.0, mag_limit - 9.0)
    u = rng.random(count)
    mag = min_mag + (mag_limit - min_mag) * np.sqrt(u)
    return StarCatalog(ra_deg + dra, dec_deg + ddec, mag, "Synthetic offline field")


def _tap_csv(url: str, adql: str) -> pd.DataFrame:
    body = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}).encode()
    with urlopen(url, data=body, timeout=45) as resp:  # noqa: S310 - user-triggered query to fixed astronomy TAP services.
        text = resp.read().decode("utf-8", errors="replace")
    return pd.read_csv(StringIO(text))


def _query_apass(ra_deg: float, dec_deg: float, radius_deg: float, mag_limit: float) -> StarCatalog:
    adql = (
        "SELECT TOP 50000 RAJ2000, DEJ2000, Vmag "
        'FROM "II/336/apass9" '
        "WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), "
        f"CIRCLE('ICRS', {ra_deg:.8f}, {dec_deg:.8f}, {radius_deg:.8f})) "
        f"AND Vmag IS NOT NULL AND Vmag <= {mag_limit:.3f}"
    )
    table = _tap_csv("https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync", adql)
    if table.empty:
        return StarCatalog(np.array([]), np.array([]), np.array([]), "APASS DR9 (V)")
    return StarCatalog(
        table["RAJ2000"].to_numpy(float),
        table["DEJ2000"].to_numpy(float),
        table["Vmag"].to_numpy(float),
        "APASS DR9 (V)",
    )


def _query_gaia(ra_deg: float, dec_deg: float, radius_deg: float, mag_limit: float) -> StarCatalog:
    adql = (
        "SELECT TOP 50000 ra, dec, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source "
        "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra_deg:.8f}, {dec_deg:.8f}, {radius_deg:.8f})) "
        f"AND phot_g_mean_mag <= {mag_limit:.3f}"
    )
    table = _tap_csv("https://gea.esac.esa.int/tap-server/tap/sync", adql)
    if table.empty:
        return StarCatalog(np.array([]), np.array([]), np.array([]), "Gaia DR3 (G)")
    return StarCatalog(
        table["ra"].to_numpy(float),
        table["dec"].to_numpy(float),
        table["phot_g_mean_mag"].to_numpy(float),
        "Gaia DR3 (G)",
    )
