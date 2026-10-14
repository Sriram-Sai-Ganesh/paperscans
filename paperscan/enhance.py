"""Grayscale page cleanup: shadow removal, contrast stretch, frame-edge cleanup."""

from __future__ import annotations

import cv2
import numpy as np

RENDER_PARAMS = dict(
    version=1,
    bg_dilate=15,
    bg_median=41,
    white_point=0.90,
    black_point_pct=2.0,
    edge_band=0.05,
    jpeg_quality=80,
)


def clean_page(gray: np.ndarray, p: dict = RENDER_PARAMS) -> np.ndarray:
    """Divide out the background, stretch contrast and remove frame slivers. Never binarizes."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (p["bg_dilate"], p["bg_dilate"]))
    bg = cv2.medianBlur(cv2.dilate(gray, k), p["bg_median"])
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0), 0, 1)
    white = p["white_point"]
    ink = norm[norm < white]
    black = float(np.percentile(ink, p["black_point_pct"])) if ink.size > 1000 else 0.0
    black = min(black, white - 0.2)
    out = (np.clip((norm - black) / (white - black), 0, 1) * 255 + 0.5).astype(np.uint8)
    return whiten_frame_edges(out, p["edge_band"])


def whiten_frame_edges(img: np.ndarray, band_frac: float) -> np.ndarray:
    """Remove dark blobs near the frame that are background, not writing.

    Within a margin band, a dark component is removed if it touches the frame, is a long
    thin line parallel to a side (a visible paper edge), or lies in the outermost rim.
    """
    h, w = img.shape
    b = max(4, round(band_frac * min(h, w)))
    rim = max(3, round(0.012 * min(h, w)))
    band = np.zeros((h, w), bool)
    band[:b], band[-b:], band[:, :b], band[:, -b:] = True, True, True, True
    n, lab, st, _ = cv2.connectedComponentsWithStats(((img < 200) & band).astype(np.uint8), connectivity=8)
    kill = np.zeros(n, bool)
    for i in range(1, n):
        x, y, ww, hh = st[i, :4]
        touches = x == 0 or y == 0 or x + ww == w or y + hh == h
        edge_line = (ww > 0.15 * w and hh < 30) or (hh > 0.15 * h and ww < 30)
        in_rim = x + ww <= rim or y + hh <= rim or x >= w - rim or y >= h - rim
        kill[i] = touches or edge_line or in_rim
    mask = kill[lab].astype(np.uint8)
    if not mask.any():
        return img
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool) & band
    img = img.copy()
    img[mask] = 255
    return img
