"""Photo decoding, template matching and perspective correction.

`align_photo` runs in worker processes. Template features are read from an .npz file
written once by `prepare_template`, so workers never receive large pickled arrays.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import pymupdf
from PIL import Image

from .enhance import RENDER_PARAMS, clean_page

pillow_heif.register_heif_opener(thumbnails=False)

IMAGE_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PAGE_SIZES = {"letter": (612.0, 792.0), "a4": (595.28, 841.89), "legal": (612.0, 1008.0)}

MATCH_PARAMS = dict(
    version=6,
    work_long_side=2500,
    max_features=30000,
    lowe_ratio=0.75,
    ransac_thresh=5.0,
    min_inliers=20,
    refine=True,
    refine_grid=50,
    refine_sigma=90,
    refine_max_disp=60,
)

_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


# ---------------------------------------------------------------- photos
def list_photos(src: Path) -> list[Path]:
    return [p for p in src.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")]


def capture_time(path: Path) -> str:
    """EXIF DateTimeOriginal (with sub-seconds) as an ISO string, or '' if unavailable."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)
            dt = sub.get(36867) or exif.get(306)
            subsec = str(sub.get(37521) or "").strip()
        if dt:
            s = datetime.strptime(str(dt).strip(), "%Y:%m:%d %H:%M:%S").isoformat(sep=" ")
            return f"{s}.{subsec}" if subsec else s
    except Exception:
        pass
    return ""


def load_gray(path: Path, tries: int = 4) -> np.ndarray:
    """Full-resolution grayscale with EXIF orientation applied."""
    for attempt in range(tries):
        try:
            with Image.open(path) as im:
                orientation = im.getexif().get(274, 1)
                gray = im.convert("L")
            if orientation in _TRANSPOSE:
                gray = gray.transpose(_TRANSPOSE[orientation])
            return np.asarray(gray)
        except InterruptedError:
            if attempt == tries - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


# ---------------------------------------------------------------- features
def sift_tiled(img: np.ndarray, nfeatures: int = 0, tile: int = 512, pad: int = 48):
    """SIFT over overlapping tiles. Returns (N, 2) float32 points and (N, 128) descriptors.

    OpenCV's SIFT upsamples its input 2x and keeps ~11 float layers, so running it on a
    whole photo costs several hundred MB. Tiling bounds that by the tile size. Keypoints are
    kept only from each tile's interior and the strongest `nfeatures` are chosen globally,
    so the result matches an untiled run closely.
    """
    h, w = img.shape
    ny, nx = max(1, round(h / tile)), max(1, round(w / tile))
    ys, xs = np.linspace(0, h, ny + 1).astype(int), np.linspace(0, w, nx + 1).astype(int)
    sift = cv2.SIFT_create()
    pts, resp, descs = [], [], []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            oy, ox = max(0, y0 - pad), max(0, x0 - pad)
            kp, desc = sift.detectAndCompute(img[oy:min(h, y1 + pad), ox:min(w, x1 + pad)], None)
            if desc is None:
                continue
            p = np.float32([k.pt for k in kp]) + (ox, oy)
            keep = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
            pts.append(p[keep])
            resp.append(np.float32([k.response for k in kp])[keep])
            descs.append(desc[keep])
    if not pts:
        return np.zeros((0, 2), np.float32), None
    pts, resp, descs = np.concatenate(pts), np.concatenate(resp), np.concatenate(descs)
    if nfeatures and len(pts) > nfeatures:
        top = np.argsort(-resp)[:nfeatures]
        pts, descs = pts[top], descs[top]
    return pts, descs


# ---------------------------------------------------------------- template
@dataclass
class Template:
    pdf: Path | None
    dpi: int
    size_pt: tuple[float, float]
    shape: tuple[int, int]
    n_pages: int
    features: Path | None
    digest: str


def blank_template(page_size: str, dpi: int) -> Template:
    w, h = PAGE_SIZES[page_size]
    return Template(None, dpi, (w, h), (round(h / 72 * dpi), round(w / 72 * dpi)), 0, None, page_size)


def render_template_page(pdf: Path, idx: int, dpi: int) -> np.ndarray:
    with pymupdf.open(pdf) as doc:
        pix = doc[idx].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def prepare_template(pdf: Path, dpi: int, cache_dir: Path) -> Template:
    """Read page size from the PDF and cache SIFT features for every page."""
    digest = hashlib.sha1(pdf.read_bytes()).hexdigest()[:12]
    with pymupdf.open(pdf) as doc:
        sizes = {(round(p.rect.width, 2), round(p.rect.height, 2)) for p in doc}
        n = len(doc)
    if len(sizes) != 1:
        raise SystemExit(f"Template pages have different sizes: {sizes}")
    features = cache_dir / f"template_{digest}_{dpi}dpi.npz"
    if not features.exists():
        def page_features(i):
            img = render_template_page(pdf, i, dpi)
            kp, desc = sift_tiled(img)
            return img.shape, kp, desc

        with ThreadPoolExecutor(2) as ex:
            pages = list(ex.map(page_features, range(n)))
        arrays = {"shape": np.array(pages[0][0])}
        for i, (_, kp, desc) in enumerate(pages):
            arrays[f"kp{i}"], arrays[f"desc{i}"] = kp, desc
        tmp = features.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.rename(features)
    with np.load(features) as data:
        shape = tuple(int(v) for v in data["shape"])
    return Template(pdf, dpi, sizes.pop(), shape, n, features, digest)


# ---------------------------------------------------------------- worker state
_W: dict = {}


def init_worker(features: str | None):
    cv2.setNumThreads(1)
    _W.clear()
    if not features:
        return
    with np.load(features) as data:
        n = sum(1 for k in data.files if k.startswith("kp"))
        _W["kp"] = [data[f"kp{i}"] for i in range(n)]
        _W["matchers"] = []
        for i in range(n):
            m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
            m.add([data[f"desc{i}"]])
            m.train()
            _W["matchers"].append(m)


# ---------------------------------------------------------------- matching
def match_page(idx: int, kp: np.ndarray, desc: np.ndarray, mp: dict):
    """Homography (photo -> template page idx) and its RANSAC inlier count."""
    if desc is None or len(desc) < 10:
        return None, 0
    good = [m for m, n in (x for x in _W["matchers"][idx].knnMatch(desc, k=2) if len(x) == 2)
            if m.distance < mp["lowe_ratio"] * n.distance]
    if len(good) < 8:
        return None, 0
    src = kp[[m.queryIdx for m in good]]
    dst = _W["kp"][idx][[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, mp["ransac_thresh"], maxIters=5000, confidence=0.999)
    return (None, 0) if H is None else (H, int(mask.sum()))


def check_homography(H: np.ndarray, photo_shape, page_shape) -> str:
    """Sanity-check the page outline as it sits in the photo. Empty string means OK."""
    th, tw = page_shape
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    try:
        q = cv2.perspectiveTransform(corners, np.linalg.inv(H)).reshape(4, 2)
    except np.linalg.LinAlgError:
        return "singular homography"
    if not np.all(np.isfinite(q)):
        return "non-finite outline"
    cross = [np.cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return "outline self-intersecting or non-convex"
    ratio = cv2.contourArea(q) / float(photo_shape[0] * photo_shape[1])
    if not 0.15 <= ratio <= 1.3:
        return f"page area ratio {ratio:.2f} out of range"
    sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    for a, b in ((sides[0], sides[2]), (sides[1], sides[3])):
        if max(a, b) / max(1e-6, min(a, b)) > 1.8:
            return "extreme perspective (opposite sides differ >1.8x)"
    for i in range(4):
        u, v = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        ang = np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
        if not 45 <= ang <= 135:
            return f"extreme skew (corner angle {ang:.0f} deg)"
    aspect, expect = (sides[0] + sides[2]) / (sides[1] + sides[3]), tw / th
    if not 0.6 * expect <= aspect <= 1.6 * expect:
        return f"aspect ratio {aspect:.2f} vs expected {expect:.2f}"
    return ""


def _kernel_regress(pts, vals, query, sigma, prior_w=1e-3):
    d2 = ((query[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * sigma * sigma))
    W = w.sum(1)
    return (w @ vals) / (W + prior_w)[:, None], W


def refine_alignment(warped: np.ndarray, page_idx: int, mp: dict):
    """Fit a smooth displacement field d(t) with warped(t + d(t)) ~ template(t).

    Corrects page curl that a single homography cannot model, using matches between
    printed content in the warped photo and the template.
    """
    wk, desc = sift_tiled(warped)
    if desc is None or len(wk) < 20:
        return None, {}
    t_pts, d_vals = [], []
    for x in _W["matchers"][page_idx].knnMatch(desc, k=2):
        if len(x) == 2 and x[0].distance < 0.8 * x[1].distance:
            t = _W["kp"][page_idx][x[0].trainIdx]
            d = wk[x[0].queryIdx] - t
            if np.hypot(*d) < mp["refine_max_disp"]:
                t_pts.append(t)
                d_vals.append(d)
    if len(t_pts) < 30:
        return None, {"refine_matches": len(t_pts)}
    t_pts, d_vals = np.float32(t_pts), np.float32(d_vals)
    sigma = mp["refine_sigma"]
    keep = np.ones(len(t_pts), bool)
    for _ in range(3):
        est, _ = _kernel_regress(t_pts[keep], d_vals[keep], t_pts, sigma)
        r = np.hypot(*(d_vals - est).T)
        keep = r < max(3.0, 3 * np.median(r[keep]))
    t_pts, d_vals = t_pts[keep], d_vals[keep]
    est, _ = _kernel_regress(t_pts, d_vals, t_pts, sigma)
    resid, before = np.hypot(*(d_vals - est).T), np.hypot(*d_vals.T)

    th, tw = warped.shape
    g = mp["refine_grid"]
    gx, gy = np.arange(0, tw + g, g, dtype=np.float32), np.arange(0, th + g, g, dtype=np.float32)
    q = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    fine, W = _kernel_regress(t_pts, d_vals, q, sigma)
    coarse, _ = _kernel_regress(t_pts, d_vals, q, sigma * 4)
    field = (W[:, None] * fine + 2.0 * coarse) / (W[:, None] + 2.0)
    field = np.clip(field, -mp["refine_max_disp"], mp["refine_max_disp"])
    stats = dict(refine_matches=int(len(t_pts)),
                 err_before_px=round(float(np.median(before)), 2),
                 err_after_px=round(float(np.median(resid)), 2),
                 err_p90_after_px=round(float(np.percentile(resid, 90)), 2))
    return field.reshape(len(gy), len(gx), 2).astype(np.float32), stats


# ---------------------------------------------------------------- warping
def warp_mapped(full: np.ndarray, H: np.ndarray, shape, grid: np.ndarray | None = None,
                step: int = 0, strip: int = 256) -> np.ndarray:
    """out(t) = full(H^-1 (t + d(t))), resampled once, in horizontal strips to bound memory."""
    th, tw = shape
    Hinv = np.linalg.inv(H)
    if grid is not None:
        gh, gw, _ = grid.shape
        size = ((gw - 1) * step + 1, (gh - 1) * step + 1)
        dx = cv2.resize(grid[..., 0], size, interpolation=cv2.INTER_LINEAR)[:th, :tw]
        dy = cv2.resize(grid[..., 1], size, interpolation=cv2.INTER_LINEAR)[:th, :tw]
    out = np.empty((th, tw), np.uint8)
    xs = np.arange(tw, dtype=np.float32)
    for y0 in range(0, th, strip):
        y1 = min(th, y0 + strip)
        px, py = np.meshgrid(xs, np.arange(y0, y1, dtype=np.float32))
        if grid is not None:
            px, py = px + dx[y0:y1], py + dy[y0:y1]
        src = cv2.perspectiveTransform(np.stack([px, py], -1).reshape(-1, 1, 2), Hinv).reshape(y1 - y0, tw, 2)
        out[y0:y1] = cv2.remap(full, src[..., 0], src[..., 1], cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out


def _quad(contour: np.ndarray) -> np.ndarray:
    """Four corners (TL, TR, BR, BL) approximating a contour's convex hull."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 10):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            break
    else:
        pts = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def detect_outline(gray: np.ndarray, aspect: float):
    """Page corners (TL, TR, BR, BL) in `gray` coordinates, or None.

    Candidate outlines come from several brightness thresholds and from edge contours;
    the winner is the one whose interior is most uniformly paper and whose aspect ratio
    (width / height) is closest to `aspect`.
    """
    scale = min(1.0, 1000 / max(gray.shape))
    img = cv2.GaussianBlur(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), (7, 7), 0)
    min_area = 0.15 * img.shape[0] * img.shape[1]
    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cands = []
    for t in (otsu, otsu + 0.25 * (255 - otsu), otsu + 0.5 * (255 - otsu)):
        mask = (img > t).astype(np.uint8) * 255
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ell(25)), cv2.MORPH_OPEN, ell(31))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands += [_quad(c) for c in cnts if cv2.contourArea(c) >= min_area]
    edges = cv2.dilate(cv2.Canny(img, 50, 150), np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if cv2.contourArea(c) >= min_area and len(approx) == 4 and cv2.isContourConvex(approx):
            cands.append(_quad(approx))

    def score(q):
        poly = np.zeros(img.shape, np.uint8)
        cv2.fillConvexPoly(poly, q.astype(np.int32), 1)
        paper = float((img[poly > 0] > otsu).mean())
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        area = poly.mean()
        return paper - 0.5 * abs(np.log(max(w, 1) / max(h, 1) / aspect)) + 0.1 * area

    return max(cands, key=score) / scale if cands else None


def fit_into(full: np.ndarray, shape) -> np.ndarray:
    """Scale the whole photo into the page, preserving aspect ratio, on white."""
    th, tw = shape
    s = min(tw / full.shape[1], th / full.shape[0])
    img = cv2.resize(full, (round(full.shape[1] * s), round(full.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.full((th, tw), 255, np.uint8)
    y, x = (th - img.shape[0]) // 2, (tw - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


# ---------------------------------------------------------------- worker entry point
@dataclass
class Job:
    path: str
    out_prefix: str
    shape: tuple[int, int]
    dpi: int
    force_page: int | None = None
    cached: dict | None = None


def _locate(full: np.ndarray, job: Job, mp: dict) -> dict:
    th, tw = job.shape
    scale = min(1.0, mp["work_long_side"] / max(full.shape))
    small = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else full
    res = dict(page=None, inliers=0, second_inliers=0, second_page=None, method="outline",
               H=None, grid=None, sanity="")

    if "matchers" in _W:
        kp, desc = sift_tiled(small, mp["max_features"])
        pages = [job.force_page - 1] if job.force_page else range(len(_W["matchers"]))
        found = {i: match_page(i, kp, desc, mp) for i in pages}
        ranked = sorted(found, key=lambda i: found[i][1], reverse=True)
        H, inl = found[ranked[0]]
        res.update(page=ranked[0] + 1, inliers=inl)
        if len(ranked) > 1:
            res.update(second_page=ranked[1] + 1, second_inliers=found[ranked[1]][1])
        if H is not None and inl >= mp["min_inliers"]:
            H = H @ np.diag([scale, scale, 1.0])
            res["sanity"] = check_homography(H, full.shape, job.shape)
            if not res["sanity"]:
                res.update(method="homography", H=H.tolist())
                if mp["refine"]:
                    coarse = cv2.warpPerspective(full, H, (tw, th), flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                    grid, stats = refine_alignment(coarse, ranked[0], mp)
                    res.update(stats, grid=None if grid is None else grid.round(2).tolist())
                return res
        elif not res["sanity"]:
            res["sanity"] = f"no usable homography ({inl} inliers)"

    corners = detect_outline(full, tw / th)
    if corners is None:
        res["method"] = "resize"
        return res
    dst = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
    H = cv2.getPerspectiveTransform(corners, dst)
    res["H"] = H.tolist()
    if not res["sanity"]:
        res["outline_check"] = check_homography(H, full.shape, job.shape)
    return res


def align_photo(job: Job) -> tuple[str, dict | None, str]:
    """Locate, warp, clean and save one photo. Returns (path, result, error)."""
    t0 = time.time()
    try:
        mp = MATCH_PARAMS
        full = load_gray(Path(job.path))
        res = job.cached if job.cached is not None else _locate(full, job, mp)
        if res["H"] is None:
            warped = fit_into(full, job.shape)
        else:
            grid = None if res.get("grid") is None else np.float32(res["grid"])
            warped = warp_mapped(full, np.array(res["H"]), job.shape, grid, mp["refine_grid"])
        del full
        prefix = Path(job.out_prefix)
        out = prefix.with_name(prefix.name + (f"__p{res['page']}" if res["page"] else "") + ".jpg")
        for stale in prefix.parent.glob(prefix.name + "*.jpg"):
            if stale != out and stale.name[len(prefix.name):len(prefix.name) + 3] in ("__p", ".jpg"):
                stale.unlink()
        Image.fromarray(clean_page(warped)).save(out, quality=RENDER_PARAMS["jpeg_quality"], dpi=(job.dpi, job.dpi))
        res["image"] = out.name
        res["seconds"] = round(time.time() - t0, 1)
        return job.path, res, ""
    except Exception as e:
        return job.path, None, repr(e)
