"""paperscan: phone photos of paper pages -> clean, aligned PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from . import align, report
from .enhance import RENDER_PARAMS


def _key(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def default_workers() -> int:
    try:
        ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError, AttributeError):
        ram_gb = 8
    return max(1, min(4, (os.cpu_count() or 2) - 1, int(ram_gb / 4 / 0.75)))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="paperscan",
        description="Turn phone photos of paper pages into a clean, straightened PDF. With --template, "
                    "each photo is identified and aligned to the matching page of a blank PDF.")
    ap.add_argument("photos", type=Path, help="folder of photos (HEIC/JPG/PNG/TIFF/WebP)")
    ap.add_argument("-t", "--template", type=Path, help="blank PDF to align pages to")
    ap.add_argument("-o", "--out", type=Path, default=Path("paperscan-out"), help="output folder")
    ap.add_argument("--name", default="scans", help="output PDF name (default: scans)")
    ap.add_argument("--pages-per-doc", type=int,
                    help="pages per document (default: template page count, else 1)")
    ap.add_argument("--page-size", choices=sorted(align.PAGE_SIZES), default="letter",
                    help="output page size without a template (default: letter)")
    ap.add_argument("--dpi", type=int, default=200, help="output resolution (default: 200)")
    ap.add_argument("--expected", type=int, help="warn if the photo count differs")
    ap.add_argument("--limit", type=int, help="only process the first N photos")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="parallel photo workers, ~0.75 GB RAM each (default: up to 4, within 1/4 of RAM)")
    ap.add_argument("--min-inliers", type=int, default=60, help="flag matches below this (default: 60)")
    ap.add_argument("--min-ratio", type=float, default=1.5,
                    help="flag if best/second-best page inliers is below this (default: 1.5)")
    ap.add_argument("--max-pdf-mb", type=float, default=100,
                    help="also write part files split at document boundaries above this size (default: 100)")
    ap.add_argument("--overrides", type=Path,
                    help='JSON of manual fixes, e.g. {"IMG_1.HEIC": {"page": 3}, "IMG_2.HEIC": "skip"} '
                         "(default: <out>/overrides.json if present)")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    ap.add_argument("--allow-problems", action="store_true",
                    help="build the PDF even if document grouping has problems")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out
    dirs = {d: out / d for d in ("pages", "cache", "review", "contact_sheets")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    ov_path = args.overrides or out / "overrides.json"
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    if overrides:
        print(f"Loaded {len(overrides)} override(s) from {ov_path}")

    paths = align.list_photos(args.photos)
    with ThreadPoolExecutor(16) as ex:
        times = dict(zip(paths, tqdm(ex.map(align.capture_time, paths), total=len(paths),
                                     desc="Reading EXIF", unit="photo", leave=False)))
    paths.sort(key=lambda p: (times[p] == "", times[p], p.name))
    print(f"Found {len(paths)} photos in {args.photos}/")
    if args.expected and len(paths) != args.expected:
        print(f"WARNING: expected {args.expected} photos, found {len(paths)}. Continuing.")
    skipped = [p for p in paths if overrides.get(p.name) == "skip"]
    paths = [p for p in paths if overrides.get(p.name) != "skip"]
    if skipped:
        print(f"Skipping per overrides: {', '.join(p.name for p in skipped)}")
    if args.limit:
        paths = paths[:args.limit]
        print(f"--limit: processing the first {len(paths)}")
    if not paths:
        sys.exit("No photos to process.")

    if args.template:
        tmpl = align.prepare_template(args.template, args.dpi, dirs["cache"])
        print(f"Template: {tmpl.n_pages} pages, {tmpl.size_pt[0]:g}x{tmpl.size_pt[1]:g} pt, "
              f"{tmpl.shape[1]}x{tmpl.shape[0]} px @ {args.dpi} dpi")
    else:
        tmpl = align.blank_template(args.page_size, args.dpi)
        print(f"No template: detecting page outlines, output {args.page_size} @ {args.dpi} dpi")
    templated = tmpl.features is not None
    per_doc = args.pages_per_doc or tmpl.n_pages or 1
    mkey = _key({**align.MATCH_PARAMS, "tmpl": tmpl.digest, "dpi": args.dpi})
    rkey = _key(RENDER_PARAMS)

    results, jobs, sigs = {}, [], {}
    for p in paths:
        ov = overrides.get(p.name)
        force_page = ov.get("page") if isinstance(ov, dict) else None
        st = p.stat()
        sigs[p] = f"{st.st_size}:{st.st_mtime_ns}:{force_page}"
        cache_f = dirs["cache"] / f"{p.name}.json"
        cached = None
        if cache_f.exists() and not args.force:
            c = json.loads(cache_f.read_text())
            if c.get("src") == sigs[p] and c.get("mkey") == mkey:
                cached = c["res"]
                if c.get("rkey") == rkey and (dirs["pages"] / cached.get("image", "")).is_file():
                    results[p] = cached
                    continue
        prefix = dirs["pages"] / p.name.replace(".", "_")
        jobs.append(align.Job(str(p), str(prefix), tmpl.shape, args.dpi, force_page, cached))
    print(f"{len(results)} cached, {len(jobs)} to process "
          f"({sum(j.cached is not None for j in jobs)} only need re-rendering)")

    errors = {}
    if jobs:
        t0 = time.time()
        ctx = mp.get_context("spawn")
        features = str(tmpl.features) if templated else None
        with ctx.Pool(min(args.workers, len(jobs)), align.init_worker, (features,), maxtasksperchild=25) as pool:
            for path, res, err in tqdm(pool.imap_unordered(align.align_photo, jobs), total=len(jobs),
                                       desc="Aligning", unit="photo"):
                p = Path(path)
                if err:
                    errors[p] = err
                    tqdm.write(f"ERROR {p.name}: {err}")
                    continue
                results[p] = res
                (dirs["cache"] / f"{p.name}.json").write_text(
                    json.dumps(dict(src=sigs[p], mkey=mkey, rkey=rkey, res=res)))
        print(f"Processed {len(jobs) - len(errors)} photos in {time.time() - t0:.0f}s")
    if errors:
        sys.exit(f"Failed to process {len(errors)} photo(s): {', '.join(p.name for p in errors)}")

    rows = [report.Row(p, times[p], dirs["pages"] / results[p]["image"], results[p]) for p in paths]
    for r in rows:
        report.flag(r, args.min_inliers, args.min_ratio, templated)
    rows, problems, notes = report.group(rows, per_doc, templated)
    report.write_csv(rows, out / "report.csv")

    @lru_cache(maxsize=None)
    def template_page(i):
        return align.render_template_page(tmpl.pdf, i, args.dpi)

    for d in ("review", "contact_sheets"):
        for old in dirs[d].iterdir():
            old.unlink()
    flagged = [r for r in rows if r.flagged]
    docs = [rows[s:s + per_doc] for s in range(0, len(rows), per_doc)]
    tasks = [(report.write_review, r, template_page(r.page - 1) if templated and r.res["page"] else None,
              dirs["review"] / f"{r.image.stem}.png") for r in flagged]
    tasks += [(report.write_contact_sheet, i + 1, d, dirs["contact_sheets"] / f"doc_{i + 1:03d}.jpg")
              for i, d in enumerate(docs)]
    with ThreadPoolExecutor(args.workers) as ex:
        list(tqdm(ex.map(lambda t: t[0](*t[1:]), tasks), total=len(tasks), desc="Writing previews", leave=False))

    pdfs = []
    if problems and not args.allow_problems:
        print("\nPDF NOT BUILT: fix the problems below (overrides file) or re-run with --allow-problems.")
    else:
        pdfs = report.build_pdfs(rows, tmpl.size_pt, per_doc, out / f"{args.name}.pdf",
                                 int(args.max_pdf_mb * 1024 * 1024))

    methods = {m: sum(r.res["method"] == m for r in rows) for m in ("homography", "outline", "resize")}
    print("\n================ SUMMARY ================")
    print(f"Photos processed : {len(rows)} ({len(docs)} documents of {per_doc} pages)")
    print("Aligned by       : " + ", ".join(f"{n} {m}" for m, n in methods.items() if n))
    print(f"Flagged          : {len(flagged)}")
    for r in flagged:
        print(f"  - {r.path.name} (doc {r.doc}, p{r.page}): {' | '.join(r.reasons)}")
    for n in notes:
        print(f"Reordered        : {n}")
    print(f"Grouping problems: {len(problems)}")
    for p in problems:
        print(f"  - {p}")
    print(f"Report           : {out / 'report.csv'}")
    print(f"Contact sheets   : {dirs['contact_sheets']}/ ({len(docs)})")
    print(f"Review images    : {dirs['review']}/ ({len(flagged)})")
    for p in pdfs:
        n = len(rows) if p.name == f"{args.name}.pdf" else None
        print(f"PDF              : {p} ({p.stat().st_size / 1e6:.1f} MB" + (f", {n} pages)" if n else ")"))
    if problems and not args.allow_problems:
        sys.exit(2)


if __name__ == "__main__":
    main()
