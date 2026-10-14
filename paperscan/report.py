"""Flagging, grouping into documents, CSV report, review images, contact sheets and PDF output."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Row:
    path: Path
    captured: str
    image: Path
    res: dict
    flagged: bool = False
    reasons: list[str] = field(default_factory=list)
    doc: int = 0
    position: int = 0

    @property
    def page(self) -> int:
        return self.res["page"] or self.position


def flag(row: Row, min_inliers: int, min_ratio: float, templated: bool):
    r = row.res
    if templated:
        if r["method"] != "homography":
            row.reasons.append(f"template alignment failed ({r['sanity']}); used {r['method']}, page number unreliable")
        if r["inliers"] < min_inliers:
            row.reasons.append(f"low inliers ({r['inliers']} < {min_inliers})")
        if r["second_inliers"] and r["inliers"] / r["second_inliers"] < min_ratio:
            row.reasons.append(f"ambiguous page: p{r['page']} {r['inliers']} vs p{r['second_page']} "
                               f"{r['second_inliers']} inliers (ratio < {min_ratio})")
    elif r["method"] == "resize":
        row.reasons.append("no page outline found; photo was only resized")
    elif r.get("outline_check"):
        row.reasons.append(f"suspicious page outline: {r['outline_check']}")
    row.flagged = bool(row.reasons)


def group(rows: list[Row], per_doc: int, templated: bool) -> tuple[list[Row], list[str], list[str]]:
    """Split rows into documents of per_doc pages. Returns (ordered rows, problems, notes).

    With a template, a document whose pages are a confident permutation of 1..N is
    reordered; anything else (missing, duplicate, shifted boundary) is reported, never fixed.
    """
    problems, notes, out = [], [], []
    if len(rows) % per_doc:
        problems.append(f"{len(rows)} photos is not a multiple of {per_doc} "
                        f"({len(rows) // per_doc} full documents + {len(rows) % per_doc} extra).")
    expected = list(range(1, per_doc + 1))
    for g in range(0, len(rows), per_doc):
        grp, doc = rows[g:g + per_doc], g // per_doc + 1
        for i, r in enumerate(grp):
            r.doc, r.position = doc, i + 1
        if not templated:
            out.extend(grp)
            continue
        pages = [r.res["page"] for r in grp]
        if pages == expected:
            out.extend(grp)
            continue
        if sorted(pages) == expected and not any(r.flagged for r in grp):
            grp = sorted(grp, key=lambda r: r.res["page"])
            for i, r in enumerate(grp):
                r.position = i + 1
            notes.append(f"Document {doc:02d}: pages were in order {pages}; reordered to 1-{per_doc}.")
            out.extend(grp)
            continue
        desc = []
        if len(grp) < per_doc:
            desc.append(f"only {len(grp)} photos in final group")
        missing = [p for p in expected if p not in pages]
        dups = sorted({p for p in pages if pages.count(p) > 1})
        if missing:
            desc.append(f"missing page(s) {missing}")
        if dups:
            desc.append(f"duplicate page(s) {dups}")
        ones = [i + 1 for i, p in enumerate(pages) if p == 1 and i > 0]
        if ones:
            desc.append(f"page 1 at slot(s) {ones}: document boundary probably shifted "
                        f"(missing/extra photo before {grp[ones[0] - 1].path.name})")
        if sorted(pages) == expected:
            desc.append("out of order but contains flagged matches, so not auto-reordered: "
                        + ", ".join(r.path.name for r in grp if r.flagged))
        detected = ", ".join(f"{r.path.name}=p{r.res['page']}" for r in grp)
        problems.append(f"Document {doc:02d} (photos {g + 1}-{g + len(grp)}): {'; '.join(desc)}.\n"
                        f"      detected: {detected}")
        out.extend(grp)
    return out, problems, notes


def write_csv(rows: list[Row], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_file", "capture_time", "document", "position", "detected_page", "best_inliers",
                    "second_best_inliers", "second_best_page", "method", "align_err_px", "flagged", "reason"])
        for r in rows:
            w.writerow([r.path.name, r.captured, r.doc, r.position, r.res["page"] or "", r.res["inliers"],
                        r.res["second_inliers"], r.res["second_page"] or "", r.res["method"],
                        r.res.get("err_after_px", ""), "yes" if r.flagged else "no", " | ".join(r.reasons)])


def _font(size: int):
    return ImageFont.load_default(size=size)


def write_review(row: Row, template_page: np.ndarray | None, out: Path):
    """Template | result | overlay (template ink in red), or just the result without a template."""
    res = np.asarray(Image.open(row.image).convert("L"))
    panels = [cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)]
    if template_page is not None and template_page.shape == res.shape:
        overlay = panels[0].copy()
        ink = template_page < 160
        overlay[ink] = (0.4 * overlay[ink] + 0.6 * np.array([230, 0, 0])).astype(np.uint8)
        panels = [cv2.cvtColor(template_page, cv2.COLOR_GRAY2RGB), panels[0], overlay]
    h, w = res.shape
    sheet = np.full((h + 70, len(panels) * (w + 20) - 20, 3), 255, np.uint8)
    for i, p in enumerate(panels):
        sheet[70:, i * (w + 20):i * (w + 20) + w] = p
    im = Image.fromarray(sheet)
    ImageDraw.Draw(im).text((10, 10), f"{row.path.name}  doc {row.doc}  p{row.page}  [{row.res['method']}]  "
                            + " | ".join(row.reasons), fill=(200, 0, 0), font=_font(28))
    im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(out, optimize=True)


def write_contact_sheet(doc: int, grp: list[Row], out: Path, cols: int = 4):
    tw, th, lab = 425, 550, 44
    n_rows = (len(grp) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, n_rows * (th + lab + 10) + 50), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), f"Document {doc:02d}", fill="black", font=_font(30))
    for i, r in enumerate(grp):
        with Image.open(r.image) as im:
            im.draft("L", (tw, th))
            im = im.convert("RGB")
            im.thumbnail((tw, th))
        x, y = 10 + (i % cols) * (tw + 10), 50 + (i // cols) * (th + lab + 10)
        color = (0, 120, 0) if r.page == i + 1 and not r.flagged else (210, 0, 0)
        d.text((x, y + 4), f"p{r.page}  {r.path.stem}" + ("  FLAG" if r.flagged else ""), fill=color, font=_font(26))
        sheet.paste(im, (x, y + lab))
        d.rectangle([x - 1, y + lab - 1, x + im.width, y + lab + im.height], outline=color, width=3)
    sheet.save(out, quality=85)


def write_pdf(images: list[Path], page_size_pt: tuple[float, float], out: Path):
    """Stream JPEGs into a PDF, one full-bleed image per page, without loading them all into memory."""
    W, H = page_size_pt
    tmp = out.with_suffix(".pdf.tmp")
    offsets = {}
    with open(tmp, "wb") as f:
        def obj(num: int, body: bytes, stream: Path | bytes | None = None):
            offsets[num] = f.tell()
            f.write(b"%d 0 obj\n" % num + body)
            if stream is not None:
                f.write(b"\nstream\n")
                if isinstance(stream, bytes):
                    f.write(stream)
                else:
                    with open(stream, "rb") as src:
                        shutil.copyfileobj(src, f)
                f.write(b"\nendstream")
            f.write(b"\nendobj\n")

        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        kids = []
        for i, img in enumerate(images):
            im_n, ct_n, pg_n = 3 + 3 * i, 4 + 3 * i, 5 + 3 * i
            with Image.open(img) as im:
                if im.format != "JPEG":
                    raise ValueError(f"{img} is not a JPEG")
                (w, h), cs = im.size, b"/DeviceGray" if im.mode == "L" else b"/DeviceRGB"
            obj(im_n, b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s "
                      b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>"
                % (w, h, cs, img.stat().st_size), img)
            content = b"q %.4f 0 0 %.4f 0 0 cm /Im0 Do Q" % (W, H)
            obj(ct_n, b"<< /Length %d >>" % len(content), content)
            obj(pg_n, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.4f %.4f] "
                      b"/Resources << /XObject << /Im0 %d 0 R >> >> /Contents %d 0 R >>" % (W, H, im_n, ct_n))
            kids.append(b"%d 0 R" % pg_n)
        obj(2, b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(kids), len(kids)))
        xref = f.tell()
        size = max(offsets) + 1
        f.write(b"xref\n0 %d\n0000000000 65535 f \n" % size)
        f.write(b"".join(b"%010d 00000 n \n" % offsets[n] for n in range(1, size)))
        f.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref))
    tmp.replace(out)


def build_pdfs(rows: list[Row], page_size_pt, per_doc: int, out: Path, max_bytes: int) -> list[Path]:
    """Write the full PDF, plus part files split at document boundaries if it exceeds max_bytes."""
    images = [r.image for r in rows]
    write_pdf(images, page_size_pt, out)
    written = [out]
    for old in out.parent.glob(f"{out.stem}_part*.pdf"):
        old.unlink()
    if out.stat().st_size <= max_bytes:
        return written
    sizes = [p.stat().st_size for p in images]
    chunks, cur, cur_size = [], [], 0
    for s in range(0, len(images), per_doc):
        doc_size = sum(sizes[s:s + per_doc])
        if cur and cur_size + doc_size > max_bytes * 0.97:
            chunks.append(cur)
            cur, cur_size = [], 0
        cur += images[s:s + per_doc]
        cur_size += doc_size
    chunks.append(cur)
    for i, ch in enumerate(chunks, 1):
        p = out.with_name(f"{out.stem}_part{i}.pdf")
        write_pdf(ch, page_size_pt, p)
        written.append(p)
    return written
