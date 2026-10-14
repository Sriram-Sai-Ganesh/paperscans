# paperscan

Turn photos of paper pages into a clean, perspective-corrected, straightened PDF.

Give it a blank PDF that the pages were printed from (`--template`). 
For each photo, it identifies, perspective-corrects and aligns to its matching template page. 
Without a template, it falls back to page-outline detection.

Typical use: photographed exams going into a bulk "Exam / Quiz" assignment upload.

- Reads HEIC, JPG, PNG, TIFF and WebP, applies EXIF orientation, and orders photos by capture time
- Removes shadows and stretches contrast in grayscale, without binarizing, so light pencil survives
- Groups photos into documents and checks that each one is pages 1..N, reporting problems instead of guessing
- Caches results per photo, so re-runs only redo what changed.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Sriram-Sai-Ganesh/paperscans.git && cd paperscans
uv run paperscan photos/ --template blank.pdf
```

Output goes to `paperscan-out/`:

| Path | Contents |
| --- | --- |
| `scans.pdf` | every page in document order, at the template's exact page size |
| `scans_partN.pdf` | written only if `scans.pdf` exceeds `--max-pdf-mb`; split at document boundaries |
| `report.csv` | one row per photo: detected page, match strength, method, flags |
| `contact_sheets/` | one thumbnail grid per document, for a quick visual check |
| `review/` | template / result / overlay for each flagged photo |
| `pages/` | the individual cleaned pages |

Try a small batch first, then look at `contact_sheets/` and `review/`:

```sh
uv run paperscan photos/ -t blank.pdf --limit 16
```

## Common options

```
-o, --out DIR          output folder (default: paperscan-out)
--pages-per-doc N      pages per document (default: template page count)
--expected N           warn if the photo count differs
--workers N            parallel workers, ~0.75 GB RAM each (default: up to 4)
--page-size SIZE       letter | a4 | legal, used only without a template
--allow-problems       build the PDF even if grouping problems were found
--force                ignore the cache
```

Run `uv run paperscan --help` for everything.

## Fixing problems

If a document is missing a page, has a duplicate, or its boundary has shifted, the PDF
is not built. The summary says exactly which photos are affected. To fix it, create
`paperscan-out/overrides.json`:

```json
{"IMG_1600.HEIC": {"page": 3}, "IMG_1601.HEIC": "skip"}
```

`page` forces the match to that template page and `skip` drops the photo. Then run again;
only the changed photos are reprocessed.
