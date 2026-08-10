#!/usr/bin/env python3
# 10 Aug 2026 - E13C-DOC-01: Retained the project report renderer required by
# the research-report skill; it does not alter experiment runtime behavior.
"""Render an analysis Markdown report to PDF and optionally upload it to Dropbox.

Keeps every report on the same pipeline so they look alike and land in the same
places:

    analysis/<YYYY-MM-DD>_<slug>.md   ->   analysis/assets/<same-stem>.pdf

Figures referenced as `assets/<file>.png` resolve relative to `analysis/`, which
is also where the PDF is written, so links keep working in both the Markdown and
the rendered document.

Usage:
    python tools/reports/publish_report.py analysis/2026-08-08_my_report.md
    python tools/reports/publish_report.py analysis/2026-08-08_my_report.md --upload
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FIGURE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def check_pandoc() -> None:
    if shutil.which("pandoc") is None:
        raise SystemExit(
            "pandoc not found. Install it, or render manually and pass --skip-pdf."
        )


def stage_figures(markdown: Path, assets: Path) -> list[str]:
    """Copy any referenced figure that is not already beside the PDF."""
    staged = []
    for match in FIGURE_RE.finditer(markdown.read_text(encoding="utf-8")):
        ref = match.group(1).split()[0].strip("<>")
        if ref.startswith(("http://", "https://")):
            continue
        source = (markdown.parent / ref).resolve()
        if source.is_file():
            target = assets / source.name
            if source != target:
                assets.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                staged.append(source.name)
        else:
            print(f"[warn] figure not found, PDF will show a placeholder: {ref}",
                  file=sys.stderr)
    return staged


def render(markdown: Path, pdf: Path, engine: str) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    base = [
        "pandoc", str(markdown), "-o", str(pdf),
        "-V", "geometry:margin=2cm", "-V", "fontsize=9pt",
        "-V", "colorlinks=true",
        "--resource-path", str(markdown.parent),
    ]
    attempts = [base + ["--pdf-engine", engine], base]
    last = None
    for cmd in attempts:
        done = subprocess.run(cmd, capture_output=True, text=True)
        # Unicode glyph warnings are noise; anything else is worth showing.
        noise = [l for l in done.stderr.splitlines() if "Missing character" not in l]
        if done.returncode == 0 and pdf.is_file():
            for line in noise[:5]:
                print(f"[pandoc] {line}", file=sys.stderr)
            return
        last = "\n".join(noise[-15:]) or done.stderr[-800:]
    raise SystemExit(f"pandoc failed to produce {pdf}\n{last}")


def upload(pdf: Path) -> int:
    tool = PROJECT / "tools" / "dropbox" / "upload_to_dropbox.py"
    if not tool.is_file():
        raise SystemExit(f"Dropbox tool not found: {tool}")
    print("\n--- Dropbox upload ---")
    done = subprocess.run([sys.executable, str(tool), str(pdf)],
                          text=True, capture_output=True)
    sys.stdout.write(done.stdout)
    if done.stderr:
        sys.stderr.write(done.stderr)
    if done.returncode != 0:
        return done.returncode
    # The upload is only complete if a temporary link came back - the reply to
    # the user must quote it, so fail loudly when it is absent.
    if "https://" not in done.stdout:
        print("[error] upload reported success but returned no temporary link",
              file=sys.stderr)
        return 1
    print("\nPaste the link above into the reply and note it expires in ~4 hours.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("markdown", help="path to the report .md")
    ap.add_argument("--output", default=None, help="PDF path (default analysis/assets/<stem>.pdf)")
    ap.add_argument("--engine", default="xelatex", help="pandoc --pdf-engine (default xelatex)")
    ap.add_argument("--upload", action="store_true", help="upload the PDF to Dropbox")
    ap.add_argument("--skip-pdf", action="store_true", help="only stage figures")
    args = ap.parse_args()

    markdown = Path(args.markdown).resolve()
    if not markdown.is_file():
        raise SystemExit(f"Report not found: {markdown}")

    pdf = Path(args.output).resolve() if args.output else \
        markdown.parent / "assets" / f"{markdown.stem}.pdf"

    staged = stage_figures(markdown, pdf.parent)
    if staged:
        print(f"staged {len(staged)} figure(s): {', '.join(staged)}")

    if args.skip_pdf:
        return
    check_pandoc()
    render(markdown, pdf, args.engine)
    print(f"PDF: {pdf}  ({pdf.stat().st_size} bytes)")

    if args.upload:
        raise SystemExit(upload(pdf))


if __name__ == "__main__":
    main()
