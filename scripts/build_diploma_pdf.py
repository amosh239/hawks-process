"""Build diploma.pdf from diploma.md.

Steps:
  1. Render diploma/diploma.md to HTML using `markdown` (venv).
  2. Call system `weasyprint` to convert HTML -> PDF.

Outputs:
  diploma/diploma.html  (intermediate)
  diploma/diploma.pdf
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "diploma" / "diploma.md"
HTML_PATH = ROOT / "diploma" / "diploma.html"
PDF_PATH = ROOT / "diploma" / "diploma.pdf"

WEASYPRINT = "/opt/homebrew/bin/weasyprint"

CSS_TEXT = """
@page {
    size: A4;
    margin: 22mm 18mm 22mm 22mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "Times New Roman", "DejaVu Serif", serif;
        font-size: 10pt;
        color: #555;
    }
}
body {
    font-family: "Times New Roman", "DejaVu Serif", serif;
    font-size: 11.5pt;
    line-height: 1.45;
    text-align: justify;
    color: #111;
}
h1 {
    font-size: 18pt;
    margin-top: 1.4em;
    margin-bottom: 0.7em;
    page-break-before: always;
    page-break-after: avoid;
}
h1:first-of-type {
    page-break-before: avoid;
}
h2 {
    font-size: 14pt;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}
h3 {
    font-size: 12.5pt;
    margin-top: 0.9em;
    margin-bottom: 0.3em;
    page-break-after: avoid;
}
p {
    margin: 0.3em 0 0.5em 0;
    orphans: 3;
    widows: 3;
}
ul, ol { margin: 0.3em 0 0.6em 0; }
li { margin: 0.15em 0; }
code {
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 0.92em;
    background: #f4f4f4;
    padding: 0 0.2em;
    border-radius: 2px;
}
pre {
    background: #f4f4f4;
    padding: 0.6em 0.8em;
    border-radius: 3px;
    font-size: 9.5pt;
    overflow-x: auto;
    page-break-inside: avoid;
}
table {
    border-collapse: collapse;
    margin: 0.5em auto;
    font-size: 10.5pt;
    page-break-inside: avoid;
}
table th, table td {
    border: 1px solid #aaa;
    padding: 3px 7px;
    text-align: left;
    vertical-align: top;
}
table th { background: #eef2f7; font-weight: bold; }
img {
    max-width: 95%;
    height: auto;
    display: block;
    margin: 0.6em auto;
    page-break-inside: avoid;
}
blockquote {
    border-left: 3px solid #bbb;
    padding-left: 0.8em;
    color: #444;
    margin: 0.4em 0;
}
.math {
    font-family: "Menlo", "DejaVu Sans Mono", monospace;
    background: #f8f8f8;
    padding: 0.4em 0.8em;
    margin: 0.4em auto;
    font-size: 10.5pt;
    border-left: 2px solid #ccc;
    page-break-inside: avoid;
    white-space: pre-wrap;
    text-align: left;
    display: block;
}
.math-inline {
    font-family: "Menlo", "DejaVu Sans Mono", monospace;
    font-size: 0.95em;
}
"""


def render_html(md_text: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
            "md_in_html",
        ],
        output_format="html5",
    )
    head = f"""<!DOCTYPE html>
<html lang=\"ru\">
<head>
<meta charset=\"utf-8\">
<title>Диплом</title>
<style>
{CSS_TEXT}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    return head


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")

    # Pre-process: wrap $$...$$ block math in <div class="math"> for nicer rendering
    import re
    md_text = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: f"\n\n<div class=\"math\">{m.group(1).strip()}</div>\n\n",
        md_text,
        flags=re.DOTALL,
    )
    # Inline math: $...$ -> <span class="math-inline">...</span>. Skip $-signs that
    # look like currency by requiring no whitespace adjacent to the dollar signs.
    md_text = re.sub(
        r"\$(?=\S)([^$\n]+?)(?<=\S)\$",
        lambda m: f"<span class=\"math-inline\">{m.group(1)}</span>",
        md_text,
    )

    html = render_html(md_text)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote intermediate {HTML_PATH}")

    # weasyprint via CLI; base URL = diploma/ so relative image paths resolve
    result = subprocess.run(
        [
            WEASYPRINT,
            str(HTML_PATH),
            str(PDF_PATH),
            "--base-url",
            str(MD_PATH.parent) + "/",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("weasyprint stderr:")
        print(result.stderr)
        raise SystemExit(result.returncode)
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
