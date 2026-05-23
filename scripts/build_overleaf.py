"""Build Overleaf-ready LaTeX bundle from diploma/diploma.md.

Outputs:
  overleaf/main.tex       — full LaTeX source
  overleaf/images/*.jpg   — all referenced PNGs converted to JPG
  overleaf/README.md      — build instructions

Usage:
  .venv/bin/python scripts/build_overleaf.py
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "diploma" / "diploma.md"
OUT_DIR = ROOT / "overleaf"
IMG_DIR = OUT_DIR / "images"
TEX_PATH = OUT_DIR / "main.tex"
README_PATH = OUT_DIR / "README.md"

OUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# Use a unique control character as a placeholder for backslash during escaping
BS_PH = "\x01"


def latex_escape(s: str) -> str:
    """Escape special LaTeX chars in plain text."""
    s = s.replace("\\", BS_PH)
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    s = s.replace("<", r"\textless{}")
    s = s.replace(">", r"\textgreater{}")
    s = s.replace(BS_PH, r"\textbackslash{}")
    return s


def tt_escape(s: str) -> str:
    """Escape for inside \\texttt{...}. Also normalises common Unicode chars
    that the default cmtt typewriter font cannot render."""
    # Normalise Unicode punctuation/symbols to ASCII equivalents
    s = s.replace("−", "-")    # U+2212 Unicode minus → ASCII hyphen
    s = s.replace("–", "-")    # U+2013 en-dash → hyphen
    s = s.replace("—", "--")   # U+2014 em-dash → -- (becomes em-dash in TeX)
    s = s.replace("≈", " ~ ")  # rough approximation as fallback
    s = s.replace("≤", "<=")
    s = s.replace("≥", ">=")
    s = s.replace("\\", BS_PH)
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("$", r"\$")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("^", r"\textasciicircum{}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace(BS_PH, r"\textbackslash{}")
    return s


# ============================================================
# STEP 1: image discovery and conversion to JPG
# ============================================================
md_text = MD.read_text(encoding="utf-8")
img_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

img_map: dict[str, str] = {}
counter = 0
for alt, src_path in img_pattern.findall(md_text):
    if src_path in img_map:
        continue
    counter += 1
    src = ROOT / "diploma" / src_path
    if not src.exists():
        print(f"!!! MISSING: {src}")
        continue
    base = src.stem
    new_name = f"fig{counter:02d}_{base}.jpg"
    dest = IMG_DIR / new_name

    img = Image.open(src)
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    max_dim = 1800
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    img.save(dest, "JPEG", quality=85, optimize=True)
    img_map[src_path] = new_name

print(f"Converted {len(img_map)} images")


# ============================================================
# STEP 2: stash math, code, images with placeholders
# ============================================================
math_blocks: list[str] = []
math_inline: list[str] = []
codes: list[str] = []
imgs_used: list[tuple[str, str]] = []


def stash_block_math(m: re.Match) -> str:
    math_blocks.append(m.group(1).strip())
    return f"@@MATHBLK{len(math_blocks)-1}@@"


def stash_inline_math(m: re.Match) -> str:
    math_inline.append(m.group(1))
    return f"@@MATHINL{len(math_inline)-1}@@"


def stash_code(m: re.Match) -> str:
    codes.append(m.group(1))
    return f"@@CODE{len(codes)-1}@@"


def stash_image(m: re.Match) -> str:
    alt, path = m.group(1), m.group(2)
    imgs_used.append((alt, path))
    return f"@@IMG{len(imgs_used)-1}@@"


text = md_text
text = re.sub(r"\$\$(.+?)\$\$", stash_block_math, text, flags=re.DOTALL)
text = re.sub(r"\$([^$\n]+?)\$", stash_inline_math, text)
text = re.sub(r"`([^`\n]+)`", stash_code, text)
text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", stash_image, text)

# Stash Unicode math symbols that appear in plain text.
# After math/code stashing, anything remaining is plain text where these
# Unicode chars would crash pdfLaTeX (no glyph in default font/encoding).
# Replace each with a placeholder that survives latex_escape, restore as math.
UNICODE_TEXT_MAP: dict[str, str] = {
    "≈": r"\approx",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "∈": r"\in",
    "∞": r"\infty",
    "∝": r"\propto",
    "↦": r"\mapsto",
    "→": r"\to",
    "←": r"\leftarrow",
    "⟶": r"\longrightarrow",
    "×": r"\times",
    "·": r"\cdot",
    "⊤": r"\top",
    "⊥": r"\bot",
    "Δ": r"\Delta",
    "Σ": r"\Sigma",
    "Π": r"\Pi",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\varepsilon",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ρ": r"\rho",
    "τ": r"\tau",
    "σ": r"\sigma",
    "ω": r"\omega",
    "φ": r"\varphi",
    "ψ": r"\psi",
    "ℝ": r"\mathbb{R}",
    "ℓ": r"\ell",
    "−": r"-",       # Unicode minus → ASCII hyphen-minus (becomes ‑ in PDF)
    "‖": r"\|",
    "·": r"\cdot",
}
# Two-char combos go first so single-char replacement doesn't break them.
TWO_CHAR_MAP: dict[str, str] = {
    "λ̄": r"\bar{\lambda}",
    "ĉ": r"\hat{c}",
    "λ̂": r"\hat{\lambda}",
    "α̂": r"\hat{\alpha}",
    "μ̂": r"\hat{\mu}",
}
unicode_stash: dict[str, str] = {}  # placeholder -> LaTeX (without surrounding $)

def _stash_unicode_combo(m: re.Match) -> str:
    ch = m.group(0)
    cmd = TWO_CHAR_MAP[ch]
    ph = f"@@U{len(unicode_stash):03d}@@"
    unicode_stash[ph] = cmd
    return ph

if TWO_CHAR_MAP:
    pat = re.compile("|".join(re.escape(k) for k in TWO_CHAR_MAP))
    text = pat.sub(_stash_unicode_combo, text)

def _stash_unicode_char(m: re.Match) -> str:
    ch = m.group(0)
    cmd = UNICODE_TEXT_MAP[ch]
    ph = f"@@U{len(unicode_stash):03d}@@"
    unicode_stash[ph] = cmd
    return ph

if UNICODE_TEXT_MAP:
    pat = re.compile("[" + "".join(re.escape(c) for c in UNICODE_TEXT_MAP) + "]")
    text = pat.sub(_stash_unicode_char, text)


PLACEHOLDER_RE = re.compile(r"@@(?:MATHBLK|MATHINL|CODE|IMG)\d+@@")


def process_inline(s: str) -> str:
    """Process inline text: handle **bold**, then escape, preserving placeholders."""
    # Mark bold with control chars so they survive escaping
    # We use a temp marker that won't appear naturally
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: f"\x02BOLD\x02{m.group(1)}\x02ENDBOLD\x02", s)

    # Split by placeholders and bold markers
    pattern = re.compile(r"(@@(?:MATHBLK|MATHINL|CODE|IMG|U)\d+@@|\x02BOLD\x02.+?\x02ENDBOLD\x02)")
    parts = pattern.split(s)
    out = []
    for p in parts:
        if not p:
            continue
        if p.startswith("@@"):
            out.append(p)
        elif p.startswith("\x02BOLD\x02"):
            inner = p[len("\x02BOLD\x02"):-len("\x02ENDBOLD\x02")]
            inner_proc = process_inline_no_bold(inner)
            out.append("\\textbf{" + inner_proc + "}")
        else:
            out.append(latex_escape(p))
    return "".join(out)


def process_inline_no_bold(s: str) -> str:
    """Inline processing without bold (used inside already-bold context)."""
    pattern = re.compile(r"(@@(?:MATHBLK|MATHINL|CODE|IMG|U)\d+@@)")
    parts = pattern.split(s)
    out = []
    for p in parts:
        if not p:
            continue
        if p.startswith("@@"):
            out.append(p)
        else:
            out.append(latex_escape(p))
    return "".join(out)


def render_table(table_lines: list[str]) -> list[str]:
    if len(table_lines) < 2:
        return ["% bad table"] + table_lines

    def split_row(line: str) -> list[str]:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [c.strip() for c in line.split("|")]

    header_cells = split_row(table_lines[0])
    separator_cells = split_row(table_lines[1])

    aligns = []
    for sep in separator_cells:
        s = sep.strip()
        if s.startswith(":") and s.endswith(":"):
            aligns.append("c")
        elif s.endswith(":"):
            aligns.append("r")
        else:
            aligns.append("l")
    col_spec = "".join(aligns)

    body_rows = [split_row(line) for line in table_lines[2:]]

    out = []
    out.append("")
    out.append("\\begin{table}[ht]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\begin{tabular}{" + col_spec + "}")
    out.append("\\toprule")
    out.append(" & ".join(process_inline(c) for c in header_cells) + " \\\\")
    out.append("\\midrule")
    for row in body_rows:
        while len(row) < len(header_cells):
            row.append("")
        out.append(" & ".join(process_inline(c) for c in row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")
    return out


def render_figure(alt: str, path: str) -> str:
    new_name = img_map.get(path, Path(path).name)
    lines = [
        "",
        "\\begin{figure}[ht]",
        "\\centering",
        "\\includegraphics[width=0.85\\textwidth]{images/" + new_name + "}",
        "\\caption{" + process_inline(alt) + "}",
        "\\end{figure}",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# Main loop: line by line
# ============================================================
lines = text.split("\n")
out_lines: list[str] = []
in_list: str | None = None
in_biblio = False
i = 0


def close_list() -> None:
    global in_list
    if in_list:
        out_lines.append("\\end{" + in_list + "}")
        out_lines.append("")
        in_list = None


while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    if not stripped:
        close_list()
        out_lines.append("")
        i += 1
        continue

    m = re.match(r"^(#{1,6})\s+(.+)$", line)
    if m:
        close_list()
        level = len(m.group(1))
        title = m.group(2).strip()
        # Strip leading "N." or "N.M." or "N.M.K." numbering — LaTeX auto-numbers
        title = re.sub(r"^\d+(?:\.\d+)*\.\s+", "", title)
        title_proc = process_inline(title)

        if level == 1:
            if title == "Аннотация":
                out_lines.append("\\section*{Аннотация}")
                out_lines.append("\\addcontentsline{toc}{section}{Аннотация}")
            elif title == "Содержание":
                out_lines.append("\\tableofcontents")
                i += 1
                while i < len(lines) and not re.match(r"^#\s+", lines[i]):
                    i += 1
                continue
            elif title == "Список литературы":
                in_biblio = True
                out_lines.append("")
                out_lines.append("\\begin{thebibliography}{99}")
                i += 1
                continue
            else:
                out_lines.append("\\section{" + title_proc + "}")
        elif level == 2:
            out_lines.append("\\subsection{" + title_proc + "}")
        else:
            out_lines.append("\\subsubsection{" + title_proc + "}")
        i += 1
        continue

    if in_biblio:
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            num = m.group(1)
            content = process_inline(m.group(2))
            out_lines.append("\\bibitem{ref-" + num + "} " + content)
        i += 1
        continue

    m = re.match(r"^(\s*)-\s+(.+)$", line)
    if m:
        content = m.group(2)
        if in_list != "itemize":
            close_list()
            out_lines.append("\\begin{itemize}")
            in_list = "itemize"
        out_lines.append("  \\item " + process_inline(content))
        i += 1
        continue

    m = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)
    if m:
        content = m.group(3)
        if in_list != "enumerate":
            close_list()
            out_lines.append("\\begin{enumerate}")
            in_list = "enumerate"
        out_lines.append("  \\item " + process_inline(content))
        i += 1
        continue

    if line.startswith("|"):
        table_lines = []
        while i < len(lines) and lines[i].startswith("|"):
            table_lines.append(lines[i])
            i += 1
        close_list()
        out_lines.extend(render_table(table_lines))
        continue

    m = re.match(r"^@@IMG(\d+)@@$", stripped)
    if m:
        close_list()
        idx = int(m.group(1))
        alt, path = imgs_used[idx]
        out_lines.append(render_figure(alt, path))
        i += 1
        continue

    close_list()
    out_lines.append(process_inline(line))
    i += 1

close_list()
if in_biblio:
    out_lines.append("\\end{thebibliography}")

text_body = "\n".join(out_lines)

# ============================================================
# Restore placeholders
# ============================================================

def restore_code(m: re.Match) -> str:
    idx = int(m.group(1))
    content = codes[idx]
    # Heuristic: if code span contains LaTeX-like commands or Unicode math symbols
    # that won't render in cmtt typewriter font, treat it as inline math instead.
    UNICODE_MATH = set("ℝℕℤℚℂℵλμαβγδρτφψω⟨⟩∈∋∀∃∞⊆⊂⊇⊃∪∩∖⊕⊗∇∂≤≥≠≈≡⊤⊥")
    has_backslash = "\\" in content
    has_math_sym = any(c in UNICODE_MATH for c in content)
    if has_backslash or has_math_sym:
        # Try to render as math: replace some common unicode chars with TeX
        m_text = content
        replacements = {
            "ℝ": r"\mathbb{R}",
            "ℕ": r"\mathbb{N}",
            "ℤ": r"\mathbb{Z}",
            "ℚ": r"\mathbb{Q}",
            "ℂ": r"\mathbb{C}",
            "×": r"\times ",
            "·": r"\cdot ",
            "≤": r"\leq ",
            "≥": r"\geq ",
            "≠": r"\neq ",
            "≈": r"\approx ",
            "≡": r"\equiv ",
            "∈": r"\in ",
            "∞": r"\infty ",
            "⊤": r"\top ",
            "→": r"\rightarrow ",
            "←": r"\leftarrow ",
            "α": r"\alpha ",
            "β": r"\beta ",
            "γ": r"\gamma ",
            "δ": r"\delta ",
            "λ": r"\lambda ",
            "μ": r"\mu ",
            "ρ": r"\rho ",
            "τ": r"\tau ",
            "φ": r"\varphi ",
            "ψ": r"\psi ",
            "ω": r"\omega ",
        }
        for k, v in replacements.items():
            m_text = m_text.replace(k, v)
        # `_` is a subscript in math mode. We want it literal only when it's
        # part of a multi-token identifier like `to_cart`. To avoid escaping
        # subscripts in genuine LaTeX commands like `\ell_2` or `\lambda_u`,
        # first protect `\command_<...>` patterns, then escape identifier
        # underscores, then restore.
        protected: list[str] = []
        def _protect(mt: re.Match) -> str:
            protected.append(mt.group(0))
            return f"\x03P{len(protected)-1}\x03"
        m_text = re.sub(r"\\[A-Za-z]+_\{[^}]*\}", _protect, m_text)
        m_text = re.sub(r"\\[A-Za-z]+_[A-Za-z0-9]+", _protect, m_text)
        m_text = re.sub(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])", r"\\_", m_text)
        m_text = re.sub(r"\x03P(\d+)\x03", lambda mt: protected[int(mt.group(1))], m_text)
        m_text = m_text.replace("%", r"\%")
        m_text = m_text.replace("#", r"\#")
        m_text = m_text.replace("&", r"\&")
        return "$" + m_text + "$"
    return "\\texttt{" + tt_escape(content) + "}"


def restore_inline_math(m: re.Match) -> str:
    idx = int(m.group(1))
    return "$" + math_inline[idx] + "$"


def restore_block_math(m: re.Match) -> str:
    idx = int(m.group(1))
    return "\\[\n" + math_blocks[idx] + "\n\\]"


text_body = re.sub(r"@@CODE(\d+)@@", restore_code, text_body)
text_body = re.sub(r"@@MATHINL(\d+)@@", restore_inline_math, text_body)
text_body = re.sub(r"@@MATHBLK(\d+)@@", restore_block_math, text_body)

# Restore Unicode math symbols as inline math $...$
for ph, cmd in unicode_stash.items():
    text_body = text_body.replace(ph, "$" + cmd + "$")


# ============================================================
# Final LaTeX document
# ============================================================
PREAMBLE = r"""\documentclass[12pt,a4paper]{article}

\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian,english]{babel}
\usepackage[a4paper, margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{
  colorlinks=true,
  linkcolor=black,
  urlcolor=blue,
  citecolor=blue,
}
\renewcommand{\arraystretch}{1.15}
\setlength{\parindent}{1.25em}
\setlength{\parskip}{0.4ex}
\sloppy

\title{Применимость процессов Хокса для предсказания пользовательской активности в e-commerce}
\author{Алексей Мошкин\\\small МКН СПбГУ\\\small Научный руководитель: Николаев Максим Сергеевич}
\date{2026}

\begin{document}

\maketitle

"""

POSTAMBLE = "\n\\end{document}\n"

TEX_PATH.write_text(PREAMBLE + text_body + POSTAMBLE, encoding="utf-8")
print(f"Wrote {TEX_PATH} ({TEX_PATH.stat().st_size / 1024:.1f} KB)")


README_PATH.write_text(
    """# Overleaf-ready LaTeX bundle

Этот каталог содержит LaTeX-версию диплома, готовую для загрузки в Overleaf
или локальной сборки через `pdflatex` / `latexmk`.

## Файлы

- `main.tex` — основной исходник (русский, T2A + babel).
- `images/` — все рисунки в JPG (сжатые до ≤1800px по большей стороне).
- `README.md` — этот файл.

## Загрузка в Overleaf

1. На Overleaf: *New Project* → *Upload Project* → выбрать ZIP всей папки `overleaf/`.
2. Project menu → `Settings` → `Compiler: pdfLaTeX`, `Main document: main.tex`,
   `Spell check: Russian`.
3. Нажать *Recompile* (двойной прогон автоматический).

## Локальная сборка

```bash
cd overleaf
latexmk -pdf main.tex
# или вручную:
pdflatex main.tex && pdflatex main.tex
```

Двойной запуск нужен для нумерации `\\tableofcontents`.

## Как пересоздать этот каталог из `diploma/diploma.md`

```bash
.venv/bin/python scripts/build_overleaf.py
```

Скрипт читает `diploma/diploma.md`, конвертирует все PNG в JPG в `overleaf/images/`,
и регенерирует `main.tex`.
""",
    encoding="utf-8",
)
print(f"Wrote {README_PATH}")
