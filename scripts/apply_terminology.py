"""Apply terminology and tone-reduction substitutions to main_v2.tex.

Pattern list inferred from the user's manual edits in the first 6 pages:
  - baseline (with Cyrillic declension suffix) -> бейзлайн
  - Hawkes-<Cyrillic suffix>                  -> Хоукс-<Cyrillic suffix>
  - tree-ensemble                              -> бустинг
  - чёрно-ящичные                              -> слабо интерпретируемые
  - гетерогенность пользователей               -> сильное различие в поведении
  - parenthetical English duplicates: (self-exciting), (background), (explosion) -> dropped
  - production-ready (as adjective)            -> dropped

Model names are protected (Scaled-baseline, Joint Hawkes, etc. stay as-is).
"""
from __future__ import annotations
import re
from pathlib import Path

SRC = Path("/Users/amosh239/repo/mkn/diploma/overleaf/main_v2.tex")
text = SRC.read_text(encoding="utf-8")
report: list[tuple[str, int]] = []


def repl(pattern: str, replacement: str, flags: int = 0) -> None:
    global text
    new_text, n = re.subn(pattern, replacement, text, flags=flags)
    if n > 0:
        report.append((f"{pattern!r} -> {replacement!r}", n))
    text = new_text


# === Step 1: protect model names with placeholders ===
# Use \x01 sentinel so they survive all substitutions and are restored at end.
PROTECT = {
    "Scaled-baseline": "\x01MODEL_SB\x01",
    "\\text{baseline}": "\x01MATH_BL\x01",
}
for orig, ph in PROTECT.items():
    text = text.replace(orig, ph)

# === Step 2: baseline'<Cyrillic suffix> -> бейзлайн<suffix> ===
baseline_endings = [
    ("ах",  "ах"),
    ("ам",  "ам"),
    ("ами", "ами"),
    ("ом",  "ом"),
    ("ов",  "ов"),
    ("ы",   "ы"),
    ("у",   "у"),
    ("а",   "а"),
    ("е",   "е"),
]
# longer suffixes first to avoid partial matches
baseline_endings.sort(key=lambda x: -len(x[0]))
for src_end, dst_end in baseline_endings:
    repl(rf"baseline'{src_end}\b", f"бейзлайн{dst_end}")

# === Step 3: Hawkes-<Cyrillic> -> Хоукс-<Cyrillic> ===
repl(r"Hawkes-([а-яА-Я])", r"Хоукс-\1")

# === Step 4: tree-ensemble -> бустинг ===
text = text.replace("современные tree-ensemble методы", "современные бустинговые методы")
text = text.replace("tree-ensemble моделями", "бустинговыми моделями")
text = text.replace("tree-ensemble", "бустинг")
report.append(("tree-ensemble forms", -1))

# === Step 5: чёрно-ящичные модели -> слабо интерпретируемые модели ===
repl(r"чёрно-ящичные", "слабо интерпретируемые")
repl(r"чёрно-ящичными", "слабо интерпретируемыми")

# === Step 6: гетерогенность пользователей -> сильное различие в поведении ===
text = text.replace("гетерогенность пользователей",
                    "сильное различие в поведении пользователей")
text = text.replace("гетерогенности пользователей",
                    "сильного различия в поведении пользователей")

# === Step 7: drop parenthetical English duplicates ===
# Remove " (self-exciting)", " (background)", " (explosion)" etc.
for paren in ["self-exciting", "background", "explosion", "branching ratio"]:
    text = text.replace(f" ({paren})", "")

# === Step 8: production-ready -> drop ===
text = text.replace(" production-ready", "")
text = text.replace("production-ready ", "")

# === Step 9: restore model names ===
for orig, ph in PROTECT.items():
    text = text.replace(ph, orig)

# === Write back ===
SRC.write_text(text, encoding="utf-8")

print("Applied substitutions:")
for pat, n in report:
    print(f"  {pat}: {n if n >= 0 else 'manual'}")
print(f"\nWrote {SRC}, {len(text)} chars")
