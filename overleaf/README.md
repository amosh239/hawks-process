# Overleaf-ready LaTeX bundle

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

Двойной запуск нужен для нумерации `\tableofcontents`.

## Как пересоздать этот каталог из `diploma/diploma.md`

```bash
.venv/bin/python scripts/build_overleaf.py
```

Скрипт читает `diploma/diploma.md`, конвертирует все PNG в JPG в `overleaf/images/`,
и регенерирует `main.tex`.
