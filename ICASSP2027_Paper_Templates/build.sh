#!/usr/bin/env bash
set -euo pipefail
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUS_TEX_BIN="$PAPER_DIR/../.texlive/bin/x86_64-linux"
if [[ -x "$MUS_TEX_BIN/pdflatex" ]]; then
  export PATH="$MUS_TEX_BIN:$PATH"
fi
cd "$PAPER_DIR"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build Template.tex > build/pass1.stdout
(cd build && BIBINPUTS="$PAPER_DIR:" BSTINPUTS="$PAPER_DIR:" bibtex Template > bibtex.stdout)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build Template.tex > build/pass2.stdout
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build Template.tex > build/pass3.stdout
# Normalize imported PDF dictionaries before distributing the compiled paper.
# qpdf status 3 means a repaired warning; the normalized file must then be clean.
qpdf build/Template.pdf build/Template.normalized.pdf > build/pdf_normalization.log 2>&1 || [[ $? -eq 3 ]]
qpdf --check build/Template.normalized.pdf > build/pdf_check.log 2>&1
mv build/Template.normalized.pdf build/Template.pdf
cp build/Template.pdf Template.pdf
printf 'Compiled: %s/Template.pdf\n' "$PAPER_DIR"
