#!/usr/bin/env bash
set -euo pipefail
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUS_TEX_BIN="$PAPER_DIR/../.texlive/bin/x86_64-linux"
if [[ -x "$MUS_TEX_BIN/pdflatex" ]]; then
  export PATH="$MUS_TEX_BIN:$PATH"
fi
cd "$PAPER_DIR"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass1.stdout
(cd build && BIBINPUTS="$PAPER_DIR:" BSTINPUTS="$PAPER_DIR:" bibtex main > bibtex.stdout)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass2.stdout
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass3.stdout
cp build/main.pdf paper.pdf
printf 'Compiled: %s/paper.pdf\n' "$PAPER_DIR"
