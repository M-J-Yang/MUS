#!/usr/bin/env bash

# Load the project-local TeX Live installation.
export MUS_TEXLIVE_ROOT="/data/zb/ymj/MUS/.texlive"
export PATH="$MUS_TEXLIVE_ROOT/bin/x86_64-linux:$PATH"
export TEXMFVAR="$MUS_TEXLIVE_ROOT/texmf-var"
export TEXMFCONFIG="$MUS_TEXLIVE_ROOT/texmf-config"

echo "Using TeX Live from: $MUS_TEXLIVE_ROOT"
xelatex --version | head -n 2
