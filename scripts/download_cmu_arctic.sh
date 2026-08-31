#!/usr/bin/env bash
set -euo pipefail

# Download the four native CMU ARCTIC speakers from the official FestVox endpoint used by torchaudio.datasets.CMUARCTIC.
# Usage: bash scripts/download_cmu_arctic.sh [output_root] [parallel_segments]
DATA_ROOT=${1:-data/raw/cmu_arctic}
SEGMENTS=${2:-4}
ARCHIVE_ROOT="$DATA_ROOT/ARCTIC"
BASE_URL='http://www.festvox.org/cmu_arctic/packed'
mkdir -p "$ARCHIVE_ROOT" "$DATA_ROOT/.segments"

declare -A SIZES=([bdl]=73590286 [slt]=81326064 [clb]=90892292 [rms]=92541266)
declare -A SHA256=([bdl]=26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904 [slt]=7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea [clb]=3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6 [rms]=c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4)

download_one() {
  local speaker=$1
  local filename="cmu_us_${speaker}_arctic.tar.bz2"
  local url="${BASE_URL}/${filename}"
  local archive="${ARCHIVE_ROOT}/${filename}"
  local unpacked="${ARCHIVE_ROOT}/cmu_us_${speaker}_arctic"
  local part_dir="${DATA_ROOT}/.segments/${speaker}"
  local size=${SIZES[$speaker]}
  local expected=${SHA256[$speaker]}
  if [[ -f "$archive" ]] && [[ "$(sha256sum "$archive" | awk '{print $1}')" == "$expected" ]]; then
    echo "[$speaker] archive checksum already verified"
  else
    mkdir -p "$part_dir"
    local chunk=$(( (size + SEGMENTS - 1) / SEGMENTS ))
    local i start end expected_chunk actual_chunk
    for ((i=0; i<SEGMENTS; i++)); do
      start=$((i * chunk))
      end=$((start + chunk - 1))
      (( end >= size )) && end=$((size - 1))
      expected_chunk=$((end - start + 1))
      actual_chunk=0
      [[ -f "$part_dir/part_${i}" ]] && actual_chunk=$(stat -c '%s' "$part_dir/part_${i}")
      if (( actual_chunk != expected_chunk )); then
        echo "[$speaker] fetching bytes ${start}-${end}"
        curl --noproxy '*' --fail --location --retry 5 --connect-timeout 30 --max-time 0 --silent --show-error --range "${start}-${end}" "$url" --output "$part_dir/part_${i}"
      fi
      actual_chunk=$(stat -c '%s' "$part_dir/part_${i}")
      (( actual_chunk == expected_chunk )) || { echo "[$speaker] bad chunk ${i}: ${actual_chunk}/${expected_chunk}" >&2; return 1; }
    done
    local assembled="${archive}.assembled"
    : > "$assembled"
    for ((i=0; i<SEGMENTS; i++)); do cat "$part_dir/part_${i}" >> "$assembled"; done
    [[ "$(stat -c '%s' "$assembled")" == "$size" ]] || { echo "[$speaker] assembled size mismatch" >&2; return 1; }
    [[ "$(sha256sum "$assembled" | awk '{print $1}')" == "$expected" ]] || { echo "[$speaker] archive checksum mismatch" >&2; return 1; }
    mv "$assembled" "$archive"
    echo "[$speaker] archive checksum verified"
  fi
  if [[ ! -f "$unpacked/etc/txt.done.data" ]]; then tar -xjf "$archive" -C "$ARCHIVE_ROOT"; fi
  [[ -f "$unpacked/etc/txt.done.data" ]] || { echo "[$speaker] extracted transcript missing" >&2; return 1; }
  local count
  count=$(awk 'END {print NR}' "$unpacked/etc/txt.done.data")
  echo "[$speaker] ready: ${count} transcript entries at ${unpacked}"
}
for speaker in bdl slt clb rms; do download_one "$speaker" & done
wait
echo 'CMU ARCTIC BDL/SLT/CLB/RMS download and extraction complete.'
