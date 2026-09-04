#!/bin/bash
# The carrier for the other six objects: the O-Voxel models this work fits, from the project3
# release, voxelised into grids/<object>.pt the way grids/orange.pt (shipped) was made.
#   bash code/fetch_carrier.sh [object ...]      default: all six
set -eu
SP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROOT=$(dirname "$SP"); PY=${PY:-python}
W=${W:-$ROOT/carrier}; mkdir -p "$W" "$ROOT/grids"
if [ ! -d "$W/trained" ]; then
  echo "== trained_v3.tar (~430 MB), the seven fitted O-Voxel models"
  curl -fL --progress-bar https://github.com/gino6178/project3/releases/download/v1-inputs/trained_v3.tar -o "$W/trained_v3.tar"
  tar xf "$W/trained_v3.tar" -C "$W" && rm "$W/trained_v3.tar"
fi
for o in "${@:-apple bread cake doughnut pomegranate watermelon}"; do
  [ -s "$ROOT/grids/$o.pt" ] && { echo "== grids/$o.pt already here"; continue; }
  echo "== voxelising $o"
  env PLY="$W/trained/$o.ply" META="$W/lattice_meta/$o" N=128 DEV="${DEV:-cuda:0}" OUT="$ROOT/grids/$o.pt" $PY "$SP/voxelize_ov.py"
done
