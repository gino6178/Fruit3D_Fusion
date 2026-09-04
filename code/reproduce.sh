#!/bin/bash
# Reproduce the paper's orange from the priors in code/baseline/ -- no training, minutes on one GPU.
#   bash code/reproduce.sh [gpu]
# Writes runs/orange_repro/ and prints the paper's own scorers over the carrier and the result.
set -eu
SP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROOT=$(dirname "$SP"); cd "$SP"
PY=${PY:-python}; G=cuda:${1:-0}; O=${OUT:-$ROOT/runs/orange_repro}; GRID=${GRID:-$ROOT/grids/orange.pt}
mkdir -p "$O"
echo "== sampling the interior (100 steps, ~4 min on an L40)"
env T0H=0.3 T0V=0.3 R0=0.15 WFAR=0.1 DCFIX=16 NPHI=512 NSTEP=100 DEV="$G" \
    GRID="$GRID" CKV="$SP/baseline/prior_long.pt" CKH="$SP/baseline/prior_trans_polar.pt" \
    MULTV=1,2 MULTH=1,2,4 OUT="$O/cyl" $PY x3dcyl.py
echo
echo "== texture, by region, against the photographs (paper section 4)"; $PY fidelity.py "$GRID" "$O/cyl/state.pt"
echo "== flesh colour";                                                  $PY colour.py   "$GRID" "$O/cyl/state.pt"
echo "== horizontal banding";                                            $PY hband.py    "$GRID" "$O/cyl/state.pt"
echo "== the transverse view";                                           $PY transverse.py "$GRID" "$O/cyl/state.pt"
echo
echo "faces: $O/cyl/xfill.png   volume: $O/cyl/state.pt"
echo "for the held-out DreamSim of Table 1, see README (it needs its own environment)"
