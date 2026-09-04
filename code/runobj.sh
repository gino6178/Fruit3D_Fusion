#!/bin/bash
# One program per object: prep -> priors (longitudinal 30k, polar 4k, the recipe found on the
# orange) -> the cylinder -> scores and a composite.  $1 = object, $2 = GPU.  AXD comes from
# axis.env (a data declaration written once by the occupancy check), default 1.
SP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)          # code/
ROOT=$(dirname "$SP")
PY=${PY:-python}                                          # the environment's interpreter
o=$1; G=cuda:${2:-0}; cd "$SP"
O=${OUT:-$ROOT/runs/$o}; mkdir -p "$O"
GRID=${GRID:-$ROOT/grids/$o.pt}                           # built by voxelize_ov.py
AXD=$(grep "^$o " axis.env 2>/dev/null | awk '{print $2}'); AXD=${AXD:-1}; export AXD
OUT=$O $PY prep_obj.py $o
CKV=none; CKH=none
if ls $O/spl_long/*.png >/dev/null 2>&1; then
  env FAM=long MULT=1,2 PDIR=$O/spl_long STEPS=30000 OUT=$O/u_long DEV=$G GRID=$GRID LR=5e-4 PMIX=1.0 MASKED=0 BS=4 $PY sd3d_train.py > $O/u_long.log 2>&1; CKV=$O/u_long/model.pt; fi
if ls $O/polar_spl_trans/*.png >/dev/null 2>&1; then
  env PDIR=$O/polar_spl_trans OUT=$O/p_trans DEV=$G STEPS=4000 $PY polar_train.py > $O/p_trans.log 2>&1; CKH=$O/p_trans/model.pt; fi
env T0H=0.3 T0V=0.3 WFAR=0.1 NSTEP=100 DEV=$G GRID=$GRID CKV=$CKV CKH=$CKH OUT=$O/cyl $PY x3dcyl.py > $O/cyl.log 2>&1
env OBJDIR=$O $PY dsscore.py "$GRID" $O/cyl/state.pt > $O/score.log 2>&1
env OBJDIR=$O ${PY_SCORE:-$PY} dsrun.py 2>&1 | grep -v Warning >> $O/score.log
echo "OBJ_DONE $o" >> $O/score.log; cat $O/SPLIT.txt $O/score.log
