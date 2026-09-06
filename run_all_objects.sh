#!/bin/bash
# The remaining objects through the whole pipeline on one GPU: grid -> priors -> cylinder -> material field -> force curves -> eval.
ROOT=/home/gino/project/Fruit3D_Fusion; D=/home/gino/project/phy_inside_data; PY=/home/gino/miniconda3/envs/genrecon/bin/python; G=${GPU:-1}
export PHOTOS=/home/gino/project/FruitNinja_clean/data_finetune_images
for o in watermelon apple bread cake doughnut pomegranate; do
  echo "=== $o $(date +%H:%M)"; mkdir -p $ROOT/runs/$o
  [ -f $ROOT/grids/$o.pt ] || (cd $ROOT/code && PLY=$D/trained/$o.ply META=$D/lattice_meta/$o OUT=$ROOT/grids/$o.pt DEV=cuda:$G $PY voxelize_ov.py > $ROOT/runs/$o/grid.log 2>&1) || { echo "grid failed $o"; continue; }
  [ -f $ROOT/runs/$o/cyl/state.pt ] || (cd $ROOT/code && PY=$PY bash runobj.sh $o $G > $ROOT/runs/$o/runobj.log 2>&1)
  [ -f $ROOT/runs/$o/cyl/state.pt ] || { echo "cylinder failed $o"; tail -3 $ROOT/runs/$o/cyl.log 2>/dev/null; continue; }
  cd $ROOT && CUDA_VISIBLE_DEVICES=$G $PY material/stage1_backproject.py runs/$o/cyl/state.pt grids/$o.pt runs/$o/material > runs/$o/stage1_backproject.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $PY material/stage1_photo2asset.py runs/$o runs/$o/cyl/state.pt grids/$o.pt runs/$o/material_v2 > runs/$o/photo2asset.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $PY material/stage2_field.py runs/$o/material_v2/cells_photo.npz runs/$o/material/cells.npz runs/$o/cyl/state.pt grids/$o.pt runs/$o/material_v2 > runs/$o/stage2.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $PY material/stage4_cutforce.py runs/$o/material_v2/field_cells.npz runs/$o/material_v2/field_cells.npz runs/$o/cyl/state.pt grids/$o.pt runs/$o/stage4 > runs/$o/stage4.log 2>&1
  echo "--- $o done $(date +%H:%M): K=$(grep -o 'chosen K=[0-9]*' runs/$o/photo2asset.log) $(grep -o 'agreement [0-9.]*' runs/$o/photo2asset.log | tail -1)"
done
echo ALL_OBJECTS_DONE
