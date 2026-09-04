# Fruit3D Fusion

An object's interior, synthesised from **two or three unposed photographs per cut family** — a few
longitudinal cuts and a few transverse ones, of *different specimens* — and written back into a
cuttable voxel asset. No 3-D data, no 3-D training, no per-object generative fine-tuning.

The interior is recovered as a **product of two single-image diffusion priors over the slices of a
cylindrical lattice** `A[r, φ, z]`, where both cut families are exact axis-aligned slices of one
shared state and the object's radial anatomy — segment membranes, the columella — becomes
stationary, axis-aligned texture that a patch prior can hold. A directly fitted occupancy lattice
supplies the layout the priors cannot invent, and its shell is never written.

**Paper and figures:** <https://gino6178.github.io/Fruit3D_Fusion/> (`index.html` in this repository).

On held-out photographs of specimens the priors never saw, the orange scores **0.112 / 0.062 /
0.087** (DreamSim, longitudinal / transverse / mean, lower is better) against the fitted carrier's
0.148 / 0.077 / 0.113 and FruitNinja's 0.154. `code/reproduce.sh` reproduces exactly that, in
minutes, from the priors in this repository.

## What is here

```
index.html assets/     the paper
code/                  the method, its scorers, and the two scripts that reproduce the paper
code/baseline/         the priors the paper's numbers come from (21 MB)
data/orange/           the orange's declared split: three photographs per family that train,
                       three per family held out and never seen by the priors
grids/orange.pt        the fitted carrier for the orange (fp16, 18 MB)
```

Nothing else is in here. The route that preceded this method, the carrier's own paper and its
pipeline live in <https://github.com/gino6178/project3>.

## Install

```bash
conda create -y -n f3d python=3.10 && conda activate f3d
pip install -r requirements.txt
```

One GPU. The scorers run on CPU if you have to.

The held-out DreamSim of Table 1 needs its own environment, because `dreamsim` pins versions of
`peft` and `accelerate` that conflict with the rest:

```bash
conda create -y -p ./dsenv python=3.10 && ./dsenv/bin/pip install torch torchvision dreamsim
```

## Reproduce the paper's orange

Five minutes on one GPU, no training — the priors are in `code/baseline/`:

```bash
bash code/reproduce.sh 0            # 0 is the GPU index
```

It samples the interior and prints the paper's own scorers over the carrier and the result:

```
  tex15 by radius, canonical frame:            real     O-Voxel         cyl
  columella                                  0.1263      0.0966      0.0831
  flesh                                      0.0665      0.0604      0.0489
  pith band                                  0.1238      0.1209      0.1148
  flesh colour (value)   photo 0.961      carrier 0.962         ours 0.962
  horizontal banding     photo 0.0079     carrier 0.0123        ours 0.0087
```

and writes `runs/orange_repro/cyl/xfill.png` — three longitudinal cuts and one transverse — and
`state.pt`, the volume. Then the headline number, held-out DreamSim:

```bash
cd code
OBJDIR=../data/orange python dsscore.py ../grids/orange.pt ../runs/orange_repro/cyl/state.pt
OBJDIR=../data/orange ../dsenv/bin/python dsrun.py
```

```
  held-out DreamSim          long    trans     mean   (lower is better)
  real (spl vs hld)        0.0703   0.0404   0.0554      the photographs' own floor
  O-Voxel                  0.1481   0.0770   0.1125      the fitted carrier
  cyl                      0.1119   0.0618   0.0868      this method
```

Both blocks are the output of the commands above on one L40, and they are the numbers in the paper.

## Your own object

You need cross-section photographs and a fitted carrier.

**1. Photographs.** One directory per object; the two families in `vertical/` and `horizontal/`.
Any background: the corner colour is read and replaced. Two or three per family is the setting the
paper measures; one already beats the carrier, five gain nothing over three.

```
photographs/myfruit/vertical/*.png       longitudinal cuts
photographs/myfruit/horizontal/*.png     transverse cuts
```

```bash
PHOTOS=./photographs python code/prep_obj.py myfruit     # -> data/myfruit/{spl,hld}_*, polar strips
```

**2. Carrier.** A two-level occupancy lattice fitted to the same photographs, from
<https://github.com/gino6178/project3> (`code/run.sh`), voxelised here:

```bash
PLY=path/to/myfruit.ply META=path/to/lattice_meta/myfruit N=128 OUT=grids/myfruit.pt \
  python code/voxelize_ov.py
```

For the six objects of the paper, one command does both:

```bash
bash code/fetch_carrier.sh pomegranate         # or: no argument for all six
```

**3. Priors, sampling, scores.** One program, one per-object declaration — the polar axis, in
`code/axis.env`, read from the object's occupancy (the doughnut lies in a vertical plane, so 2):

```bash
bash code/runobj.sh myfruit 0
```

which trains the longitudinal prior (30k steps, ~40 min) and the polar transverse prior (4k, ~5
min), samples the cylinder at `T0=0.3, WFAR=0.1`, and scores. A family with no photographs is
simply inactive.

**4. The asset.** The interior into the lattice cells; the skin keeps its fitted colour and the
cut operator, which is the carrier's, is untouched:

```bash
python code/writeback.py myfruit.ply lattice_meta/myfruit grids/myfruit.pt \
                         runs/myfruit/cyl/state.pt myfruit_filled.ply
```

## The knobs, and what they cost

Every one of these was swept on the orange; the defaults are what the paper reports.

| variable | default | what it does |
|---|---|---|
| `T0H`, `T0V` | 0.3 | starting noise per family. Lower keeps more of the carrier's layout; the curve is flat 0.2–0.3 and turns back toward the carrier below |
| `WFAR` | 0.1 | how much the longitudinal family leads away from the axis. Less is better: it is the weak family |
| `R0` | 0.15 | the radius over which the longitudinal family leads at the axis, where it sees the columella whole |
| `DCFIX` | 16 px | the low-pass of the carrier pinned inside the chain. Above the carrier's 4–8 px tiling, below anything generated |
| `NSTEP` | 100 | DDIM steps |
| longitudinal prior | 30k | keeps improving to 30k |
| polar prior | 4k | memorises its three strips past 4k, and the held-out distance rises with every step after |

## Files

```
code/planes.py          the lattice, and the two slice families
code/prep_obj.py        photographs -> canonical frames, polar strips, the declared split
code/sd3d_train.py      the longitudinal prior;   code/polar_train.py  the transverse one
code/x3dcyl.py          the sampler: both families on one cylindrical state
code/writeback.py       the interior into the carrier's cells
code/voxelize_ov.py     a fitted O-Voxel model -> the grid this code reads
code/voxelize_gs.py     a 3DGS baseline -> the same grid, for a like-for-like comparison
code/fidelity.py colour.py hband.py transverse.py    texture, colour, banding, the other view
code/dsscore.py dsrun.py                             held-out DreamSim
code/runobj.sh          all of it, per object;  code/reproduce.sh  the paper's orange
```

## Honest limits

- Where the carrier is already near the photographs — smooth flesh with discrete seeds, the
  watermelon — the priors smear the seeds along the axis and the lift is a loss. Reported as one.
- Where the carrier fits badly — the bread, the cake — the lift smooths a bad fit and stays far
  from the photographs. A per-object fine-tuned baseline wins the pomegranate and the cake.
- Oblique cuts are supervised by no photograph. They are consistent, and that is a property of the
  two families agreeing, not a measured realism.
- Scores here are canonical-frame slices of the voxel grid, not renders; the ranking is fair
  because every candidate walks the same path, but the absolute numbers are not the carrier
  paper's renderer protocol.
