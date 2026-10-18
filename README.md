# CPM

Implementation of circular phase equivariance for tempo-robust repetitive-action error localization.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation and preparation

The following input route generates repetitive-motion feature sequences with recorded phase/error labels and disjoint groups. These are generated features, not extracted camera recordings. `source.json` records assumptions and seed.

```bash
python -m cpm.synthetic --output data/generated --groups 30 --seed 17
cpm prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
```

For external data: Use [RepCount](https://svip-lab.github.io/dataset/RepCount_dataset.html), [Assembly101](https://assembly101.github.io/) or [IndustReal](https://timschoonbeek.github.io/IndustReal/). Export frozen visual features to NPZ `features` [frame,dimension], integer `cycle` [frame] (cycle ID, -1 outside cyclic intervals), boolean evaluation `error` [frame], optional reference `phase` in radians and scalar `reference_length` in frames/cycle. A JSON manifest has `id,path,actor,sequence,session,split,normal,onset_anchored`. Keep actors, whole sequences, objects and sessions together when making the manifest. The importer enforces actor/sequence/session separation; object-disjoint assignments remain part of dataset construction.

## Training and inference

`configs/reproduce.json` fixes seed `17`. Use the same config and prepared inputs throughout; change `seed` there for another run and keep its output directories separate.

```bash
cpm train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
cpm memory --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output checkpoints/reproduce
```

## Evaluation

```bash
cpm evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --memory checkpoints/reproduce/memory.npz --output runs/reproduce
```

## Comparisons and ablations

Fit each supported `--variant` into a separate directory, evaluate its own checkpoint, then pass the evaluated run directories to `compare --inputs`.

```bash
cpm compare --inputs runs/reproduce --output outputs/comparison
cpm plot --output outputs/comparison
```

## Outputs

The selected commands write their generated inputs under `data/generated/`, prepared data under `data/reproduce/`, and checkpoints/results in the explicit output directories. Corresponding steps produce `model.pt`, `history.csv`, `memory.npz`, `sequences.csv`, `metrics.json`, `source.json`. Comparisons write CSV/JSON and plots under their output directory.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
