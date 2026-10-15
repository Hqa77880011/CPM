# CPM

PyTorch implementation of circular phase equivariance for tempo-robust repetitive-action error localization.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Grouped frame features

Use [RepCount](https://svip-lab.github.io/dataset/RepCount_dataset.html), [Assembly101](https://assembly101.github.io/) or [IndustReal](https://timschoonbeek.github.io/IndustReal/). Export frozen visual features to NPZ `features` [frame,dimension], integer `cycle` [frame] (cycle ID, -1 outside cyclic intervals), boolean evaluation `error` [frame], optional reference `phase` in radians and scalar `reference_length` in frames/cycle. A JSON manifest has `id,path,actor,sequence,session,split,normal,onset_anchored`. Keep actors, whole sequences, objects and sessions together when making the manifest. The importer enforces actor/sequence/session separation; object-disjoint assignments remain part of dataset construction.

```bash
cpm prepare --input data/manifest.json --data data/video_features
cpm train --data data/video_features --output runs/full
cpm memory --data data/video_features --checkpoint runs/full/model.pt --output checkpoints
cpm evaluate --data data/video_features --checkpoint runs/full/model.pt --memory checkpoints/memory.npz --output runs/full
cpm compare --inputs runs/full --output outputs/comparison
cpm plot --output outputs/comparison
```

## Circular learning and memory

A compact temporal head predicts an S1 unit vector, interval gate and positive phase uncertainty from the supplied frozen features. Monotone paired warps share one sequence-level circular Procrustes rotation, or zero rotation when onset anchored. Only normal training cycles learn phase; training interval annotations learn the shared gate. Local forward orientation, one winding per annotated cycle and phase coverage prevent collapsed solutions; these explicit regularizers fill under-specified optimization details. Error labels are read only at evaluation.

Memory pools successful training features by predicted phase. Onset normalization gives a common training origin. At test time a single rotation is chosen for the entire sequence using memory fit, or the supplied onset rule. Same-phase feature distance is normalized by memory spread and phase uncertainty. The predicted gate defers outside cyclic intervals. The memory builder fails if a phase bin has no successful support rather than inventing a reference.

Train separate `linear`, `no_equivariance` and `time_varying` variants. Evaluation controls are shuffled phase memory, boundary-oracle gate, no-phase reference and an actual dynamic-time-warping feature-path baseline. `time_varying` deliberately permits the prohibited per-frame rotation during training as a mechanism ablation. External TimeCycle/TCC/RepNet/TransRAC/PREGO/EgoPER training systems are separate architectures, not aliases of these controls.

## Metrics and outputs

`configs/default.json` controls temporal width, warp strength, phase bins, loss weights, optimizer, gate/error thresholds and device. Train writes the final checkpoint and loss history; memory writes means, spreads and one successful reference sequence. Evaluation reports circular phase MAE after one allowed global alignment, phase-equivalent alignment error in frames (when reference_length is supplied), error AUROC, detected-error delay, deferral and gate error. Undetected error intervals are assigned their full interval duration as capped delay. The optional phase annotations are evaluation-only.

`sequences.csv` records gate behavior; `metrics.json` contains aggregate measurements. Tempo/pause/skip/missing-cycle stress uses separately exported feature sequences and the same checkpoint/memory. The phase-equivalent frame metric measures circular correspondence rather than a full repeated-cycle-index match. Comparison CSV/JSON and PNG/SVG plots use computed values.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
cpm-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.

Wrap error is the fraction of valid frames within pi/8 of the annotated phase wrap whose globally aligned circular error exceeds pi/4. Missing phase annotations yield null phase/wrap metrics.
