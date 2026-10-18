"""Finite cyclic kinematics exported through a fixed unsupervised feature map."""

import argparse
import json
from pathlib import Path

import numpy as np


def sequence(rng, projection, error=False, frames=180):
    speed = np.exp(0.2 * np.sin(np.linspace(0, 3 * np.pi, frames)) + rng.normal(0, 0.03, frames))
    phase = np.cumsum(speed) / speed.sum() * 6 * np.pi
    cycle = np.floor(phase / (2 * np.pi)).astype(int)
    cycle[:12] = -1
    cycle[-12:] = -1
    joints = np.column_stack([np.cos(phase), np.sin(phase),
                              0.6 * np.cos(2 * phase + 0.3), 0.6 * np.sin(2 * phase + 0.3)])
    abnormal = np.zeros(frames, bool)
    if error:
        start = int(rng.integers(frames // 3, frames // 2))
        abnormal[start:start + 24] = True
        # A missing limb excursion changes the actual motion, not an appended flag.
        joints[abnormal, 2:] *= 0.1
        joints[abnormal, 0] += 0.3 * np.sin(phase[abnormal] * 3)
    idle = cycle < 0
    joints[idle] = joints[np.flatnonzero(~idle)[0]] + rng.normal(0, 0.03, (idle.sum(), 4))
    features = joints @ projection + rng.normal(0, 0.035, (frames, projection.shape[1]))
    features += rng.normal(0, 0.025, (1, projection.shape[1]))
    return dict(features=features.astype(np.float32), cycle=cycle, error=abnormal,
                phase=phase.astype(np.float32), reference_length=np.float32(frames / 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.groups < 6:
        parser.error("Use at least six independent actor/session groups")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    projection = np.random.default_rng(args.seed).normal(size=(4, 16)) / 2
    records = []
    for group in range(args.groups):
        split = "train" if group < 0.6 * args.groups else "validation" if group < 0.8 * args.groups else "test"
        for repeat in range(2):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, group, repeat]))
            abnormal = split == "test" and repeat == 1
            name = f"sequence_{group:03d}_{repeat}.npz"
            np.savez_compressed(output / name, **sequence(rng, projection, abnormal))
            records.append(dict(id=f"{group}:{repeat}", path=name, actor=f"actor-{group}",
                                sequence=f"sequence-{group}-{repeat}", session=f"session-{group}",
                                split=split, normal=not abnormal, onset_anchored=False))
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(
        source="cyclic four-coordinate kinematics with fixed random feature projection",
        truth="phase/cycle from motion clock; errors from omitted limb excursion",
        seed=args.seed, groups=args.groups, features=16,
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
