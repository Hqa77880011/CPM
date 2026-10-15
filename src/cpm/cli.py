import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import Phase, interpolate, rotate, dtw_path
from .common import load, save, table, plot, auc


def prepare(a, c):
    manifest = load(a.input)
    owners = {}
    records = []
    for r in manifest:
        for key in ["actor", "sequence", "session"]:
            group = (key, str(r[key]))
            if group in owners and owners[group] != r["split"]:
                raise ValueError(key + " crosses splits")
            owners[group] = r["split"]
        path = Path(r["path"])
        path = path if path.is_absolute() else Path(a.input).parent / path
        data = np.load(path)
        if data["features"].ndim != 2 or len(data["features"]) < 3:
            raise ValueError(
                "Expected features [frame,dimension] with at least three frames"
            )
        records.append(dict(r, path=str(path.resolve())))
    save(Path(a.data) / "manifest.json", records)


def train(a, c):
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    records = [r for r in load(Path(a.data) / "manifest.json") if r["split"] == "train"]
    dimension = np.load(records[0]["path"])["features"].shape[1]
    model = Phase(dimension, c, a.variant == "linear").to(c["device"])
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    rows = []
    for epoch in range(c["epochs"]):
        total = 0.0
        rng.shuffle(records)
        for r in records:
            data = np.load(r["path"])
            x = torch.tensor(data["features"], dtype=torch.float32, device=c["device"])
            cycles = data["cycle"]
            valid = torch.tensor(cycles >= 0, device=c["device"])
            z, gate, sigma = model(x)
            loss = c["gate_weight"] * nn.functional.binary_cross_entropy_with_logits(
                gate, valid.float()
            )
            if r["normal"] and valid.any():
                increments = torch.tensor(
                    rng.lognormal(0, c["warp_strength"], len(x) - 1),
                    device=c["device"],
                    dtype=torch.float32,
                )
                positions = torch.cat([increments.new_zeros(1), increments.cumsum(0)])
                positions = positions / positions[-1] * (len(x) - 1)
                warped = interpolate(x, positions)
                zw, _, _ = model(warped)
                reference = nn.functional.normalize(interpolate(z, positions), dim=1)
                accepted = valid[positions.round().long()] & valid
                if accepted.any():
                    aligned = (
                        zw
                        if r.get("onset_anchored", False)
                        else rotate(
                            reference, zw, accepted, a.variant == "time_varying"
                        )
                    )
                    distance = 1 - (aligned * reference).sum(1)
                    if a.variant == "linear":
                        distance = (
                            torch.atan2(zw[:, 1], zw[:, 0])
                            - torch.atan2(reference[:, 1], reference[:, 0])
                        ).square()
                    if a.variant != "no_equivariance":
                        loss = (
                            loss
                            + (
                                distance[accepted] / sigma[accepted].square()
                                + sigma[accepted].log()
                            ).mean()
                        )
                for cycle in np.unique(cycles[cycles >= 0]):
                    ids = np.flatnonzero(cycles == cycle)
                    if len(ids) < 3:
                        continue
                    zz = z[ids]
                    angular = torch.atan2(
                        zz[:-1, 0] * zz[1:, 1] - zz[:-1, 1] * zz[1:, 0],
                        (zz[:-1] * zz[1:]).sum(1),
                    )
                    loss = (
                        loss
                        + c["orientation_weight"]
                        * torch.relu(0.1 * 2 * torch.pi / len(ids) - angular).mean()
                    )
                    loss = loss + c["winding_weight"] * (
                        (angular.sum() - 2 * torch.pi) ** 2 + zz.mean(0).square().sum()
                    )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        rows.append(dict(epoch=epoch + 1, loss=total / len(records)))
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            model=model.cpu().state_dict(),
            dimension=dimension,
            config=c,
            variant=a.variant,
        ),
        output / "model.pt",
    )
    table(output / "history.csv", rows)


def phase_origin(z, features, memory, anchored):
    phase = np.arctan2(z[:, 1], z[:, 0]) % (2 * np.pi)
    if anchored:
        return (phase - phase[0]) % (2 * np.pi)
    best = None
    for offset in np.linspace(0, 2 * np.pi, len(memory), endpoint=False):
        bins = ((phase + offset) / (2 * np.pi) * len(memory)).astype(int) % len(memory)
        error = float(np.mean((features - memory[bins]) ** 2))
        if best is None or error < best[0]:
            best = (error, offset)
    return (phase + best[1]) % (2 * np.pi)


def memory(a, c):
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = state["config"]
    model = Phase(state["dimension"], c, state["variant"] == "linear")
    model.load_state_dict(state["model"])
    model.eval()
    sums = np.zeros((c["phase_bins"], state["dimension"]))
    squares = sums.copy()
    counts = np.zeros(c["phase_bins"])
    reference = None
    with torch.no_grad():
        for r in load(Path(a.data) / "manifest.json"):
            if r["split"] != "train" or not r["normal"]:
                continue
            data = np.load(r["path"])
            x = data["features"]
            z = model(torch.tensor(x, dtype=torch.float32))[0].numpy()
            angle = np.arctan2(z[:, 1], z[:, 0])
            angle = (angle - angle[0]) % (2 * np.pi)
            bins = (angle / (2 * np.pi) * c["phase_bins"]).astype(int) % c["phase_bins"]
            valid = data["cycle"] >= 0
            if reference is None:
                reference = x[valid]
            for k in range(c["phase_bins"]):
                selected = valid & (bins == k)
                sums[k] += x[selected].sum(0)
                squares[k] += (x[selected] ** 2).sum(0)
                counts[k] += selected.sum()
    if not np.all(counts):
        raise ValueError(
            "Successful training cycles do not cover every phase bin; train longer or reduce phase_bins"
        )
    mean = sums / counts[:, None]
    std = np.sqrt(np.maximum(squares / counts[:, None] - mean**2, c["variance_floor"]))
    Path(a.output).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(a.output) / "memory.npz", mean=mean, std=std, reference=reference
    )


def evaluate(a, c):
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = state["config"]
    model = Phase(state["dimension"], c, state["variant"] == "linear")
    model.load_state_dict(state["model"])
    model.eval()
    stored = np.load(a.memory)
    mem = stored["mean"]
    std = stored["std"]
    rng = np.random.default_rng(c["seed"])
    if a.control == "shuffled_memory":
        mem = mem[rng.permutation(len(mem))]
    rows = []
    scores = []
    labels = []
    delays = []
    phase_errors = []
    wrap_errors = []
    align_errors = []
    with torch.no_grad():
        for r in load(Path(a.data) / "manifest.json"):
            if r["split"] != "test":
                continue
            data = np.load(r["path"])
            x = data["features"]
            z, gate, sigma = model(torch.tensor(x, dtype=torch.float32))
            z = z.numpy()
            predicted_gate = gate.sigmoid().numpy() >= c["gate_threshold"]
            valid = data["cycle"] >= 0
            active = valid if a.control == "boundary_oracle" else predicted_gate
            phase = phase_origin(z, x, mem, r.get("onset_anchored", False))
            bins = (phase / (2 * np.pi) * len(mem)).astype(int) % len(mem)
            residual = np.sqrt(np.mean((x - mem[bins]) ** 2, axis=1)) / (
                np.sqrt(np.mean(std[bins] ** 2, axis=1)) + sigma.numpy() + 1e-8
            )
            if a.control == "no_phase":
                residual = np.sqrt(np.mean((x - mem.mean(0)) ** 2, axis=1)) / (
                    std.mean() + 1e-8
                )
            if a.control == "dtw":
                correspondence = dtw_path(x, stored["reference"])
                residual = np.sqrt(
                    np.mean((x - stored["reference"][correspondence]) ** 2, axis=1)
                ) / (std.mean() + 1e-8)
            error = data["error"].astype(bool)
            scores.extend(residual[active].tolist())
            labels.extend(error[active].tolist())
            if "phase" in data:
                difference = np.angle(np.exp(1j * (phase - data["phase"])))
                if not r.get("onset_anchored", False):
                    difference = np.angle(
                        np.exp(
                            1j
                            * (
                                difference
                                - np.angle(np.mean(np.exp(1j * difference[valid])))
                            )
                        )
                    )
                phase_errors.extend(np.abs(difference[valid]).tolist())
                true_phase = data["phase"] % (2 * np.pi)
                near_wrap = valid & (
                    np.minimum(true_phase, 2 * np.pi - true_phase) <= np.pi / 8
                )
                wrap_errors.extend((np.abs(difference[near_wrap]) > np.pi / 4).tolist())
                if "reference_length" in data:
                    align_errors.extend(
                        (
                            np.abs(difference[valid])
                            / (2 * np.pi)
                            * float(data["reference_length"])
                        ).tolist()
                    )
            starts = np.flatnonzero(error & ~np.r_[False, error[:-1]])
            for start in starts:
                stop = next(
                    (i for i in range(start + 1, len(error)) if not error[i]),
                    len(error),
                )
                detected = np.flatnonzero(
                    active[start:stop] & (residual[start:stop] >= c["error_threshold"])
                )
                delays.append(int(detected[0]) if len(detected) else stop - start)
            rows.append(
                dict(
                    id=r["id"],
                    method=state["variant"] + ":" + a.control,
                    deferral=1 - active.mean(),
                    gate_error=np.mean(predicted_gate != valid),
                    frames=len(x),
                )
            )
    summary = dict(
        method=state["variant"] + ":" + a.control,
        phase_mae=np.mean(phase_errors) if phase_errors else None,
        wrap_error=np.mean(wrap_errors) if wrap_errors else None,
        alignment_frames=np.mean(align_errors) if align_errors else None,
        error_auroc=auc(labels, scores),
        delay_frames=np.mean(delays) if delays else None,
        deferral=np.mean([r["deferral"] for r in rows]),
        gate_error=np.mean([r["gate_error"] for r in rows]),
    )
    table(Path(a.output) / "sequences.csv", rows)
    save(Path(a.output) / "metrics.json", summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train", "memory", "evaluate", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/video_features")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--checkpoint")
    p.add_argument("--memory", default="checkpoints/memory.npz")
    p.add_argument("--inputs", nargs="*")
    p.add_argument(
        "--variant",
        choices=["cpm", "linear", "no_equivariance", "time_varying"],
        default="cpm",
    )
    p.add_argument(
        "--control",
        choices=["none", "shuffled_memory", "boundary_oracle", "no_phase", "dtw"],
        default="none",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train":
        train(a, c)
    elif a.command == "memory":
        memory(a, c)
    elif a.command == "evaluate":
        evaluate(a, c)
    elif a.command == "compare":
        rows = [load(Path(d) / "metrics.json") for d in a.inputs]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
