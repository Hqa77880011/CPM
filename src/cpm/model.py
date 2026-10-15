import torch
from torch import nn
from torch.nn import functional as F


class Phase(nn.Module):
    def __init__(self, dimension, c, linear=False):
        super().__init__()
        self.linear = linear
        self.variance_floor = c["variance_floor"]
        self.encoder = nn.Sequential(
            nn.Conv1d(dimension, c["width"], 5, padding=2),
            nn.GELU(),
            nn.Conv1d(c["width"], c["width"], 3, padding=1),
            nn.GELU(),
        )
        self.phase = nn.Conv1d(c["width"], 1 if linear else 2, 1)
        self.gate = nn.Conv1d(c["width"], 1, 1)
        self.uncertainty = nn.Conv1d(c["width"], 1, 1)

    def forward(self, x):
        encoded = self.encoder(x.T[None])
        raw = self.phase(encoded)[0].T
        if self.linear:
            angle = raw[:, 0].sigmoid() * 2 * torch.pi
            z = torch.stack([angle.cos(), angle.sin()], dim=1)
        else:
            z = F.normalize(raw, dim=1)
        return (
            z,
            self.gate(encoded)[0, 0],
            F.softplus(self.uncertainty(encoded)[0, 0]) + self.variance_floor,
        )


def interpolate(value, indices):
    left = indices.floor().long().clamp(0, len(value) - 2)
    fraction = indices - left
    return value[left] * (1 - fraction[:, None]) + value[left + 1] * fraction[:, None]


def rotate(reference, moving, valid, time_varying=False):
    cross = moving[:, 0] * reference[:, 1] - moving[:, 1] * reference[:, 0]
    dot = (moving * reference).sum(1)
    if time_varying:
        angle = torch.atan2(cross, dot)
    else:
        angle = torch.atan2(cross[valid].sum(), dot[valid].sum())
    cosine = angle.cos()
    sine = angle.sin()
    return torch.stack(
        [
            moving[:, 0] * cosine - moving[:, 1] * sine,
            moving[:, 0] * sine + moving[:, 1] * cosine,
        ],
        dim=1,
    )


def dtw_path(left, right):
    import numpy as np

    cost = ((left[:, None] - right[None]) ** 2).mean(-1)
    distance = np.full((len(left) + 1, len(right) + 1), np.inf)
    distance[0, 0] = 0
    for i in range(len(left)):
        for j in range(len(right)):
            distance[i + 1, j + 1] = cost[i, j] + min(
                distance[i, j], distance[i, j + 1], distance[i + 1, j]
            )
    i, j = len(left), len(right)
    matches = [[] for _ in left]
    while i and j:
        matches[i - 1].append(j - 1)
        step = int(
            np.argmin([distance[i - 1, j - 1], distance[i - 1, j], distance[i, j - 1]])
        )
        if step in (0, 1):
            i -= 1
        if step in (0, 2):
            j -= 1
    return np.array([int(np.mean(m)) if m else 0 for m in matches])
