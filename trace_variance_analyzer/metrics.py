"""AUROC compared only within groups, with an optional task-level bootstrap."""
import random
from collections import defaultdict


def auroc(labels, scores, groups=None):
    """P(a failing item outranks a passing one), counting only pairs in the same group.

    Pooling groups (benchmarks, task families) lets a detector score well just by knowing
    which group is harder, so every number here compares items within a group.
    Ties count half. Items with no score are skipped. Returns None when no group has
    both a failing and a passing item. Rank-based, so it is fast on large files.
    """
    groups = groups if groups is not None else [None] * len(labels)
    by = defaultdict(list)
    for y, s, g in zip(labels, scores, groups):
        if s is not None:
            by[g].append((s, bool(y)))
    num = den = 0.0
    for items in by.values():
        n_pos = sum(1 for _, y in items if y)
        n_neg = len(items) - n_pos
        if not n_pos or not n_neg:
            continue
        items.sort(key=lambda t: t[0])
        rank_sum, i = 0.0, 0
        while i < len(items):
            j = i
            while j + 1 < len(items) and items[j + 1][0] == items[i][0]:
                j += 1
            rank_sum += ((i + j) / 2 + 1) * sum(1 for k in range(i, j + 1) if items[k][1])
            i = j + 1
        num += rank_sum - n_pos * (n_pos + 1) / 2
        den += n_pos * n_neg
    return num / den if den else None


def bootstrap(rows, stat, units, draws=1000, seed=0):
    """95% interval of stat(rows) when resampling units (e.g. tasks) with replacement."""
    by_unit = defaultdict(list)
    for r, u in zip(rows, units):
        by_unit[u].append(r)
    keys = sorted(by_unit, key=str)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in by_unit[k]]
        v = stat(sample)
        if v is not None:
            values.append(v)
    if len(values) < draws * 0.5:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]
