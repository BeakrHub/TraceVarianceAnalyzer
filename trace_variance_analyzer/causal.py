"""Synthetic causal-inference tasks with a planted answer, for testing a failure detector on runs you can grade.

Each task is a table of 16,384 rows: eight baseline attributes (B01-B08), a 0/1 program and an outcome, plus a
study design stating the identifying assumptions. The agent must estimate the program's effect. In the hard arm,
a three-way interaction of B02, B05 and B08 drives both who gets the program and the outcome: their product
("smooth") or the product of their signs ("thresholded"). The other attributes are distractors. Each even-numbered
hard task has a matched control with the same outcome model but random allocation, so a failure on the hard task
points at the hidden confounding, not at the data.

Generation needs numpy (`pip install ".[generate]"`). Grading uses the standard library only.
"""
import hashlib
import json
import os
import re
from itertools import combinations

N_ROWS = 16384
FLAVOURS = ("smooth", "thresholded")
EFFECTS = (-1.30, -0.85)       # alternate by task, as in the post
TOLERANCE = 0.20               # an estimate further than this from the planted effect is wrong
STUDY_SEED = 2026102302        # regenerates the post's tasks
STUDY_TASKS_PER_FLAVOUR = 16   # seeds are laid out as in the post's batch, so --n 4 gives its first 4 per flavour

STUDY_DESIGN = (
    "The table is an equally weighted sample of the target population. B01 through B08 are continuous attributes "
    "measured before the program; program is 0 or 1 and outcome is measured afterward. The program allocation "
    "process uses the recorded baseline attributes only. Conditional on those attributes, program allocation is "
    "independent of both potential outcomes, and every population region has nonzero probability of both allocation "
    "states. No outcome affects allocation. There is no interference, no missingness, and no unrecorded common cause "
    "of allocation and outcome. Neither the outcome response surface nor the allocation function is specified. The "
    "program effect is constant across units; attribute effects need not be additive.")

_ASK = ("Estimate the population-average causal effect of the program on outcome: the mean outcome if all represented "
        "units received the program minus the mean if none did. Use the observations and study design. Give the effect "
        "in outcome units within 0.20 units, its direction, and justify your choice of analysis.")
STUDY_PROMPT = _ASK + " Use the current working directory for scratch files; /tmp is unavailable."
# The post's runs were read by people and LLMs. A fixed last line lets `grade` read the estimate without guessing.
PROMPT = _ASK + (" Use the current working directory for scratch files. End your answer with a final line of the form "
                 "`Effect: <number>`.")


def _np():
    try:
        import numpy
    except ImportError:
        raise SystemExit('`tva generate` needs numpy: pip install ".[generate]" (or pip install numpy).')
    return numpy


def task_id(flavour, seed, arm):
    return "Q" + hashlib.sha256(f"confirmation-v1|{flavour}|{seed}|{arm}".encode()).hexdigest()[:10]


def plan(n, seed=STUDY_SEED, flavours=FLAVOURS, controls=True):
    """The tasks to make: n hard tasks per flavour, plus a control for each even-numbered one."""
    np = _np()
    stride = max(n, STUDY_TASKS_PER_FLAVOUR)
    seeds = np.random.default_rng(seed).choice(np.arange(10000, 1000000), max(64, stride * len(FLAVOURS)),
                                               replace=False).tolist()
    out = []
    for f in flavours:
        for j in range(n):
            s = seeds[FLAVOURS.index(f) * stride + j]
            for arm in ["hard"] + (["control"] if controls and j % 2 == 0 else []):
                out.append({"task_id": task_id(f, s, arm), "flavour": f, "arm": arm, "seed": s,
                            "effect": EFFECTS[j % 2], "tolerance": TOLERANCE,
                            "group": f"causal-{f}" + ("-control" if arm == "control" else "")})
    return out


def simulate(spec):
    """Attributes X (n x 8), program a and outcome y for one task."""
    np = _np()
    rng = np.random.default_rng(spec["seed"])
    X = rng.uniform(-1, 1, (N_ROWS, 8))
    if spec["flavour"] == "smooth":
        z = X[:, 1] * X[:, 4] * X[:, 7]
        p = 1 / (1 + np.exp(-3 * z))
    else:
        z = np.prod(np.sign(X[:, [1, 4, 7]]), axis=1)
        p = np.where(z > 0, .875, .125)
    eps = rng.normal(0, .6, N_ROWS)
    u = rng.uniform(size=N_ROWS)
    a = (u < (p if spec["arm"] == "hard" else .5)).astype(float)
    y = 4 + 10 * z + 1.1 * X[:, 0] - .7 * X[:, 3] + spec["effect"] * a + eps
    return X, a, y


def _interactions(m):
    """Intercept, every column, and every product of two or three distinct columns."""
    np = _np()
    k = m.shape[1]
    cols = [np.ones(len(m))] + [m[:, i] for i in range(k)]
    cols += [m[:, i] * m[:, j] for i, j in combinations(range(k), 2)]
    cols += [m[:, i] * m[:, j] * m[:, l] for i, j, l in combinations(range(k), 3)]
    return np.column_stack(cols)


def _logistic(Z, a, c=1e4, iters=100):
    """Lightly penalized logistic regression by Newton steps; returns fitted probabilities."""
    np = _np()
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + np.eye(len(w)) / c, Z.T @ (p - a) + w / c)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))


def check(spec, X, a, y):
    """The acceptance rule: two estimators that know the interaction structure recover the planted effect within
    0.10, and (hard arm) adjusting for the attributes additively misses it by more than 0.40."""
    np = _np()
    # Some numpy builds (2.0 on macOS Accelerate) warn spuriously in matmul; a real NaN fails the checks below.
    with np.errstate(all="ignore"):
        src = X if spec["flavour"] == "smooth" else np.sign(X)
        poly = _interactions(src)
        basis = poly if spec["flavour"] == "smooth" else np.c_[poly, X]
        outcome_model = float(np.linalg.lstsq(np.c_[basis, a], y, rcond=None)[0][-1])
        ps = _logistic(poly, a)
        w1, w0 = a / ps, (1 - a) / (1 - ps)
        weighting = float(w1 @ y / w1.sum() - w0 @ y / w0.sum())
        naive = float(np.linalg.lstsq(np.c_[np.ones(len(y)), a, X], y, rcond=None)[0][1])
    b = spec["effect"]
    ok = abs(outcome_model - b) < .1 and abs(weighting - b) < .1 and (spec["arm"] == "control" or abs(naive - b) > .4)
    return {"outcome_model": round(outcome_model, 4), "weighting": round(weighting, 4), "naive": round(naive, 4),
            "accepted": bool(ok)}


def write_task(folder, spec, X, a, y, prompt=PROMPT):
    os.makedirs(folder, exist_ok=False)
    with open(os.path.join(folder, "observations.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(",".join([f"B{i:02}" for i in range(1, 9)] + ["program", "outcome"]) + "\n")
        for xs, ai, yi in zip(X.tolist(), a.tolist(), y.tolist()):
            f.write(",".join(map(repr, xs + [ai, yi])) + "\n")
    with open(os.path.join(folder, "study_design.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(STUDY_DESIGN)
    with open(os.path.join(folder, "task.md"), "w", encoding="utf-8", newline="") as f:
        f.write(prompt + "\n")


# ------------------------------------------------------------------------------ grade
_EFFECT_LINE = re.compile(r"^[\s>*_`#-]*effect[\s*_`]*[:=][\s*_`]*([+\-−]?\s*\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)",
                          re.IGNORECASE | re.MULTILINE)


def read_effect(answer):
    """The number on the answer's last `Effect: <number>` line, or None. No other guessing."""
    found = _EFFECT_LINE.findall(answer or "")
    if not found:
        return None
    return float(found[-1].replace("−", "-").replace(" ", ""))


def grade(estimate, answer_row):
    return abs(estimate - answer_row["effect"]) > answer_row["tolerance"] + 1e-9


def load_answers(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("accepted", True):
                    rows[r["task_id"]] = r
    return rows
