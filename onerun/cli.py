"""onerun: ingest agent logs, score each run with one JEV read, optionally compare two runs, evaluate."""
import argparse
import glob
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import __version__, jev, judge
from .ingest import PARSERS, detect_format
from .metrics import auroc, bootstrap
from .state import build_state, encoded, normalize_run


def read_jsonl(path, required=False):
    """Rows of a JSONL file. Inputs (required=True) must exist; outputs being resumed may not yet."""
    if not path or not os.path.exists(path):
        if required:
            sys.exit(f"No such file: {path}")
        return []
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    sys.exit(f"{path} line {i} is not valid JSON ({exc.msg}).")
    return rows


def latest_by(rows, key):
    """One row per key (the last one), keeping file order."""
    out = {}
    for r in rows:
        out[key(r)] = r
    return list(out.values())


class Appender:
    def __init__(self, path, truncate=False):
        self.path, self.lock = path, threading.Lock()
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        if truncate:
            open(path, "w").close()

    def write(self, row):
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_from_env(name):
    key = os.environ.get(name, "").strip()
    if not key:
        sys.exit(f"Set {name} in your environment first.")
    return key


def with_retries(fn, attempts=4):
    for i in range(attempts):
        try:
            return fn()
        except (jev.HTTPError, OSError) as exc:  # OSError: connection resets and timeouts
            if not getattr(exc, "retryable", True) or i == attempts - 1:
                raise
            time.sleep(2 ** i * 2)


# ---------------------------------------------------------------------------- ingest
def cmd_ingest(a):
    task = a.task
    if a.task_file:
        with open(a.task_file, encoding="utf-8") as f:
            task = f.read()
    # Expand quoted patterns here, so the same command works on Windows and on huge session folders.
    # A pattern that matches nothing is kept as is, so it is reported as a missing file.
    logs = []
    for x in map(os.path.expanduser, a.logs):
        logs += (sorted(glob.glob(x, recursive=True)) or [x]) if any(c in x for c in "*?[") else [x]
    if a.run_id and len(logs) > 1:
        sys.exit("--run-id only works with a single log file.")
    seen = {r.get("run_id") for r in read_jsonl(a.out)}
    out = Appender(a.out)
    policy = {"model": a.agent_model, "effort": a.effort,
              "web_access": None if a.web_access is None else a.web_access == "yes"}
    for path in logs:
        try:
            fmt = detect_format(path) if a.format == "auto" else a.format
            run = PARSERS[fmt](path, task=task, run_id=a.run_id, task_id=a.task_id, group=a.group,
                               policy=policy)
        except (ValueError, OSError) as exc:  # one bad file never stops the batch
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue
        if run["run_id"] in seen:
            print(f"skip {path}: run {run['run_id']} is already in {a.out} (a resumed Codex session keeps "
                  "its id: ingest only its newest file, or pass --run-id)", file=sys.stderr)
            continue
        seen.add(run["run_id"])
        out.write(run)
        print(f"{path} [{fmt}] -> run {run['run_id']} | task {run['task_id']} | {len(run['steps'])} steps, "
              f"{run['telemetry']['tool_calls']} tool calls, answer {len(run['answer'])} chars")


# ----------------------------------------------------------------------------- score
FINAL_STATUSES = {"ok", "observer_refusal", "task_too_large", "invalid_run"}  # never retried


def cmd_score(a):
    runs = read_jsonl(a.runs, required=True)
    done = {r.get("run_id") for r in read_jsonl(a.out) if r.get("status") in FINAL_STATUSES}
    todo, rows, seen = [], [], set()
    for r in runs:
        rid = r.get("run_id") if isinstance(r, dict) else None
        if rid in done:
            continue
        if not rid or rid in seen:
            print(f"skip a run with {'a duplicate' if rid else 'no'} run_id: {rid!r}", file=sys.stderr)
            continue
        seen.add(rid)
        try:
            r = normalize_run(r)
            state, view, redactions, final_ok = build_state(r)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"invalid run {rid!r}: {exc!r}", file=sys.stderr)
            rows.append({"run_id": rid, "status": "invalid_run", "error": repr(exc)[:200]})
            continue
        base = {"run_id": rid, "task_id": r["task_id"], "group": r.get("group"),
                "view": view.get("view", view.get("reason")), "final_answer_present": final_ok,
                "redactions": redactions}
        if state and state["observable_history"] and state["observable_history"][-1].get("observer_excerpt"):
            base["final_answer_excerpted"] = True  # trimming kept only its beginning and end
        if state is None:
            rows.append({**base, "status": "task_too_large"})
            continue
        todo.append((base, jev.request_body(a.model, state)))
    tokens, usd = jev.estimate_cost([p for _, p in todo])
    print(f"{len(todo)} runs to score ({len(done)} already scored), ~{tokens:,.0f} input tokens, "
          f"~${usd:.5f} at ${jev.USD_PER_M_INPUT_TOKENS}/M.")
    if a.dry_run:
        out = Appender(a.dry_run, truncate=True)
        for base, payload in todo:
            out.write({**base, "payload_bytes": len(encoded(payload)), "payload": payload})
        print(f"Dry run: wrote the exact requests to {a.dry_run}. Nothing was sent.")
        return
    out = Appender(a.out)
    for row in rows:
        out.write(row)
    if not todo:
        return
    if usd > a.max_cost and not a.yes:
        sys.exit(f"Estimated cost is above --max-cost ${a.max_cost}; rerun with --yes to proceed.")
    key = key_from_env("TYPESAFE_API_KEY")
    try:
        models = jev.available_models(key)
    except jev.HTTPError as exc:
        if exc.status in (401, 403):
            sys.exit(f"TypeSafe rejected the key ({exc}).")
        print(f"Could not list models ({exc}); continuing with {a.model}.", file=sys.stderr)
        models = None
    except (OSError, ValueError) as exc:
        sys.exit(f"Could not reach {jev.MODELS_URL}: {exc}")
    if models is not None and a.model not in models:
        sys.exit(f"{a.model} is not available to this key (have: {', '.join(sorted(models))}). "
                 f"The post's numbers used {jev.STUDY_MODEL}; any other version is a different instrument.")
    if a.model != jev.STUDY_MODEL:
        print(f"Note: the post's numbers used {jev.STUDY_MODEL}; {a.model} may differ. Each row records the resolved version.",
              file=sys.stderr)

    def one(base, payload):
        try:
            parsed = jev.parse_response(with_retries(lambda: jev.http_json(jev.ENDPOINT, key, payload)))
        except Exception as exc:  # keep going; failures are recorded and retried on the next run
            return {**base, "status": "error", "error": str(exc)[:200]}
        p = parsed["probabilities"]
        if p is None:
            return {**base, "status": parsed["status"], "model": a.model}
        return {**base, "status": "ok", "risk": round(1 - p["correct"], 6),
                **{"p_" + k: round(v, 6) for k, v in p.items()}, "choice": parsed["choice"],
                "model": a.model, "resolved_model": parsed["resolved_model"]}

    status = Counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(one, b, p) for b, p in todo]):
            row = fut.result()
            status[row["status"]] += 1
            out.write(row)
    print(f"Wrote {a.out}: {dict(status)}. Higher risk = more likely wrong. Rerun to retry errors.")


# ------------------------------------------------------------------------------ pair
def make_pairs(runs):
    by_task = defaultdict(list)
    for r in runs:
        by_task[r["task_id"]].append(r)
    for rs in by_task.values():
        for i in range(0, len(rs) - 1, 2):
            yield rs[i], rs[i + 1]


def cmd_pair(a):
    runs, seen = [], set()
    for r in read_jsonl(a.runs, required=True):
        rid = r.get("run_id") if isinstance(r, dict) else None
        if not rid or rid in seen:
            continue
        seen.add(rid)
        try:
            runs.append(normalize_run(r))
        except ValueError as exc:
            print(f"skip run {rid!r}: {exc}", file=sys.stderr)
    risk = {s["run_id"]: s.get("risk") for s in read_jsonl(a.scores, required=bool(a.scores))
            if s.get("status") == "ok"}
    pairs = list(make_pairs(runs))
    if not pairs:
        sys.exit("No task has two runs. Pairs are formed from runs that share a task_id.")
    done = {(p["run_a"], p["run_b"]) for p in read_jsonl(a.out) if p.get("relation")}
    todo = [(x, y) for x, y in pairs if (x["run_id"], y["run_id"]) not in done]
    print(f"{len(pairs)} pairs ({len(pairs) - len(todo)} already judged); judge {a.judge_model} at {a.effort} effort.")
    if a.dry_run:
        out = Appender(a.dry_run, truncate=True)
        for x, y in todo:
            out.write({"run_a": x["run_id"], "run_b": y["run_id"],
                       "request": judge.request_body(x["task"], x["answer"], y["answer"], a.judge_model, a.effort)})
        print(f"Dry run: wrote the exact requests to {a.dry_run}. Nothing was sent.")
        return
    key = key_from_env("OPENAI_API_KEY") if todo else None

    def one(x, y):
        row = {"task_id": x["task_id"], "group": x.get("group"), "run_a": x["run_id"], "run_b": y["run_id"]}
        try:
            body = judge.request_body(x["task"], x["answer"], y["answer"], a.judge_model, a.effort)
            j = with_retries(lambda: judge.call(body, key))
            return {**row, "relation": j["conclusions"]["relation"],
                    "disagree": judge.disagreement(j, x["answer"], y["answer"]), "judgment": j}
        except Exception as exc:  # recorded with no relation, so the next run retries it
            return {**row, "relation": None, "error": str(exc)[:200]}

    out = Appender(a.out)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(one, x, y) for x, y in todo]):
            out.write(fut.result())
    rows = [p for p in read_jsonl(a.out) if p.get("relation")]
    for p in rows:
        ra, rb = risk.get(p["run_a"]), risk.get(p["run_b"])
        p["risk_a"], p["risk_b"] = ra, rb
        p["jev_mean"] = None if ra is None or rb is None else (ra + rb) / 2
    quiet = sorted((p for p in rows if p.get("disagree") == 0.0 and p["jev_mean"] is not None),
                   key=lambda p: -p["jev_mean"])
    loud = sum(1 for p in rows if p.get("disagree") == 1.0)
    print(f"{loud} of {len(rows)} pairs disagree: likely failures, check them first.")
    if quiet:
        print("Agreeing pairs JEV rates riskiest (possible quiet failures):")
        for p in quiet[:a.top]:
            print(f"  task {p['task_id']}  runs {p['run_a']} / {p['run_b']}  JEV risk {p['jev_mean']:.2f}")
    elif rows and not risk:
        print("Pass --scores to rank the agreeing pairs by JEV risk.")


# -------------------------------------------------------------------------- evaluate
def _fmt(v, ci=None):
    if v is None:
        return "n/a"
    return f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def cmd_evaluate(a):
    labels = {}
    for i, r in enumerate(read_jsonl(a.labels, required=True), 1):
        if not isinstance(r, dict) or "run_id" not in r or "failed" not in r:
            sys.exit(f"{a.labels} line {i}: each label needs \"run_id\" and \"failed\".")
        f = r["failed"]
        labels[r["run_id"]] = bool(f) if not isinstance(f, str) else f.strip().lower() in ("1", "true", "yes", "fail", "failed")
    scores = latest_by([s for s in read_jsonl(a.scores, required=True)
                        if s.get("status") == "ok" and s.get("run_id") in labels], lambda s: s["run_id"])
    report = {}

    def line(name, rows, label, score, unit="task_id"):
        def stat(rs):
            return auroc([label(r) for r in rs], [score(r) for r in rs], [r.get("group") for r in rs])
        v = stat(rows)
        units = [r.get(unit) or r.get("run_id") or i for i, r in enumerate(rows)]
        ci = bootstrap(rows, stat, units, a.bootstrap) if a.bootstrap and v is not None else None
        report[name] = {"auroc": v, "ci95": ci, "n": len(rows), "failing": sum(1 for r in rows if label(r))}
        print(f"  {name:<46} {_fmt(v, ci):<20} n={len(rows)}, failing={report[name]['failing']}")

    print("AUROC (0.5 = coin flip; compared only within each group), 95% interval from resampling tasks:")
    print("Single runs")
    line("JEV one read: did this run fail?", scores, lambda r: labels[r["run_id"]], lambda r: r["risk"])
    pairs = latest_by([p for p in read_jsonl(a.pairs, required=True) if p.get("relation")
                       and p.get("run_a") in labels and p.get("run_b") in labels],
                      lambda p: (p["run_a"], p["run_b"])) if a.pairs else []
    if pairs:
        risk = {s["run_id"]: s["risk"] for s in scores}
        for p in pairs:
            p["y"] = labels[p["run_a"]] or labels[p["run_b"]]
            p["risk_a"], p["risk_b"] = risk.get(p["run_a"]), risk.get(p["run_b"])
            p["jev_mean"] = None if None in (p["risk_a"], p["risk_b"]) else (p["risk_a"] + p["risk_b"]) / 2
            p["combined"] = None if p.get("disagree") is None or p["jev_mean"] is None else p["disagree"] + p["jev_mean"]
        y = lambda p: p["y"]
        agree = [p for p in pairs if p.get("disagree") == 0.0]
        dis = [p for p in pairs if p.get("disagree") == 1.0]
        print("Pairs (label: either run failed)")
        line("Judge: do the two runs disagree?", pairs, y, lambda p: p.get("disagree"))
        line("JEV, one read of run A only", pairs, y, lambda p: p["risk_a"])
        line("JEV, both runs averaged", pairs, y, lambda p: p["jev_mean"])
        line("Judge + JEV", pairs, y, lambda p: p["combined"])
        print("Pairs where the judge says the runs agree (the judge is 0.5 here by construction)")
        line("JEV, one read of run A only (agreeing pairs)", agree, y, lambda p: p["risk_a"])
        line("JEV, both runs averaged (agreeing pairs)", agree, y, lambda p: p["jev_mean"])
        for name, rows in (("disagree", dis), ("agree", agree)):
            if rows:
                k = sum(1 for p in rows if p["y"])
                report[f"P(fail | judge says {name})"] = k / len(rows)
                print(f"  P(either run failed | judge says {name}): {k}/{len(rows)} = {k / len(rows):.0%}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="onerun", description=__doc__)
    ap.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="convert Codex / Claude Code logs into runs.jsonl")
    p.add_argument("logs", nargs="+", help="log files or quoted glob patterns (codex exec --json, Codex rollouts, "
                   "Claude stream-json or transcripts)")
    p.add_argument("-o", "--out", required=True, help="runs JSONL to append to")
    p.add_argument("--format", choices=["auto", *PARSERS], default="auto")
    p.add_argument("--task", help="the task text (required for codex exec and claude -p streams)")
    p.add_argument("--task-file", help="file holding the task text")
    p.add_argument("--task-id", help="default: a hash of the task text, so runs of one task pair up")
    p.add_argument("--group", help="benchmark or task family; evaluation compares runs only within a group")
    p.add_argument("--run-id")
    p.add_argument("--agent-model", help="record the solver model (codex exec streams don't include it)")
    p.add_argument("--effort", help="record the solver effort level")
    p.add_argument("--web-access", choices=["yes", "no"], help="record whether the agent could use the web")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("score", help="one JEV read per run -> risk = 1 - P(correct)")
    p.add_argument("runs")
    p.add_argument("-o", "--out", required=True, help="scores JSONL (appended; finished runs are skipped)")
    p.add_argument("--model", default=jev.DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-cost", type=float, default=1.0, help="stop before spending more than this many USD")
    p.add_argument("--yes", action="store_true", help="proceed even above --max-cost")
    p.add_argument("--dry-run", metavar="PAYLOADS_JSONL", help="write the exact requests locally; send nothing")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("pair", help="optional: judge whether two runs of a task agree")
    p.add_argument("runs")
    p.add_argument("--scores", help="scores JSONL, to rank agreeing pairs by JEV risk")
    p.add_argument("-o", "--out", required=True, help="pairs JSONL (appended)")
    p.add_argument("--judge-model", default=judge.DEFAULT_MODEL)
    p.add_argument("--effort", default=judge.DEFAULT_EFFORT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--dry-run", metavar="REQUESTS_JSONL")
    p.set_defaults(fn=cmd_pair)

    p = sub.add_parser("evaluate", help="check the scores against labels you trust")
    p.add_argument("--scores", required=True)
    p.add_argument("--labels", required=True, help='JSONL of {"run_id": ..., "failed": true/false}')
    p.add_argument("--pairs")
    p.add_argument("--bootstrap", type=int, default=1000, help="resamples for the interval (0 = none)")
    p.add_argument("--json", help="also write the numbers to this file")
    p.set_defaults(fn=cmd_evaluate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
