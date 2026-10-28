"""What JEV sees: the task plus the run's visible history, sanitized and trimmed if needed.

This is the instrument from the study, unchanged: the same projection, redaction,
size limits and trimming view. Changing any of it makes a different instrument.
"""
import copy
import hashlib
import json
import re
from collections import Counter

# Earlier JEV max_tokens_exceeded errors began at ~48.4 KB request bodies.
FULL_MAX_STATE_BYTES = 44000
BOUNDED_STATE_BYTES = 40000
EVENT_EXCERPT_BYTES = 4000

# Best-effort redaction, not a guarantee that free text is safe to send.
PATTERNS = {
    "credential": re.compile(r"(?i)(?:Bearer\s+\S+|\b(?:sk|ts)-[A-Za-z0-9_-]{12,}|"
                             r"(?:api[_-]?key|password|access[_-]?token|secret)\s*[=:]\s*[^\s,;}]+)"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "uuid": re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"),
    "local_path": re.compile(r"(?:/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\Users\\)[^\s\"'`<>]*"),
}
# Keys that could leak a grade or an answer key are dropped wherever they appear.
FORBIDDEN_KEYS = {"gold", "reference", "fixture_private", "scores", "correct_answer",
                  "final_answer", "answer", "system_prompt", "total_steps", "remaining_steps",
                  "progress", "progress_fraction", "run_id", "task_id"}
ALLOWED_POLICY = {"provider", "model", "effort", "harness", "available_tools", "web_access"}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sanitize(value, counts=None):
    counts = counts if counts is not None else Counter()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k.lower() in FORBIDDEN_KEYS:
                counts["removed_key:" + k.lower()] += 1
            else:
                out[k] = sanitize(v, counts)
        return out
    if isinstance(value, list):
        return [sanitize(v, counts) for v in value]
    if isinstance(value, str):
        for name, pattern in PATTERNS.items():
            value, n = pattern.subn("[REDACTED_" + name.upper() + "]", value)
            counts[name] += n
        return value
    return value


def project_step(step):
    role = step.get("role")
    if role not in ("assistant", "user", "tool"):
        return None
    out = {"role": role}
    if isinstance(step.get("content"), str):
        out["content"] = step["content"]
    if role == "assistant" and step.get("tool_call"):
        call = step["tool_call"]
        out["tool_call"] = {"name": call["name"], "arguments": call.get("arguments", {})}
    if role == "tool" and step.get("name"):
        out["name"] = step["name"]
    return out


def bounded_state(state, max_bytes=BOUNDED_STATE_BYTES, event_bytes=EVENT_EXCERPT_BYTES):
    """Keep the task and policy intact plus the most recent history that fits; flag every omission.

    Large events (including a final answer over ~4 KB) keep a marked beginning and end.
    The final answer is the last event, so it is the last to be dropped, but with a task
    near the budget even it may not fit: build_state reports that as final_answer_present
    = False. Returns (None, reason) if the task alone is too large.
    """
    value = copy.deepcopy(state)
    history = value.pop("observable_history", [])
    metadata = {"view": "recent-history-with-explicit-excerpts/v1",
                "omitted_earlier_events": 0, "excerpted_events": 0,
                "notice": "Observer-only context restriction. The solver had the original history. "
                          "Omitted content must not be interpreted as evidence of solver failure."}
    value["observable_history"] = []
    value["observer_context"] = metadata
    if len(encoded(value)) > max_bytes:
        return None, {"reason": "task_and_policy_exceed_recovery_budget"}
    processed = []
    for event in history:
        e = copy.deepcopy(event)
        if len(encoded(e)) > event_bytes:
            text = encoded(e).decode()
            keep = event_bytes // 4
            while True:  # character slices can expand in UTF-8: shrink until the byte bound holds
                clipped = {"role": e["role"], "content": text[:keep] +
                           "\n[PAST EVENT CONTENT OMITTED FOR OBSERVER CONTEXT LIMIT]\n" + text[-keep:],
                           "observer_excerpt": True}
                if len(encoded(clipped)) <= event_bytes or keep <= 1:
                    break
                keep //= 2
            e = clipped
        processed.append(e)
    selected = []
    for e in reversed(processed):
        trial = [e] + selected
        value["observable_history"] = trial
        metadata["omitted_earlier_events"] = len(history) - len(trial)
        metadata["excerpted_events"] = sum(bool(x.get("observer_excerpt")) for x in trial)
        if len(encoded(value)) > max_bytes:
            break
        selected = trial
    value["observable_history"] = selected
    metadata["omitted_earlier_events"] = len(history) - len(selected)
    metadata["excerpted_events"] = sum(bool(x.get("observer_excerpt")) for x in selected)
    return value, dict(metadata)


def task_id_for(task):
    """Default task id: a hash of the task text, so runs of the same task pair up."""
    return "t_" + hashlib.sha256(task.strip().encode("utf-8")).hexdigest()[:12]


def normalize_run(run):
    """Accept the full run format or the minimal {run_id, task, answer} one.

    Returns a copy with steps ending in the final answer, an answer, and a task_id.
    """
    if not isinstance(run, dict):
        raise ValueError("each run must be a JSON object")
    if not isinstance(run.get("task"), str) or not run["task"].strip():
        raise ValueError(f"run {run.get('run_id')!r}: 'task' must be a non-empty string")
    run = dict(run)
    steps = [s for s in (run.get("steps") or []) if isinstance(s, dict)]
    answer = run.get("answer")
    if isinstance(answer, str) and answer.strip():
        last = next((s for s in reversed(steps) if s.get("role") == "assistant"
                     and not s.get("tool_call")), None)
        if last is None or steps[-1] is not last or (last.get("content") or "").strip() != answer.strip():
            steps.append({"role": "assistant", "content": answer})
    else:
        last = next((s for s in reversed(steps) if s.get("role") == "assistant" and not s.get("tool_call")
                     and isinstance(s.get("content"), str) and s["content"].strip()), None)
        run["answer"] = last["content"] if last else ""
    run["steps"] = steps
    if not run.get("task_id"):
        run["task_id"] = task_id_for(run["task"])
    return run


def build_state(run):
    """Return (state or None, view, redaction counts, final_answer_present) for one run.

    final_answer_present is checked on what JEV will actually see, after any trimming.
    """
    run = normalize_run(run)
    steps = [x for x in (project_step(s) for s in run["steps"]) if x is not None]
    policy = {k: v for k, v in (run.get("agent_policy") or {}).items() if k in ALLOWED_POLICY}
    counts = Counter()
    state = sanitize({"agent_policy": policy, "task": run["task"], "observable_history": steps}, counts)
    view = {"view": "full"}
    if len(encoded(state)) > FULL_MAX_STATE_BYTES:
        state, view = bounded_state(state)
    seen = state["observable_history"] if state else []
    last = next((s for s in reversed(seen) if s["role"] == "assistant"), None)
    final_ok = last is not None and not last.get("tool_call") and bool((last.get("content") or "").strip())
    return state, view, {k: v for k, v in counts.items() if v}, final_ok
