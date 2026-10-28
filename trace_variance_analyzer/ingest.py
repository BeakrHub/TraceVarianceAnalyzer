"""Turn Codex and Claude Code logs into runs: task, visible steps, final answer, telemetry.

Hidden reasoning (Codex reasoning items, Claude thinking blocks) is always dropped.
Subagent internals are dropped too; the call that started them and their returned
result stay, because that is what the main agent saw.
"""
import hashlib
import json
import re

from .state import task_id_for

HARNESS = "external-agent CLI; observable messages and tool calls; hidden reasoning excluded"

CLAUDE_TOOLS = {"Bash": "shell.execute", "Read": "file.read", "Write": "file.write", "Edit": "file.edit",
                "MultiEdit": "file.edit", "NotebookEdit": "file.edit", "Grep": "file.search",
                "Glob": "file.search", "LS": "file.list", "WebFetch": "web.fetch", "WebSearch": "web.search",
                "Task": "agent.delegate", "Agent": "agent.delegate", "TodoWrite": "plan.update"}
CODEX_TOOLS = {"shell": "shell.execute", "exec_command": "shell.execute", "local_shell": "shell.execute",
               "container.exec": "shell.execute", "exec": "shell.execute", "apply_patch": "file.edit", "write_stdin": "shell.input",
               "update_plan": "plan.update", "view_image": "file.read", "web_search": "web.search"}
# Text that Codex or Claude Code injects as a "user" message; it is context, not the task.
INJECTED_PREFIXES = ("<environment_context>", "<user_instructions>", "<permissions instructions>",
                     "<INSTRUCTIONS>", "# AGENTS.md", "<command-", "<local-command-", "<system-reminder>",
                     "<bash-", "Caveat:")
# Tagged context blocks that newer Codex versions prepend to what the user typed. Add to this
# list if your harness injects others; untagged pastes and unknown tags are kept as user text.
INJECTED_BLOCK = re.compile(r"\s*<(environment_context|user_instructions|system-reminder|in-app-browser-context|"
                            r"recommended_plugins|heartbeat|guardian_tool_descriptions|turn_aborted|"
                            r"external_codex_apps[\w-]*)\b[^>]*>.*?</\1>", re.S)


def _strip_injected(text):
    """What the user typed, without injected context; '' if nothing else is left."""
    m = INJECTED_BLOCK.match(text)
    while m:
        text = text[m.end():].lstrip()
        m = INJECTED_BLOCK.match(text)
    return "" if text.lstrip().startswith(INJECTED_PREFIXES) else text


def _encoding(path):
    with open(path, "rb") as f:
        head = f.read(2)
    return "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"  # PowerShell 5 '>' writes UTF-16


def _scrub(value):
    """Replace lone UTF-16 surrogates (text cut mid-emoji), which cannot be written as UTF-8."""
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _lines(path):
    with open(path, encoding=_encoding(path), errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(e, dict):
                    yield _scrub(e) if "\\ud" in line.lower() else e


def detect_format(path):
    for e in _lines(path):
        t = e.get("type")
        if isinstance(t, str) and (t.startswith("thread.") or t.startswith("item.") or t.startswith("turn.")):
            return "codex-exec"
        if t in ("session_meta", "response_item", "event_msg", "turn_context") and "payload" in e:
            return "codex-rollout"
        if t in ("user", "assistant", "system", "result") and ("parentUuid" in e or "session_id" in e
                                                               or "message" in e or "subtype" in e):
            return "claude"
    raise ValueError(f"{path}: not a recognised Codex or Claude Code log")


def _text(blocks, user=False):
    """Join a message's text parts; for user messages, drop injected context."""
    if isinstance(blocks, str):
        return _strip_injected(blocks) if user else blocks
    out = []
    for b in blocks or []:
        if isinstance(b, dict):
            if b.get("type") in ("text", "input_text", "output_text"):
                text = _strip_injected(b.get("text", "")) if user else b.get("text", "")
                if text.strip() or not user:
                    out.append(text)
            elif b.get("type") == "image":
                out.append("[image]")
    return "\n".join(out)


def _finish(run, task, defaults):
    """defaults: run_id / task_id / group to set, and policy={...} to merge into agent_policy."""
    if task is None:
        raise ValueError("could not find the task in this log; pass --task or --task-file")
    answer = next((s["content"] for s in reversed(run["steps"])
                   if s["role"] == "assistant" and not s.get("tool_call") and s.get("content")), None)
    run["answer"] = run.pop("_answer", None) or answer or ""
    run["task"] = task
    run["telemetry"]["tool_calls"] = sum(1 for s in run["steps"] if s.get("tool_call"))
    # Values given on the command line win; missing ones keep what the log recorded.
    run["agent_policy"].update({k: v for k, v in (defaults.pop("policy", None) or {}).items() if v is not None})
    run["agent_policy"] = {k: v for k, v in run["agent_policy"].items() if v is not None}
    for k, v in defaults.items():
        if v is not None:
            run[k] = v
    if not run.get("task_id"):
        run["task_id"] = task_id_for(task)
    if not run.get("run_id"):
        run["run_id"] = "r_" + hashlib.sha256(json.dumps(run["steps"], sort_keys=True).encode()).hexdigest()[:12]
    return run


def _new_run(provider):
    return {"agent_policy": {"provider": provider, "harness": HARNESS}, "steps": [], "telemetry": {}}


def parse_codex_exec(path, task=None, **defaults):
    """Output of `codex exec --json`. The stream has no prompt, so the task must be supplied."""
    run = _new_run("codex")
    tel = run["telemetry"]
    for e in _lines(path):
        t = e.get("type")
        if t == "thread.started":
            run["run_id"] = e.get("thread_id")
        elif t == "turn.completed":
            for k, v in (e.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    tel[k] = tel.get(k, 0) + v
        elif t in ("turn.failed", "error"):
            tel["errors"] = tel.get("errors", 0) + 1
        elif t == "item.completed":
            it = e.get("item") or {}
            kind = it.get("type")
            if kind == "agent_message":
                run["steps"].append({"role": "assistant", "content": it.get("text", "")})
            elif kind == "command_execution":
                run["steps"].append({"role": "assistant", "tool_call": {
                    "name": "shell.execute", "arguments": {"command": it.get("command", "")}}})
                out = it.get("aggregated_output", "")
                if it.get("exit_code") not in (None, 0):
                    out += f"\n[exit code {it['exit_code']}]"
                run["steps"].append({"role": "tool", "name": "shell.execute", "content": out})
            elif kind == "file_change":
                run["steps"].append({"role": "assistant", "tool_call": {
                    "name": "file.edit", "arguments": {"changes": it.get("changes", [])}}})
                run["steps"].append({"role": "tool", "name": "file.edit", "content": str(it.get("status", ""))})
            elif kind == "mcp_tool_call":
                name = f"mcp.{it.get('server', '')}.{it.get('tool', '')}"
                args = it.get("arguments")
                run["steps"].append({"role": "assistant", "tool_call": {
                    "name": name, "arguments": args if isinstance(args, dict) else {"input": args}}})
                result = (it.get("error") or {}).get("message") or _text((it.get("result") or {}).get("content"))
                run["steps"].append({"role": "tool", "name": name, "content": result or ""})
            elif kind == "web_search":
                run["steps"].append({"role": "assistant", "tool_call": {
                    "name": "web.search", "arguments": {"query": it.get("query", "")}}})
            elif kind == "error":
                run["steps"].append({"role": "tool", "name": "error", "content": it.get("message", "")})
            # reasoning (hidden) and todo_list (plan display) are not part of the visible work
    return _finish(run, task, defaults)


def parse_codex_rollout(path, task=None, **defaults):
    """A saved Codex session, ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl (best effort: the format
    changes between Codex versions)."""
    run = _new_run("codex")
    tel, names, prompts, user_msgs, typed = run["telemetry"], {}, [], [], False
    for e in _lines(path):
        t, p = e.get("type"), e.get("payload") or {}
        if t == "session_meta":
            if isinstance(p.get("source"), dict) and "subagent" in p["source"]:
                raise ValueError("Codex subagent session; its parent session's log already has its result")
            run.setdefault("run_id", p.get("id"))  # a spawned session also embeds its parent's meta
        elif t == "turn_context":
            run["agent_policy"]["model"] = p.get("model") or run["agent_policy"].get("model")
            run["agent_policy"]["effort"] = p.get("effort") or run["agent_policy"].get("effort")
        elif t == "event_msg":
            kind = p.get("type")
            item = p.get("item") or {}
            if kind == "user_message" and isinstance(p.get("message"), str):
                prompts.append(p["message"])  # older versions: what the user typed
                run.pop("_answer", None)  # a new turn: an earlier turn's answer is no longer final
            elif kind == "item_completed" and item.get("type") == "UserMessage":
                text = _text(item.get("content"), user=True)  # newer versions
                if text.strip():
                    if not typed:  # typed prompts replace the raw user messages collected so far
                        typed = True
                        run["steps"] = [s for s in run["steps"] if s["role"] != "user"]
                    prompts.append(text)
                    run["steps"].append({"role": "user", "content": text})
                    run.pop("_answer", None)
            elif kind == "task_complete":
                if p.get("last_agent_message"):
                    run["_answer"] = p["last_agent_message"]
                if isinstance(p.get("duration_ms"), (int, float)):
                    tel["duration_ms"] = tel.get("duration_ms", 0) + p["duration_ms"]
            elif kind == "token_count" and isinstance((p.get("info") or {}).get("total_token_usage"), dict):
                tel.update({k: v for k, v in p["info"]["total_token_usage"].items() if isinstance(v, (int, float))})
        elif t == "response_item":
            kind = p.get("type")
            if kind == "message" and p.get("role") == "user":
                text = _text(p.get("content"), user=True)
                if text.strip() and not typed:
                    user_msgs.append(text)
                    run["steps"].append({"role": "user", "content": text})
                    run.pop("_answer", None)
            elif kind == "message" and p.get("role") == "assistant":
                run["steps"].append({"role": "assistant", "content": _text(p.get("content"))})
            elif kind in ("function_call", "custom_tool_call"):
                name = CODEX_TOOLS.get(p.get("name"), p.get("name") or "tool")
                names[p.get("call_id")] = name
                if kind == "function_call":
                    try:
                        args = json.loads(p.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {"raw": p.get("arguments")}
                else:
                    args = {"input": p.get("input", "")}
                run["steps"].append({"role": "assistant", "tool_call": {
                    "name": name, "arguments": args if isinstance(args, dict) else {"input": args}}})
            elif kind in ("function_call_output", "custom_tool_call_output"):
                out = p.get("output", "")
                if isinstance(out, str):
                    try:
                        parsed = json.loads(out)
                        out = parsed.get("output", out) if isinstance(parsed, dict) else out
                    except json.JSONDecodeError:
                        pass
                elif isinstance(out, dict):
                    out = out.get("output") or out.get("content") or json.dumps(out)
                run["steps"].append({"role": "tool", "name": names.get(p.get("call_id"), "tool"),
                                     "content": out if isinstance(out, str) else _text(out)})
            elif kind == "web_search_call":
                run["steps"].append({"role": "assistant", "tool_call": {
                    "name": "web.search", "arguments": {"query": (p.get("action") or {}).get("query", "")}}})
            # reasoning items are hidden reasoning and are dropped
    if task is None:
        task = prompts[0] if prompts else (user_msgs[0] if user_msgs else None)
    if task is not None:  # the task is stated once, as the task, not again as a history step
        i = next((i for i, s in enumerate(run["steps"]) if s["role"] == "user"
                  and s.get("content", "").strip() == task.strip()), None)
        if i is not None:
            del run["steps"][i]
    return _finish(run, task, defaults)


def parse_claude(path, task=None, **defaults):
    """`claude -p ... --output-format stream-json --verbose` output, or a saved session transcript
    (~/.claude/projects/<project>/<session-id>.jsonl)."""
    run = _new_run("claude_code")
    tel, names, prompts = run["telemetry"], {}, []
    first_ts = last_ts = None
    for e in _lines(path):
        t = e.get("type")
        if e.get("isSidechain") or e.get("parent_tool_use_id"):
            continue  # subagent internals
        if e.get("timestamp"):
            first_ts, last_ts = first_ts or e["timestamp"], e["timestamp"]
        if t == "system" and e.get("subtype") == "init":
            run["run_id"] = e.get("session_id")
            run["agent_policy"]["model"] = e.get("model")
            if isinstance(e.get("tools"), list):
                run["agent_policy"]["available_tools"] = [CLAUDE_TOOLS.get(x, x) for x in e["tools"]][:40]
        elif t == "result":
            if isinstance(e.get("result"), str) and e["result"].strip():
                run["_answer"] = e["result"]
            for k in ("duration_ms", "num_turns", "total_cost_usd", "is_error"):
                if k in e:
                    tel[k] = e[k]
            usage = e.get("usage") or {}
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                if isinstance(usage.get(k), (int, float)):
                    tel[k] = usage[k]
        elif t in ("user", "assistant"):
            run.setdefault("run_id", e.get("sessionId"))
            msg = e.get("message") or {}
            content = msg.get("content")
            if t == "assistant":
                if msg.get("model") and not run["agent_policy"].get("model"):
                    run["agent_policy"]["model"] = msg["model"]
                if e.get("effort") and not run["agent_policy"].get("effort"):
                    run["agent_policy"]["effort"] = e["effort"]
                for b in content if isinstance(content, list) else [{"type": "text", "text": content or ""}]:
                    if b.get("type") == "text" and b.get("text", "").strip():
                        run["steps"].append({"role": "assistant", "content": b["text"]})
                    elif b.get("type") == "tool_use":
                        name = CLAUDE_TOOLS.get(b.get("name"), b.get("name") or "tool")
                        names[b.get("id")] = name
                        args = b.get("input")
                        run["steps"].append({"role": "assistant", "tool_call": {
                            "name": name, "arguments": args if isinstance(args, dict) else {"input": args}}})
                    # thinking blocks are hidden reasoning and are dropped
            else:
                if e.get("isMeta") or e.get("isCompactSummary"):
                    continue
                if isinstance(content, list) and any(b.get("type") == "tool_result" for b in content):
                    for b in content:
                        if b.get("type") == "tool_result":
                            out = _text(b.get("content"))
                            if b.get("is_error"):
                                out = "[tool error]\n" + out
                            run["steps"].append({"role": "tool", "name": names.get(b.get("tool_use_id"), "tool"),
                                                 "content": out})
                else:
                    if _text(content).lstrip().startswith("<task-notification>"):  # a background agent's report
                        run["steps"].append({"role": "tool", "name": "agent.delegate", "content": _text(content)})
                        continue
                    text = _text(content, user=True)
                    if text.strip():
                        prompts.append(text)
                        if len(prompts) > 1:  # follow-up prompts are part of the visible history
                            run["steps"].append({"role": "user", "content": text})
    if first_ts and last_ts and "duration_ms" not in tel:
        tel["wall_clock"] = {"first": first_ts, "last": last_ts}
    if task is None and prompts:
        task = prompts[0]
    return _finish(run, task, defaults)


PARSERS = {"codex-exec": parse_codex_exec, "codex-rollout": parse_codex_rollout, "claude": parse_claude}
