"""Offline tests: no network, no keys. Run with `python -m unittest` from the repo root."""
import contextlib
import io
import json
import os
import tempfile
import unittest

from onerun import cli, jev, judge
from onerun.ingest import detect_format, parse_claude, parse_codex_exec, parse_codex_rollout
from onerun.metrics import auroc
from onerun.state import BOUNDED_STATE_BYTES, build_state, encoded, normalize_run, sanitize

EX = os.path.join(os.path.dirname(__file__), "..", "examples")
with open(os.path.join(EX, "task.md"), encoding="utf-8") as _f:
    TASK = _f.read()
TMP = tempfile.TemporaryDirectory()


def tearDownModule():
    TMP.cleanup()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(rows, encoding="utf-8"):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", dir=TMP.name, delete=False, encoding=encoding)
    f.write("".join(json.dumps(r) + "\n" for r in rows))
    f.close()
    return f.name


def quiet(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cli.main(argv)


class State(unittest.TestCase):
    def test_minimal_run_gets_answer_as_last_step(self):
        run = normalize_run({"task": "2+2?", "answer": "4"})
        self.assertEqual(run["steps"], [{"role": "assistant", "content": "4"}])
        state, view, _, final_ok = build_state({"task": "2+2?", "answer": "4"})
        self.assertEqual(view, {"view": "full"})
        self.assertTrue(final_ok)
        self.assertEqual(set(state), {"agent_policy", "task", "observable_history"})

    def test_answer_not_duplicated(self):
        run = normalize_run({"task": "t", "answer": "a", "steps": [{"role": "assistant", "content": "a"}]})
        self.assertEqual(len(run["steps"]), 1)

    def test_sanitize_drops_answer_keys_and_redacts(self):
        out = sanitize({"answer": "x", "note": "mail me at someone@example.com"})
        self.assertNotIn("answer", out)
        self.assertIn("[REDACTED_EMAIL]", out["note"])

    def test_long_history_is_trimmed_and_keeps_final_answer(self):
        steps = [{"role": "tool", "name": "shell.execute", "content": "x" * 9000} for _ in range(20)]
        steps.append({"role": "assistant", "content": "FINAL ANSWER 42"})
        state, view, _, _ = build_state({"task": "t", "steps": steps})
        self.assertEqual(view["view"], "recent-history-with-explicit-excerpts/v1")
        self.assertLessEqual(len(encoded(state)), BOUNDED_STATE_BYTES)
        self.assertEqual(state["observable_history"][-1]["content"], "FINAL ANSWER 42")

    def test_final_answer_reported_missing_when_trimmed_away(self):
        steps = [{"role": "tool", "name": "shell.execute", "content": "x" * 8000}]
        _, _, _, final_ok = build_state({"task": "t" * 37000, "steps": steps, "answer": "y" * 3900})
        self.assertFalse(final_ok)


class Jev(unittest.TestCase):
    def test_request_shape(self):
        body = jev.request_body("jev-1.13.0", {"task": "t"})
        q = body["questions"]["completed_grade"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(list(q["criteria"]), ["correct", "partial", "incorrect", "refusal_error"])

    def test_parse_accepts_cent_rounding(self):
        raw = {"model": "jev-1.13.0", "answers": {"completed_grade": {"type": "choice", "choice": "correct",
               "probabilities": {"correct": 0.7, "partial": 0.2, "incorrect": 0.08, "refusal_error": 0.01}}}}
        p = jev.parse_response(raw)["probabilities"]
        self.assertAlmostEqual(sum(p.values()), 1.0)

    def test_parse_rejects_bad_mass(self):
        raw = {"answers": {"completed_grade": {"type": "choice", "choice": "correct",
               "probabilities": {"correct": 0.5, "partial": 0.2, "incorrect": 0.1, "refusal_error": 0.1}}}}
        with self.assertRaises(ValueError):
            jev.parse_response(raw)


class Judge(unittest.TestCase):
    def j(self, relation, qa, qb):
        dim = {"relation": relation, "quote_a": qa, "quote_b": qb, "reason": "r"}
        return {"conclusions": dim, "assumptions": dict(dim, relation="unclear"),
                "explanations": dict(dim, relation="unclear")}

    def test_disagreement_needs_real_quotes(self):
        self.assertEqual(judge.disagreement(self.j("disagree", "is 4", "is 5"), "it is 4", "it is 5"), 1.0)
        self.assertEqual(judge.disagreement(self.j("agree", "is 4", "is 4"), "it is 4", "it is 4"), 0.0)
        self.assertIsNone(judge.disagreement(self.j("disagree", "made up", "is 5"), "it is 4", "it is 5"))
        self.assertIsNone(judge.disagreement(self.j("unclear", "", ""), "a", "b"))


class Metrics(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(auroc([1, 0], [0.9, 0.1]), 1.0)
        self.assertEqual(auroc([1, 0], [0.5, 0.5]), 0.5)
        self.assertIsNone(auroc([1, 1], [0.2, 0.3]))

    def test_groups_are_not_pooled(self):
        # Group A is harder (higher scores) but within each group the score is uninformative.
        y = [1, 0, 1, 0]
        s = [0.9, 0.9, 0.1, 0.1]
        self.assertEqual(auroc(y, s, ["a", "a", "b", "b"]), 0.5)


class Ingest(unittest.TestCase):
    def test_codex_exec_stream(self):
        path = os.path.join(EX, "codex_exec_stream.jsonl")
        self.assertEqual(detect_format(path), "codex-exec")
        run = parse_codex_exec(path, task=TASK)
        self.assertEqual(run["run_id"], "example-codex-thread")
        self.assertIn("$28.05", run["answer"])
        self.assertEqual(run["telemetry"]["tool_calls"], 1)
        self.assertNotIn("Hidden reasoning", json.dumps(run))

    def test_codex_exec_needs_task(self):
        with self.assertRaises(ValueError):
            parse_codex_exec(os.path.join(EX, "codex_exec_stream.jsonl"))

    def test_claude_stream(self):
        path = os.path.join(EX, "claude_stream.jsonl")
        self.assertEqual(detect_format(path), "claude")
        run = parse_claude(path, task=TASK)
        self.assertEqual(run["agent_policy"]["model"], "claude-opus-5-5")
        self.assertEqual([s["role"] for s in run["steps"]], ["assistant", "tool", "assistant"])
        self.assertEqual(run["steps"][0]["tool_call"]["name"], "shell.execute")
        self.assertNotIn("Hidden reasoning", json.dumps(run))

    def test_claude_transcript(self):
        path = write_jsonl([
            {"type": "user", "parentUuid": None, "sessionId": "s1", "message": {"role": "user", "content": TASK}},
            {"type": "assistant", "parentUuid": "u1", "sessionId": "s1",
             "message": {"model": "m", "content": [{"type": "text", "text": "$28.05"}]}},
            {"type": "assistant", "parentUuid": "u2", "sessionId": "s1", "isSidechain": True,
             "message": {"content": [{"type": "text", "text": "subagent chatter"}]}}])
        self.assertEqual(detect_format(path), "claude")
        run = parse_claude(path)
        self.assertEqual(run["task"], TASK)
        self.assertEqual(run["answer"], "$28.05")
        self.assertEqual(len(run["steps"]), 1)

    def test_codex_rollout(self):
        path = write_jsonl([
            {"type": "session_meta", "payload": {"id": "sess-1"}},
            {"type": "turn_context", "payload": {"model": "gpt-5.5", "effort": "low"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]}},
            {"type": "response_item", "payload": {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": TASK}]}},
            {"type": "response_item", "payload": {"type": "reasoning", "summary": []}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "call_id": "c1",
             "arguments": json.dumps({"command": ["python3", "-c", "print(28.05)"]})}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1",
             "output": json.dumps({"output": "28.05\n", "metadata": {"exit_code": 0}})}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "$28.05"}]}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "$28.05", "duration_ms": 900}}])
        self.assertEqual(detect_format(path), "codex-rollout")
        run = parse_codex_rollout(path)
        self.assertEqual(run["task"], TASK)
        self.assertEqual(run["agent_policy"]["model"], "gpt-5.5")
        self.assertEqual([s["role"] for s in run["steps"]], ["assistant", "tool", "assistant"])
        self.assertEqual(run["steps"][1]["content"], "28.05\n")

    def test_codex_rollout_newer_format(self):
        typed = {"type": "UserMessage", "id": "u", "content": [  # injected context arrives glued to the prompt
            {"type": "text", "text": "<in-app-browser-context>page</in-app-browser-context>\n" + TASK}]}
        path = write_jsonl([
            {"type": "session_meta", "payload": {"id": "own", "source": "vscode"}},
            {"type": "session_meta", "payload": {"id": "parent"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "<recommended_plugins>x</recommended_plugins>"}]}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": typed}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "$28.05"}]}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "$28.05"}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": dict(typed, content=[
                {"type": "text", "text": "and in euros?"}])}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "about 26 euros"}]}},
            {"type": "event_msg", "payload": {"type": "turn_aborted"}}])
        run = parse_codex_rollout(path)
        self.assertEqual(run["run_id"], "own")
        self.assertEqual(run["task"], TASK)
        self.assertEqual(run["answer"], "about 26 euros")  # not the earlier turn's answer
        self.assertEqual([s["role"] for s in run["steps"]], ["assistant", "user", "assistant"])

    def test_codex_subagent_session_is_skipped(self):
        path = write_jsonl([{"type": "session_meta", "payload": {"id": "g", "source": {"subagent": {"other": "x"}}}}])
        with self.assertRaises(ValueError):
            parse_codex_rollout(path, task="t")

    def test_bom_and_utf16_logs(self):
        rows = [{"type": "thread.started", "thread_id": "th"},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "$28.05"}}]
        for enc in ("utf-8-sig", "utf-16"):
            run = parse_codex_exec(write_jsonl(rows, encoding=enc), task=TASK)
            self.assertEqual((run["run_id"], run["answer"]), ("th", "$28.05"))

    def test_lone_surrogate_is_scrubbed(self):
        path = os.path.join(TMP.name, "surrogate.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"type": "item.completed", "item": {"type": "agent_message", "text": "cut \\ud83d"}}\n')
        encoded(parse_codex_exec(path, task="t"))  # must not raise UnicodeEncodeError


class Cli(unittest.TestCase):
    def test_ingest_keeps_logged_model_and_skips_bad_files(self):
        d = tempfile.mkdtemp(dir=TMP.name)
        empty = os.path.join(d, "empty.jsonl")
        open(empty, "w").close()
        runs = os.path.join(d, "runs.jsonl")
        quiet(["ingest", empty, os.path.join(EX, "claude_stream.jsonl"), "--task-file",
               os.path.join(EX, "task.md"), "--effort", "high", "-o", runs])
        policy = read_jsonl(runs)[0]["agent_policy"]
        self.assertEqual((policy["model"], policy["effort"]), ("claude-opus-5-5", "high"))

    def test_score_resume_and_bad_runs(self):
        d = tempfile.mkdtemp(dir=TMP.name)
        runs = write_jsonl([{"run_id": "a", "task": "2+2?", "answer": "4"},
                            {"run_id": "a", "task": "2+2?", "answer": "4"},
                            {"run_id": "b", "task": ""},
                            {"run_id": "c", "task": "t", "steps": [{"role": "assistant", "tool_call": {"arguments": {}}}]}])
        req, out = os.path.join(d, "req.jsonl"), os.path.join(d, "scores.jsonl")
        for _ in range(2):
            quiet(["score", runs, "-o", out, "--dry-run", req])
        self.assertEqual([r["run_id"] for r in read_jsonl(req)], ["a"])  # deduped, not appended twice
        with open(out, "w") as f:
            f.write(json.dumps({"run_id": "a", "status": "observer_refusal"}) + "\n")
        quiet(["score", runs, "-o", out, "--dry-run", req])
        self.assertEqual(read_jsonl(req), [])  # a refusal is final, not resent

    def test_pair_minimal_runs_without_task_id(self):
        d = tempfile.mkdtemp(dir=TMP.name)
        runs = write_jsonl([{"run_id": "a", "task": "2+2?", "answer": "4"},
                            {"run_id": "b", "task": "2+2?", "answer": "5"}])
        req = os.path.join(d, "req.jsonl")
        quiet(["pair", runs, "-o", os.path.join(d, "pairs.jsonl"), "--dry-run", req])
        self.assertEqual(len(read_jsonl(req)), 1)

    def test_missing_input_file_is_an_error(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli.main(["score", os.path.join(TMP.name, "nope.jsonl"), "-o", os.path.join(TMP.name, "s.jsonl")])

    def test_ingest_dry_run_and_evaluate(self):
        d = tempfile.mkdtemp(dir=TMP.name)
        runs = os.path.join(d, "runs.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["ingest", os.path.join(EX, "codex_exec_stream.jsonl"), os.path.join(EX, "claude_stream.jsonl"),
                      "--task-file", os.path.join(EX, "task.md"), "-o", runs])
            cli.main(["score", runs, "-o", os.path.join(d, "scores.jsonl"), "--dry-run", os.path.join(d, "req.jsonl")])
            buf = io.StringIO()
        rows = read_jsonl(runs)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["task_id"], rows[1]["task_id"])  # same task -> they pair up
        reqs = read_jsonl(os.path.join(d, "req.jsonl"))
        self.assertEqual(len(reqs), 2)
        self.assertNotIn("answer", reqs[0]["payload"]["state"])
        with contextlib.redirect_stdout(buf):
            cli.main(["evaluate", "--scores", os.path.join(EX, "demo_scores.jsonl"),
                      "--labels", os.path.join(EX, "demo_labels.jsonl"),
                      "--pairs", os.path.join(EX, "demo_pairs.jsonl"), "--bootstrap", "50"])
        self.assertIn("agreeing pairs", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
