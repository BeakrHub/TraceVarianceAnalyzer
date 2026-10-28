"""Offline tests for the generated test tasks. Generation tests need numpy and are skipped without it."""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest

from onerun import causal, cli

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

TMP = tempfile.TemporaryDirectory()


def tearDownModule():
    TMP.cleanup()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", dir=TMP.name, delete=False, encoding="utf-8")
    f.write("".join(json.dumps(r) + "\n" for r in rows))
    f.close()
    return f.name


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        cli.main(argv)
    return buf.getvalue()


class ReadEffect(unittest.TestCase):
    def test_plain_and_marked_up_lines(self):
        self.assertEqual(causal.read_effect("Some analysis.\nEffect: -1.05"), -1.05)
        self.assertEqual(causal.read_effect("**Effect:** −1.12 units"), -1.12)
        self.assertEqual(causal.read_effect("> effect = +0.3"), 0.3)
        self.assertEqual(causal.read_effect("Effect: 1e-1"), 0.1)

    def test_last_line_wins(self):
        self.assertEqual(causal.read_effect("Effect: -0.9 (first pass)\nRefit.\nEffect: -1.28"), -1.28)

    def test_no_guessing_without_the_line(self):
        self.assertIsNone(causal.read_effect("The estimated effect is -1.05, a decrease."))
        self.assertIsNone(causal.read_effect("Effect: about minus one"))
        self.assertIsNone(causal.read_effect(""))

    def test_tolerance_edges(self):
        key = {"effect": -1.30, "tolerance": 0.20}
        self.assertFalse(causal.grade(-1.10, key))
        self.assertFalse(causal.grade(-1.50, key))
        self.assertTrue(causal.grade(-1.09, key))
        self.assertTrue(causal.grade(1.30, key))


class Grade(unittest.TestCase):
    def test_labels_unread_manual_and_other_tasks(self):
        answers = write_jsonl([
            {"task_id": "Qa", "group": "causal-smooth", "effect": -1.30, "tolerance": 0.2, "accepted": True},
            {"task_id": "Qb", "group": "causal-smooth", "effect": -0.85, "tolerance": 0.2, "accepted": True},
            {"task_id": "Qc", "group": "causal-thresholded", "effect": -1.30, "tolerance": 0.2, "accepted": False}])
        runs = write_jsonl([
            {"run_id": "a1", "task_id": "Qa", "group": "causal-smooth", "task": "t", "answer": "...\nEffect: -1.25"},
            {"run_id": "a2", "task_id": "Qa", "group": "causal-smooth", "task": "t", "answer": "...\nEffect: -1.02"},
            {"run_id": "b1", "task_id": "Qb", "group": "causal-smooth", "task": "t", "answer": "about -0.6"},
            {"run_id": "b2", "task_id": "Qb", "group": "other", "task": "t", "answer": "no line here"},
            {"run_id": "c1", "task_id": "Qc", "task": "t", "answer": "Effect: -1.3"},
            {"run_id": "x1", "task_id": "elsewhere", "task": "t", "answer": "Effect: 0"}])
        manual = write_jsonl([{"run_id": "b2", "estimate": -0.80}])
        out = os.path.join(TMP.name, "labels.jsonl")
        text = run(["grade", runs, "--answers", answers, "-o", out, "--estimates", manual])
        got = {r["run_id"]: r for r in read_jsonl(out)}
        self.assertEqual(set(got), {"a1", "a2", "b2"})
        self.assertFalse(got["a1"]["failed"])
        self.assertTrue(got["a2"]["failed"])
        self.assertEqual((got["b2"]["read"], got["b2"]["failed"]), ("manual", False))
        self.assertIn("1 runs have no `Effect: <number>` line", text)
        self.assertIn("b1", text)
        self.assertIn("1 graded runs were ingested with a group other", text)
        self.assertIn("2 runs are not of generated tasks", text)

    def test_labels_feed_evaluate(self):
        answers = write_jsonl([{"task_id": "Qa", "group": "g", "effect": -1.3, "tolerance": 0.2, "accepted": True}])
        runs = write_jsonl([{"run_id": f"r{i}", "task_id": "Qa", "group": "g", "task": "t",
                             "answer": f"Effect: {e}"} for i, e in enumerate([-1.3, -1.0, -1.25, -0.9])])
        labels = os.path.join(TMP.name, "labels2.jsonl")
        run(["grade", runs, "--answers", answers, "-o", labels])
        scores = write_jsonl([{"run_id": f"r{i}", "status": "ok", "group": "g", "task_id": "Qa", "risk": s}
                              for i, s in enumerate([0.1, 0.8, 0.2, 0.9])])
        text = run(["evaluate", "--scores", scores, "--labels", labels, "--bootstrap", "0"])
        self.assertIn("1.00", text)


@unittest.skipUnless(HAVE_NUMPY, "generation needs numpy")
class Generate(unittest.TestCase):
    def test_reproduces_the_posts_first_task(self):
        out = os.path.join(TMP.name, "gen")
        run(["generate", "-o", out, "--n", "1", "--flavours", "smooth"])
        with open(os.path.join(out, "tasks.tsv"), encoding="utf-8") as f:
            tsv = f.read()
        listed = [line.split("\t") for line in tsv.splitlines()]
        self.assertEqual([(t, g) for t, g, _ in listed], [("Qd605eaaea0", "causal-smooth"),
                                                          ("Qbeedd9da25", "causal-smooth-control")])
        folder = os.path.join(out, "tasks", "Qd605eaaea0")
        self.assertEqual(sorted(os.listdir(folder)), ["observations.csv", "study_design.txt", "task.md"])
        with open(os.path.join(folder, "observations.csv"), "rb") as f:
            csv = f.read()
        self.assertEqual(hashlib.sha256(csv).hexdigest(),
                         "3b5f2a97c980d48dc64f90461bf1415c4d1531f8582a943ce187044aee83b166")  # byte for byte the post's task
        self.assertEqual(csv.count(b"\n"), causal.N_ROWS + 1)
        with open(os.path.join(folder, "task.md"), encoding="utf-8") as f:
            self.assertIn("`Effect: <number>`", f.read())
        answers = read_jsonl(os.path.join(out, "answers.jsonl"))
        self.assertTrue(all(r["accepted"] for r in answers))
        self.assertEqual({r["effect"] for r in answers}, {-1.30})
        self.assertNotIn("effect", tsv)
        with self.assertRaises(SystemExit):
            run(["generate", "-o", out, "--n", "1"])

    def test_study_prompt_and_no_controls(self):
        out = os.path.join(TMP.name, "gen_study")
        run(["generate", "-o", out, "--n", "2", "--flavours", "thresholded", "--no-controls", "--study-prompt"])
        rows = read_jsonl(os.path.join(out, "answers.jsonl"))
        self.assertEqual([(r["arm"], r["effect"]) for r in rows], [("hard", -1.30), ("hard", -0.85)])
        with open(os.path.join(out, "tasks", rows[0]["task_id"], "task.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), causal.STUDY_PROMPT + "\n")

    def test_checks_reject_a_task_the_naive_fit_solves(self):
        spec = dict(causal.plan(1, flavours=("smooth",))[1])  # a control: random allocation
        X, a, y = causal.simulate(spec)
        self.assertTrue(causal.check(spec, X, a, y)["accepted"])
        spec["arm"] = "hard"  # same data judged as a hard task: the naive fit is not fooled, so it is rejected
        self.assertFalse(causal.check(spec, X, a, y)["accepted"])


if __name__ == "__main__":
    unittest.main()
