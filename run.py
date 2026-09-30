#!/usr/bin/env python3
"""One-command local runner for the Candor take-home.

Ingestion is forced to UTF-8 (the supplied harness reads the data with the platform
default encoding, which raises UnicodeDecodeError on e.g. cp1252 Windows), and every
step runs as a child process so a failure in one is visible rather than silent.
"""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT / "evals" / "memory_holdout.jsonl"


def child_env():
    env = dict(os.environ)
    # Python-level UTF-8 mode: applies to every child, including the harness scorers.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT), env=child_env())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--memory-questions', default='evals/memory_train.jsonl')
    ap.add_argument('--memory-out', default='memory_answers.jsonl')
    ap.add_argument('--action-commands', default='evals/actions_train.jsonl')
    ap.add_argument('--action-out', default='actions_train.jsonl')
    ap.add_argument('--no-chrono', action='store_true',
                    help='ablation: lexical-only memory, graph disabled')
    ap.add_argument('--with-holdout', action='store_true',
                    help='also score the paraphrase holdout set')
    ap.add_argument('--skip-eval', action='store_true')
    a = ap.parse_args()

    chrono_flags = ['--no-chrono'] if a.no_chrono else []

    run([sys.executable, 'run_memory.py', '--data', a.data,
         '--questions', a.memory_questions, '--out', a.memory_out] + chrono_flags)
    run([sys.executable, 'run_actions.py', '--data', a.data,
         '--commands', a.action_commands, '--out', a.action_out])
    if a.skip_eval:
        return

    print('\n=== Retrieval ===')
    run([sys.executable, 'eval_harness/score_retrieval.py',
         '--gold', a.memory_questions, '--answers', a.memory_out, '--data', a.data])
    print('\n=== Memory answers ===')
    run([sys.executable, 'eval_harness/score_memory.py',
         '--gold', a.memory_questions, '--answers', a.memory_out,
         '--data', a.data, '--judge', 'none'])
    print('\n=== Actions ===')
    run([sys.executable, 'eval_harness/score_actions.py',
         '--gold', a.action_commands, '--predictions', a.action_out])

    if a.with_holdout:
        if not HOLDOUT.exists():
            print(f'\n(holdout set not found at {HOLDOUT})')
            return
        run([sys.executable, 'run_memory.py', '--data', a.data,
             '--questions', str(HOLDOUT.relative_to(ROOT)),
             '--out', 'memory_holdout_answers.jsonl'] + chrono_flags)
        print('\n=== Retrieval (holdout) ===')
        run([sys.executable, 'eval_harness/score_retrieval.py', '--gold',
             str(HOLDOUT.relative_to(ROOT)), '--answers', 'memory_holdout_answers.jsonl',
             '--data', a.data, '--out', 'results_holdout_retrieval.json'])
        print('\n=== Memory answers (holdout) ===')
        run([sys.executable, 'eval_harness/score_memory.py', '--gold',
             str(HOLDOUT.relative_to(ROOT)), '--answers', 'memory_holdout_answers.jsonl',
             '--data', a.data, '--judge', 'none', '--out', 'results_holdout_memory.json'])


if __name__ == '__main__':
    main()

