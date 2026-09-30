#!/usr/bin/env python3
"""Generate the paraphrase holdout set from the train facts.

The train set is for development, and the hidden test asks different questions about the
same data. This script keeps every scored field from the train item (as_of, needed,
key_terms, evidence, stale, never_say) and replaces only the wording of the question,
so the holdout measures paraphrase robustness rather than luck.

  python3 tools/make_holdout.py            # writes evals/memory_holdout.jsonl

This is deliberately not a template machine: each paraphrase is written by hand against
the same fact, and the script refuses to emit a question identical to the train wording.

Known limit: a paraphrase holdout shares the underlying facts with train, so it cannot
tell you how the system does on a *new* storyline. It does tell you whether the answer
layer is keying on wording or on meaning.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "evals" / "memory_train.jsonl"
OUT = ROOT / "evals" / "memory_holdout.jsonl"

PARAPHRASES = {
    "MEM-TR-01": "What's the current launch date for Route Planner v2?",
    "MEM-TR-02": "When does Route Planner v2 go live?",
    "MEM-TR-03": "What launch date did we agree for Route Planner v2?",
    "MEM-TR-04": "Have I delivered the pricing proposal I promised to Sarah Patel?",
    "MEM-TR-05": "What rates did we put in front of Acme Freight?",
    "MEM-TR-06": "Is the Harbor demo environment still something I owe Marcus?",
    "MEM-TR-07": "Who is handling the onboarding mockups, and have they finished?",
    "MEM-TR-08": "Was John on board with dropping dark mode?",
    "MEM-TR-09": "Is a second designer role going ahead?",
    "MEM-TR-10": "Will Harbor Logistics sign before the year is out?",
    "MEM-TR-11": "What is our p95 latency for routing?",
    "MEM-TR-12": "How many days passed between the Acme call and sending the pricing proposal?",
    "MEM-TR-13": "What time is board deck prep?",
    "MEM-TR-14": "Which database did I choose for the ETA prototype, and why?",
    "MEM-TR-15": "What's my preference for Friday standups?",
    "MEM-TR-16": "Did Harbor Logistics say anything about SOC 2?",
    "MEM-TR-17": "How much does Dana get paid?",
    "MEM-TR-18": "What's Sarah Kim's take on SSO?",
    "MEM-TR-19": "What time does my Denver flight depart?",
    "MEM-TR-20": "What did I dictate to Sarah Patel on September 10, and was it sent?",
    "MEM-TR-21": "What caused the launch to move off September 30?",
    "MEM-TR-22": "Which customer caused the move to October 21, and what did they ask for?",
    "MEM-TR-23": "When am I due to follow up with Sarah Patel?",
    "MEM-TR-24": "Who owns the regression test plan, and was it delivered?",
    "MEM-TR-25": "What meetings do I have on the day of my Denver flight?",
    "MEM-TR-26": "Is the Acme contract signed yet?",
    "MEM-TR-27": "How many regression cases passed on September 16?",
}


def main():
    train = [json.loads(line) for line in open(TRAIN) if line.strip()]
    missing = [item["id"] for item in train if item["id"] not in PARAPHRASES]
    if missing:
        raise SystemExit(f"no paraphrase for {missing}")

    rows = []
    for index, item in enumerate(train, start=1):
        question = PARAPHRASES[item["id"]]
        if question.strip().lower() == item["question"].strip().lower():
            raise SystemExit(f"{item['id']} paraphrase is identical to the train question")
        row = dict(item)
        row["id"] = f"MEM-HO-{index:02d}"
        row["question"] = question
        row["derived_from"] = item["id"]
        rows.append(row)

    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} holdout questions to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
