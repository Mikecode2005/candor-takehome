#!/usr/bin/env python3
"""Answer a free-form, numbered question list (for example ``question.txt``).

Unlike ``run_memory.py`` (which expects the benchmark JSONL schema), this runner
accepts the plain list a reviewer would type: one numbered question per line.
Every item is routed to the subsystem that can actually answer it and the system
prints what it says, together with the evidence ids it relied on.

    python3 run_ask.py --questions question.txt

Routing is by intent, not by the exact train wording:

* an instruction with an action verb (send/message/email/remind/move/book/
  open/delete/announce) ...................................... the action layer
* an item that asks the system to try something unsupported ... an abstention probe
* everything else ............................................ the temporal memory
  (ranked retrieval + grounded answer)

A question may pin its own point in time: any ``YYYY-MM-DD`` in the text becomes
the ``as_of`` instant (end of that day), otherwise the latest record time in the
data is used as ``now``.
"""
import argparse, json, re
from datetime import datetime
from pathlib import Path

try:
    from src.memory import Memory
    from src.actions import action, classify
except ImportError:  # executed directly, e.g. ``python src/ask.py``
    from memory import Memory
    from actions import action, classify


# Intent vocabulary the action layer handles (kept in sync with src/actions.py).
ACTION_INTENTS = {
    'delete_email', 'open_app', 'channel_announce', 'send_email',
    'send_message', 'create_reminder', 'move_event', 'create_event',
}
# A meta-item that asks the system to demonstrate abstention rather than to answer
# a fact ("ask something unsupported and see whether it says I don't know").
ABSTAIN_CUES = re.compile(r"\bunsupported\b|\babstain\b|don'?t know|do not know", re.I)
# The deliberately unsupported probe used for the "I don't know" demonstration.
# ETA Predictor is a real internal project, but nothing in the data prices it.
PROBE_QUESTION = 'What is the pricing for the ETA Predictor?'
ISO_DATE = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')
NUMBERED = re.compile(r'^\s*\d+\s*[.)]\s*(.*)$')


def parse_items(text):
    """Split a numbered list into items; a non-numbered line continues the previous one."""
    items = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = NUMBERED.match(line)
        if m:
            items.append(m.group(1).strip())
        elif items:
            items[-1] = f"{items[-1]} {stripped}".strip()
    return items


def kind_of(item):
    """memory | action | probe, decided from the item's intent."""
    if ABSTAIN_CUES.search(item):
        return 'probe'
    if classify(item) in ACTION_INTENTS:
        return 'action'
    return 'memory'


def resolve_asof(item, default):
    """Honour an explicit ISO date so point-in-time questions time-travel to that day."""
    m = ISO_DATE.search(item)
    if not m:
        return default
    year, month, day = (int(g) for g in m.groups())
    try:
        return datetime(year, month, day, 23, 59, 59, tzinfo=default.tzinfo)
    except ValueError:
        return default


def answer_items(items, data_dir):
    """Run every item through the system and return one result row per item."""
    memory = Memory(data_dir)
    data_path = Path(data_dir)
    now = max(u['time'] for u in memory.units)
    rows = []
    for number, item in enumerate(items, 1):
        kind = kind_of(item)
        as_of = resolve_asof(item, now)
        row = {'number': number, 'question': item, 'kind': kind, 'as_of': as_of.isoformat()}
        if kind == 'action':
            row['actions'] = action(item, as_of, data_path)
        else:
            question = PROBE_QUESTION if kind == 'probe' else item
            retrieved = memory.retrieve(question, as_of, 20)
            ans, sources, abstained = memory.answer(question, as_of, retrieved)
            row.update({'answer': ans, 'sources': sources, 'abstained': abstained})
            if kind == 'probe':
                row['probe'] = PROBE_QUESTION
            else:
                row['retrieved'] = [u['id'] for _, u in retrieved[:20]]
        rows.append(row)
    return rows


def render(rows):
    """Human-readable transcript of what the system answered."""
    out = []
    for row in rows:
        out.append('=' * 78)
        out.append(f"[{row['number']}] ({row['kind']})  as_of={row['as_of']}")
        out.append(f"Q: {row['question']}")
        if row['kind'] == 'action':
            if not row['actions']:
                out.append("A: (no action)")
            for act in row['actions']:
                out.append(f"A: {act['type']}  {json.dumps(act['args'], ensure_ascii=False)}")
        elif row['kind'] == 'probe':
            out.append(f"probe: {row['probe']}")
            out.append(f"A: {row['answer']}")
            out.append(f"(abstained={row['abstained']})")
        else:
            out.append(f"A: {row['answer']}")
            evidence = ', '.join(row['sources']) or '(none)'
            out.append(f"abstained={row['abstained']}  evidence={evidence}")
        out.append('')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data')
    ap.add_argument('--questions', default='question.txt')
    ap.add_argument('--out', default='question_answers.jsonl')
    a = ap.parse_args()

    text = Path(a.questions).read_text(encoding='utf-8')
    items = parse_items(text)
    if not items:
        print(f'No questions found in {a.questions}')
        return
    rows = answer_items(items, a.data)
    print(render(rows))
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'Wrote {a.out}')


if __name__ == '__main__':
    main()
