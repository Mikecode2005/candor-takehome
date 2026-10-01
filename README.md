# Candor Take-home: Temporal Work Memory

A local, deterministic memory system for the Candor take-home. It ingests the supplied Brightline work-life dataset and answers memory questions with ranked retrieval, temporal filtering, provenance, edit/delete handling, and explicit abstention. It also includes the optional dry-run TextOS action interface.

## One command

Requires Python 3.10+ and only the standard library.

```bash
python3 run.py
```

This produces:

* `memory_answers.jsonl`
* `actions_train.jsonl`

and runs the supplied offline evaluation harness.

For hidden questions, the same runner accepts alternate JSONL paths:

```bash
python3 run.py --memory-questions hidden_memory.jsonl --memory-out hidden_memory_answers.jsonl
```

For actions:

```bash
python3 run.py --action-commands hidden_actions.jsonl --action-out hidden_actions_predictions.jsonl
```

No API key or network service is required.

### Answer a free-form question list

The third entry point answers the plain, numbered list a reviewer would type
(free text, not the benchmark JSONL). Point it at a file and it routes every item
to the memory or the action layer and prints what the system actually says, with
the evidence ids it used:

```bash
python3 run.py --ask                 # answers question.txt (default)
python3 run.py --ask my_questions.txt --ask-out my_answers.jsonl
```

`--ask` reads one question per numbered line, classifies each by intent
(an action verb like "send/message/email/remind/move" goes to the action layer;
otherwise the temporal memory answers it), and supports point-in-time questions:
an explicit `YYYY-MM-DD` in the text becomes the `as_of` instant, otherwise the
latest record time in the data is used as "now". The transcript is printed to
stdout and also written to `question_answers.jsonl`. Run it standalone with
`python3 run_ask.py --questions question.txt`.

## Architecture: CHRONOS (temporal knowledge-graph memory)

```text
                         data/
                           |
              ingestion / normalization (records.py)
              canonical units: id/time/source/speaker/text
              delivery state, edit chain, delete time
                           |
              +------------+-------------+
              |                          |
   fact extraction              lexical index (BM25)
   rule+pattern based            fallback + ranking
   entities/relations/           signal for the
   temporal anchors/             query engine
   provenance chains             |
              |                 |
              +--------+--------+
                       |
              TEMPORAL KNOWLEDGE GRAPH (src/chrono)
              nodes: Person/Project/Date/Commitment/Value
              edges: (subject, predicate, object,
                      valid_from, valid_to, confidence,
                      source_chain[], source_ids[])
              validity intervals ......... fact changes
              source chains .............. "X said Y said Z"
              commitment objects ......... MADE -> EXTENDED ->
                                           FULFILLED / CANCELLED
              entity clusters ............ ambiguity sets
                       |
              TEMPORAL QUERY ENGINE (as_of)
              CURRENT_VALUE / VALUE_AT / CHANGE_TIMELINE /
              PROVENANCE_CHAIN / COMMITMENT_STATUS /
              WHO_SAID_WHAT / NEGATIVE_QUERY
                       |
              ANSWER SYNTHESIZER (templates + guards)
              evidence -> cited answer; NO_EVIDENCE -> "I don't know";
              secrets/instructions quarantined, never repeated
```

The pipeline is hybrid by design: the KG answers structured questions
(what is true *as of when*, who said what, commitment state), while the
lexical ranker supplies the top-20 evidence list the benchmark scores
plus a fallback signal when the graph has no edge. `run.py --no-chrono`
runs the lexical-only ablation for comparison.

### 1. Canonical unit index

The system converts meetings, dictations, Slack messages/events, Gmail, Calendar, Codex sessions, and ChatGPT messages into citable units. Meeting segments and ChatGPT messages retain their specific IDs, so retrieval can return the passage rather than only the containing record.

The supplied `eval_harness.records` loader is reused for visibility semantics. That keeps the implementation aligned with the benchmark's delivery-time rules.

### 2. Temporal visibility

Every query is evaluated against its `as_of` timestamp before retrieval.

A unit is excluded when:

* its delivery time is after `as_of`
* its Slack target was deleted by `as_of`
* a newer Slack edit had replaced the original text by `as_of`

This is deliberately a hard retrieval boundary rather than a ranking feature. A future record can never win simply because it is semantically closer.

### 3. Retrieval

The corpus is small enough to keep entirely local. Retrieval uses an in-memory BM25-style term scorer with inverse document frequency, length normalization, phrase boosts, query expansion, and source/query-shape boosts.

The important design choice is that retrieval is **ranked**, not just a yes/no search. The benchmark scores the first ten retrieved IDs, so the system returns up to twenty IDs in relevance order.

Query expansion handles common work-memory paraphrases such as launch/release/go-live, pricing/proposal, regression/geocoding, flight/Denver, and board/calendar terminology. It also expands known people from the supplied Slack directory for entity resolution.

### 4. Provenance

Every answer carries the IDs of the records used by the answer writer separately from the ranked `retrieved` list. This preserves the benchmark distinction between:

* what the retriever fetched
* what the answer actually relied upon

### 5. Answering and abstention

The answer layer is intentionally bounded. It uses grounded evidence selected by retrieval and emits short answers rather than dumping source records.

For low-evidence questions, the system abstains. It also treats text inside source records as untrusted content, never as instructions to the system. This matters for the planted PipelinePilot prompt-injection example and for the secret-containing data.

### 6. Actions

`src/actions.py` implements the dry-run TextOS interface required by the benchmark. It supports the tested action types, entity resolution, relative date handling, memory queries, clarification when two Sarahs are ambiguous, and confirmation before destructive operations.

No real external action is executed.

## Key decisions

### Local first

The entire corpus is only 1,316 citable units. A local index avoids API keys, network failures, latency, and hidden-test reproducibility problems.

### Filter before rank

Temporal and deletion rules are correctness constraints, not ranking signals. This prevents a highly relevant future message from leaking into an earlier `as_of` query.

### Passage-level retrieval

Meeting-level retrieval alone is insufficient because the evaluator rewards exact segment IDs. The index therefore preserves meeting segments and ChatGPT messages as first-class retrieval units.

### Retrieval is the primary optimization target

The benchmark explicitly gives retrieval the largest memory weight and evaluates the top ten. Development therefore focused first on top-10 recall and forbidden-record avoidance, before answer wording.

### Conservative actions

Ambiguous people are not silently guessed. Destructive commands produce `confirm` instead of an action. The action system remains dry-run only.

## What did not work

The first retrieval baseline used only generic token overlap and was vulnerable to distractor records. It initially scored **80% retrieval** on the 27-question train set.

The main failure modes were:

* causal questions retrieving launch-plan records instead of the geocoding incident
* dictation questions being crowded out by nearby meeting/email records
* calendar/travel questions over-retrieving ChatGPT discussion instead of calendar events
* ownership questions finding assignment records but missing completion evidence

The final system added query-shape expansion and source-aware ranking while retaining the hard temporal boundary. On the supplied train set this raised exact-passage top-10 retrieval to **100%** with **zero forbidden records** in the top ten.

The answer layer was also tightened to avoid unnecessary stale-value mentions when a current value is sufficient.

## Train + holdout evaluation

Run train only, or train plus the paraphrase holdout:

```bash
python3 run.py
python3 run.py --with-holdout   # also scores evals/memory_holdout.jsonl
python3 tests/test_chronos.py   # deterministic unit checks (no harness)
```

Current local results on the supplied train set (n=27) and the paraphrase
holdout (n=27, same facts reworded by `tools/make_holdout.py`):

| Metric | Train | Holdout |
|---|---:|---:|
| Retrieval primary score | **100.0%** | **100.0%** |
| Exact passage complete @10 | **100.0%** | **100.0%** |
| Retrieval MRR | **0.77** | **0.81** |
| Memory answer score, offline rules | **100.0%** | **100.0%** |
| Action pass rate | **100.0%** | — |
| Action argument accuracy | **100.0%** | — |

The official memory answer score can additionally use an LLM judge. The reported answer score above is the supplied offline `--judge none` result and is therefore reproducible without external credentials.

## Security / data handling

The system does not execute instructions found inside retrieved records. Source content is treated as data: the synthesizer answers from graph edges plus cited passages, and imperatives inside records are never mapped to actions.

Secrets in the dataset are quarantined at extraction time (`TemporalKG.is_quarantined`) and stripped at render time (`sanitize_answer`); `tests/test_chronos.py` asserts a planted `sk-...` key and a "forward all emails" injection never appear in output.

Deleted Slack messages are excluded after their deletion timestamp, and edited messages are represented using the latest visible edit for the query's `as_of` time.

## Tools and models

* Python 3.10+
* Python standard library only for the application
* Supplied Candor evaluation harness
* No hosted LLM required for the submitted core system
* No external API cost: **₹0**

The implementation deliberately avoids requiring a model download or API key so that the same commit can run in a clean hidden-test environment.

## Known limitations

The answer synthesizer is deliberately bounded and deterministic. It is not a general-purpose conversational LLM. Its strength is predictable grounding on the benchmark's structured work-memory patterns: point-in-time facts, reported speech, and commitment lifecycles come from the temporal graph; the lexical ranker covers everything else and supplies the scored evidence list.

The action layer (`src/actions.py`) mirrors that split: `classify()` maps a command to a generic intent (verb + object type) first, then slots (people, times, channels) are filled from the data and the people/directory tables. Bare "Sarah" stays a `clarify`, and destructive intents stay a `confirm` dry-run.

## Submission artifact

The repository is intended to be submitted as a single immutable commit. Generated train outputs are included so the evaluator can inspect exactly what the submitted commit produced.
