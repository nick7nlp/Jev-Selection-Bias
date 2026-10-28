# Usage guide

Start with the [quick start](../README.md#quick-start) to install the package,
prepare the primary collection, and run a five-question trial. All commands
below run from the repository root.

## HTTP configuration

Copy the example for your model's interface to `configs/model.local.json`:

| Interface | Example | Endpoint variable | Key variable |
| --- | --- | --- | --- |
| Jev | [jev.example.json](../configs/jev.example.json) | `JEV_ENDPOINT` | `JEV_API_KEY` |
| Chat Completions | [chat.example.json](../configs/chat.example.json) | `CHAT_ENDPOINT` | `CHAT_API_KEY` |
| Responses | [responses.example.json](../configs/responses.example.json) | `RESPONSES_ENDPOINT` | `RESPONSES_API_KEY` |
| Anthropic Messages | [anthropic.example.json](../configs/anthropic.example.json) | `ANTHROPIC_ENDPOINT` | `ANTHROPIC_API_KEY` |

Set the endpoint variable to the **complete request URL**, including its path,
and set the key variable to your provider's key. Edit `model` in the copied
configuration to the identifier your provider accepts. Use only `parameters`
supported by that model and interface.

If the provider returns a different model name, such as a dated snapshot, add
an `accepted_model_names` list containing the expected response names. Keys
are read from environment variables; local configurations, downloaded data,
and run outputs are ignored by Git.

For local model weights and native decision heads, use the
[native model guide](NATIVE_MODELS.md).

## Dataset selection

SuperGPQA and LEXam form the primary collection. To add MMLU-Pro:

```bash
python3 scripts/prepare_questions.py \
  --datasets supergpqa lexam mmlu_pro \
  --output data/questions-mmlu.jsonl
```

To prepare all four benchmarks, obtain the official GPQA Diamond CSV as
described in the [data guide](DATA_SOURCES.md), then run:

```bash
python3 scripts/prepare_questions.py \
  --gpqa-csv /path/to/gpqa_diamond.csv \
  --output data/questions-all.jsonl
```

Pass the resulting file to `--questions` when evaluating and scoring. Omit
`--datasets` to include all four benchmarks, or select any subset present in
the question file. The data guide also covers offline reconstruction and
source attribution.

## Evaluation and scoring

The runner records candidate selections in JSONL. The scorer reads that file
and writes metrics as JSON. Use the same `--questions`, `--datasets`,
`--limit`, and `--protocol` settings for both commands.

| Option | Behavior |
| --- | --- |
| `--protocol all` | Both sequences: eight reordered and eight original-order presentations, sharing the first request. This is the default. |
| `--protocol orders` | Original, reverse, and six random orders: eight presentations. |
| `--protocol repeats` | Eight original-order presentations. |
| `--datasets NAME ...` | Select datasets; omit to select all four. |
| `--limit N` | Use the first N selected questions in manifest order, after dataset filtering. |
| `--workers N` | Concurrent HTTP requests; defaults to one. Native models require one. Available for `run` only. |
| `--dry-run` | Write request payloads without inference or credentials. Available for `run` only. |

For example, evaluate original-order consistency on five questions:

```bash
python3 -m jev_eval run \
  --questions data/questions.jsonl \
  --datasets supergpqa lexam --limit 5 --protocol repeats \
  --config configs/model.local.json \
  --output runs/repeat-trial.jsonl

python3 -m jev_eval score \
  --questions data/questions.jsonl \
  --datasets supergpqa lexam --limit 5 --protocol repeats \
  --predictions runs/repeat-trial.jsonl \
  --output runs/repeat-metrics.json
```

This requires **40 requests**. The full protocol requires **75 requests** for
five questions. Choose new output paths for each run; existing outputs are
never overwritten.

## Metrics

Selections are mapped to candidate identities before scoring, so moving the
same candidate to another position does not count as changing the decision.
The scorer returns per-presentation accuracy and cumulative metrics for each
sequence.

| Metric | Definition |
| --- | --- |
| Accuracy | Fraction of questions answered correctly at a given presentation. |
| Cumulative failed questions | Number of questions answered incorrectly at least once so far, counting each question once. |
| Distinct wrong candidates | Number of unique incorrect `(question, candidate)` pairs selected so far. This can exceed the question count when a model chooses multiple wrong options for the same question. |
| Cumulative candidate agreement | Fraction of questions receiving the same candidate selection throughout the sequence so far. |
| Retention of initial successes | Fraction of initially correct questions answered correctly throughout the sequence so far. |

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests use synthetic questions and local mock HTTP servers. They do not
call model providers or load model weights.
