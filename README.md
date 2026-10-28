<h1 align="center">Jev-Selection-Bias</h1>

<p align="center">
  <b><a href="https://arxiv.org/abs/2610.35826">Understanding Jev-Style System One Models:<br>Does Structure Reduce Selection Bias?</a></b>
</p>

<p align="center">
  Mingyang Song · Mao Zheng · Zhiyuan Ma · Xinpeng Zhou<br>
  <b>Tencent</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.35826"><img src="https://img.shields.io/badge/arXiv-2610.35826-B45B5B?style=flat-square" alt="arXiv 2610.35826"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.10%2B-526B92?style=flat-square" alt="Python 3.10 or newer"></a>
  <a href="#evaluation-data"><img src="https://img.shields.io/badge/Benchmarks-4-4F827C?style=flat-square" alt="Four benchmarks"></a>
  <a href="CITATION.cff"><img src="https://img.shields.io/badge/Cite-this_work-736885?style=flat-square" alt="Cite this work"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.35826">Paper</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#evaluation-data">Data</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="#evaluation-protocol">Protocol</a> ·
  <a href="#citation">Citation</a>
</p>

**A simple test: keep the question and options fixed, change their order.**
We study selection bias in Jev, Jev reproductions, and LLMs by comparing
reordered options with repeated presentations in the original order.
Tracking individual decisions alongside accuracy reveals whether stable
scores also mean stable choices.

<p align="center">
  <img src="assets/protocol.svg" width="960" alt="Two sequences share an initial original-order request. Reordering adds one reverse and six random orders; repetition adds seven original-order requests. Each sequence has eight presentations, with fifteen requests per question in total.">
</p>

This repository provides **evaluation code and input manifests**: exact question
selections and permutations, scripts to reconstruct the inputs, and a shared
runner and scorer for HTTP interfaces and local decision models.

## Quick start

### 1. Install

Use **Python 3.10+** for HTTP evaluation. Native models have a separate
[setup guide](docs/NATIVE_MODELS.md). Run the commands below from the repository root.

```bash
git clone https://github.com/nick7nlp/Jev-Selection-Bias.git
cd Jev-Selection-Bias
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[data]'
```

### 2. Prepare the questions

Reconstruct the primary collection of **1,700 questions** from SuperGPQA and LEXam:

```bash
python3 scripts/prepare_questions.py \
  --datasets supergpqa lexam \
  --output data/questions.jsonl
```

This downloads the pinned sources and verifies the reconstructed inputs.
For MMLU-Pro and GPQA Diamond, see the [data guide](docs/DATA_SOURCES.md).

<details>
<summary><b>Preview the requests without calling a model</b></summary>

This creates the 75 requests for five questions without credentials or inference:

```bash
python3 -m jev_eval run \
  --questions data/questions.jsonl \
  --datasets supergpqa lexam --limit 5 \
  --config configs/jev.example.json \
  --output runs/requests.jsonl --dry-run
```

</details>

### 3. Configure a model

For a Jev endpoint:

```bash
cp configs/jev.example.json configs/model.local.json
export JEV_ENDPOINT="https://YOUR_HOST/YOUR_COMPLETE_ENDPOINT_PATH"
export JEV_API_KEY="YOUR_API_KEY"
```

Replace the placeholders with your provider's endpoint and key, and edit `model`
in `configs/model.local.json` to the model ID accepted by that endpoint. The
endpoint must be the **complete request URL**, including its path. For other
interfaces, choose the matching [configuration below](#supported-models).

### 4. Evaluate and score

Start with five questions (**75 requests** across both sequences):

```bash
python3 -m jev_eval run \
  --questions data/questions.jsonl \
  --datasets supergpqa lexam --limit 5 \
  --config configs/model.local.json \
  --output runs/trial-selections.jsonl

python3 -m jev_eval score \
  --questions data/questions.jsonl \
  --datasets supergpqa lexam --limit 5 \
  --predictions runs/trial-selections.jsonl \
  --output runs/trial-metrics.json
```

Remove `--limit 5` from **both** commands for the full primary collection, and choose
new output paths. Existing outputs are never overwritten. Use matching
`--datasets`, `--limit`, and `--protocol` settings when running and scoring.

## Evaluation data

| Benchmark | Questions | Options per question |
| --- | ---: | --- |
| [SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) | 1,200 | 4–10 |
| [LEXam](https://huggingface.co/datasets/LEXam-Benchmark/LEXam) | 500 | 4, 8, 16, or 32 |
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | 300 | 8, 9, or 10 |
| [GPQA Diamond](https://github.com/idavidrein/gpqa) | 198 | 4 |

SuperGPQA and LEXam form the primary collection of **1,700 questions**.
MMLU-Pro and GPQA Diamond provide complementary evaluations, for **2,198
questions** across all four datasets.

[`data/cases.jsonl`](data/cases.jsonl) specifies the exact selected questions, source identifiers,
answer indices, metadata, and input fingerprints. [`data/permutations.jsonl`](data/permutations.jsonl)
contains every candidate permutation used by the test. Question and option text
are reconstructed from the upstream datasets; see
[data sources and attribution](docs/DATA_SOURCES.md).

## Supported models

| Interface | Configuration |
| --- | --- |
| Jev | [jev.example.json](configs/jev.example.json) |
| Chat Completions | [chat.example.json](configs/chat.example.json) |
| Responses | [responses.example.json](configs/responses.example.json) |
| Anthropic Messages | [anthropic.example.json](configs/anthropic.example.json) |
| Native decision head | [native.example.json](configs/native.example.json) |

For **AutoTrust, StartLux, and other Jev reproductions**, follow the
[native model guide](docs/NATIVE_MODELS.md) for model sources, dependencies, and
backend selection. Set `MODEL_PATH` to a local model directory. Native evaluation
uses the model's task format and decision head, with one worker. See the
[usage guide](docs/USAGE.md) for configuration, run options, and scoring.

## Evaluation protocol

Each question has **two sequences of eight presentations**:

- **Reordered:** the original order, the reverse order, and six recorded random
  permutations.
- **Repeated:** eight presentations of the original order.

The sequences share their first presentation, giving **15 requests per
question**. Every presentation uses the complete candidate set. The code maps
returned keys to stable candidate identities before scoring.

Moving a candidate to a different position does not itself count as changing
the decision.

The scorer measures **accuracy**, **cumulative failed questions**, **distinct
wrong candidates**, **candidate agreement**, and **retention of initial
successes**. See [metric definitions](docs/USAGE.md#metrics) for the meaning of
each output.

## Citation

If you use this code or the evaluation manifests, please cite our paper:

```bibtex
@misc{song2026understandingjevstylemodelsdoes,
  title = {Understanding Jev-Style System One Models: Does Structure Reduce Selection Bias?},
  author = {Mingyang Song and Mao Zheng and Zhiyuan Ma and Xinpeng Zhou},
  year = {2026},
  eprint = {2610.35826},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2610.35826}
}
```

The citation is also available in [CITATION.cff](CITATION.cff) and
[references.bib](references.bib). Please also cite the benchmarks you use;
their paper links and BibTeX entries are listed in
[data sources and attribution](docs/DATA_SOURCES.md#benchmark-papers).

## Acknowledgments

This evaluation builds on SuperGPQA, LEXam, MMLU-Pro, and GPQA, and on the native
interfaces provided by the decision-model authors. Dataset attribution and
licenses are documented in [dataset notices](third_party/DATA_NOTICES.md);
attribution for adapted code appears in [code notices](third_party/CODE_NOTICES.md).
Model weights are obtained separately from the sources linked in the
[native model guide](docs/NATIVE_MODELS.md).
