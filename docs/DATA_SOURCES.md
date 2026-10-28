# Evaluation data

The evaluation set contains 2,198 questions: 1,200 from SuperGPQA, 500 from LEXam, 300 from MMLU-Pro, and all 198 questions in GPQA Diamond. `data/cases.jsonl` identifies the selected source records and their original candidate order; `data/permutations.jsonl` specifies the presentation orders. Question and option text are reconstructed locally from the publishers' datasets.

## Reconstruct the inputs

Install the repository dependencies, then run from the repository root:

```bash
python3 scripts/prepare_questions.py \
  --datasets supergpqa lexam mmlu_pro \
  --output data/questions.jsonl
```

The script downloads the pinned source files, verifies their SHA-256 checksums, and writes the selected inputs. Cached downloads are stored in `.cache/sources`. It preserves question and candidate strings exactly, including their whitespace. It validates candidate order, answer indices, and the per-question `input_sha256` in the case manifest.

For GPQA Diamond, obtain `dataset/gpqa_diamond.csv` by following the [official GPQA repository](https://github.com/idavidrein/gpqa). Pass the local CSV explicitly:

```bash
python3 scripts/prepare_questions.py \
  --gpqa-csv /path/to/dataset/gpqa_diamond.csv \
  --output data/questions.jsonl
```

This command reconstructs all four datasets. If the output already exists, select another output path or add `--force`. Source question files and reconstructed inputs are intended for local evaluation and are excluded from version control. GPQA's publishers ask users to avoid posting examples online to limit benchmark contamination; follow the publisher's access instructions and data terms.

For offline use, add `--offline` and place the pinned source files under `--source-dir` (default `.cache/sources`) as follows:

```text
supergpqa/SuperGPQA-all.jsonl
lexam/mcq_4_choices/test-00000-of-00001.parquet
lexam/mcq_8_choices/test-00000-of-00001.parquet
lexam/mcq_16_choices/test-00000-of-00001.parquet
lexam/mcq_32_choices/test-00000-of-00001.parquet
mmlu_pro/data/test-00000-of-00001.parquet
```

GPQA continues to use the explicit `--gpqa-csv` path. Source revisions, file checksums, and download URLs are recorded in [`data/sources.json`](../data/sources.json).

## Source attribution

| Dataset | Publisher source | Selection | Dataset license |
|---|---|---|---|
| SuperGPQA | [m-a-p/SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) | 1,200 questions from the official release | [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) |
| LEXam | [LEXam-Benchmark/LEXam](https://huggingface.co/datasets/LEXam-Benchmark/LEXam) | 500 test questions across the 4-, 8-, 16-, and 32-choice configurations | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| MMLU-Pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | 300 test questions with 8–10 options | MIT, as declared in the pinned dataset card |
| GPQA Diamond | [idavidrein/gpqa](https://github.com/idavidrein/gpqa) | All 198 Diamond questions | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Each upstream dataset retains its own license. Dataset attribution and relevant notices are collected in [`third_party/DATA_NOTICES.md`](../third_party/DATA_NOTICES.md).

SuperGPQA's publisher states that ODC-BY covers the database and asserts no copyright over the underlying content; its source datasets have their own attribution and licensing requirements. MMLU-Pro's dataset card declares MIT, while its separate software repository uses Apache-2.0. GPQA's dataset license is CC BY 4.0, distinct from the MIT license for its repository software.

## Benchmark papers

Please cite the original benchmark papers when using their data. BibTeX entries
for all four are included in [`references.bib`](../references.bib).

- **SuperGPQA:** [SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a3c5af1f56fc73eef1ba0f442739f5ca-Abstract-Datasets_and_Benchmarks_Track.html).
- **LEXam:** [LEXam: Benchmarking Legal Reasoning on 340 Law Exams](https://openreview.net/forum?id=xNhbMyXsJn).
- **MMLU-Pro:** [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574).
- **GPQA:** [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022).

## Input format and transformations

The reconstructed JSONL rows contain `id`, `dataset`, `source_id`, `split`, `state`, `question`, `options`, `gold_index`, `gold_candidate_id`, `candidate_ids`, `source_labels`, `option_count`, `subject`, and `difficulty`.

- `state` is the empty string for these four datasets.
- `candidate_ids` identifies candidates in the original evaluation order as `candidate_0`, `candidate_1`, and so on. These IDs do not change when a presentation reorders the candidates.
- SuperGPQA and MMLU-Pro retain their source candidate order and letter keys. LEXam retains its source candidate order and uses `source_labels: null`; its choice list is decoded with `ast.literal_eval` without executing source text.
- GPQA's source columns list the correct answer first. The released `initial_source_order` records the permutation used to create each question's original evaluation order. Subsequent presentation permutations refer to that order. `gold_index` is the position of source candidate 0 in `initial_source_order`.
- `gold_index` is zero-based. Topic and difficulty metadata come from the upstream records; missing difficulty is represented by `null`.

The reconstruction does not rewrite question or option text. The public data manifests select source rows and specify candidate order; GPQA's initial ordering is the only initial reordering of source candidate text.

`input_sha256` checks the UTF-8 encoding of a JSON object containing exactly `state`, `question`, `options`, `gold_index`, `candidate_ids`, and `source_labels`, serialized using Python's `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))`. No whitespace stripping or Unicode normalization is applied to the strings.
