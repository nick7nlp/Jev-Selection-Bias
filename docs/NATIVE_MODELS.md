# Native decision models

The `native` API loads a local checkpoint and calls its decision head. Select the
backend that matches the checkpoint; these formats are not interchangeable.

| Backend | Checkpoints | Native interface |
| --- | --- | --- |
| `jev` | [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B), [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B), [autotrust/JEV-35B](https://huggingface.co/autotrust/JEV-35B) | `bare-v1` prompt and trained choice head; merge the checkpoint adapter when present. |
| `jev_bidir` | [autotrust/AutoTrust-JEV-27B-v1.1](https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1) | Checkpoint formatter and the official StartLux bidirectional Transformers backend. |
| `startlux` | StartLux-Decision-v1 [0.8B](https://huggingface.co/startlux-models/StartLux-Decision-v1-0.8B), [2B](https://huggingface.co/startlux-models/StartLux-Decision-v1-2B), [4B](https://huggingface.co/startlux-models/StartLux-Decision-v1-4B), [9B](https://huggingface.co/startlux-models/StartLux-Decision-v1-9B), [27B](https://huggingface.co/startlux-models/StartLux-Decision-v1-27B), [35B-A3B](https://huggingface.co/startlux-models/StartLux-Decision-v1-35B-A3B) | Official `startlux_decision` formatter and `StartLuxDecision.decide_batch`. |
| `pplx_native` | [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/tree/14c9d90137962d300f3de95633f7b6535e80c5ce) | Native chat format, noncausal full attention with recurrent attention retained, and the checkpoint's bias-free choice head. |
| `gemma4_native` | [autotrust/JEV-Gemma4-26B-A4B](https://huggingface.co/autotrust/JEV-Gemma4-26B-A4B/tree/068ec870779c3672ad552128bb0f6f24aed32516) | `bare-v1` plus the native BOS token, adapter, choice slots, and logit softcap. |

Display names prefixed with `AutoTrust-` need not match their Hugging Face
repository names. Download the complete checkpoint, including tokenizer,
configuration, formatter, readout, and adapter files where supplied. Checkpoint
downloads and model licenses are managed separately from this repository.

## Dependencies

Use Python 3.12 for native inference. Install the shared dependencies, with a
PyTorch CUDA build appropriate for the device:

```bash
python3 -m pip install -e '.[native]'
```

The following versions provide a concrete dependency target. Backend-specific
dependencies are listed below.

| Package | Version or requirement |
| --- | --- |
| PyTorch | `2.11.0` with a CUDA build appropriate for the device |
| Transformers | `5.8.1` |
| safetensors | `0.7.0` |
| PEFT | `0.18.1` for adapters merged by the `jev` and `gemma4_native` backends |
| Accelerate | `>=1.1.0`, the requirement in Transformers 5.8.1 for `device_map` loading |

For `jev` and `gemma4_native` checkpoints with adapters, also install
`peft==0.18.1`. StartLux uses its own dependency requirements below.

Transformers 5.8.1 provides the `Gemma4ForConditionalGeneration` and
`Qwen3_5Model` implementations used here. The CUDA MoE path requests
`experts_implementation="grouped_mm"`; availability also depends on the PyTorch
build and GPU architecture. An older Transformers installation is not an
equivalent fallback for these model classes or attention interfaces.

The two StartLux packages have different installation paths. Both source
references below use the official
[StartLux-Decision repository at `a99977d`](https://github.com/StartLuxLabs/StartLux-Decision/tree/a99977d9332decc9fb27a2488b4a83b533b6cd5f).

For `jev_bidir`, install the repository's root package:

```bash
python3 -m pip install 'startlux-decide[transformers] @ git+https://github.com/StartLuxLabs/StartLux-Decision.git@a99977d9332decc9fb27a2488b4a83b533b6cd5f'
```

That revision declares `startlux-decide==1.1.0`; its Transformers extra requires
`torch>=2.4` and `transformers>=5.8,<5.9`. The adapter calls the backend's `_load`
and `_readout` methods, so the pinned source revision matters.

For `startlux`, follow the official
[v1 installation instructions](https://github.com/StartLuxLabs/StartLux-Decision/blob/a99977d9332decc9fb27a2488b4a83b533b6cd5f/v1/README.md).
The root package above does **not** install `startlux_decision`. Clone the same
revision, install `v1/requirements.txt`, and add its `v1` directory to
`PYTHONPATH`:

```bash
git clone https://github.com/StartLuxLabs/StartLux-Decision.git
git -C StartLux-Decision checkout a99977d9332decc9fb27a2488b4a83b533b6cd5f
python3 -m pip install -r StartLux-Decision/v1/requirements.txt
export PYTHONPATH="$(pwd)/StartLux-Decision/v1${PYTHONPATH:+:$PYTHONPATH}"
```

Alternatively, use the `startlux_decision` package supplied inside a complete
official v1 checkpoint by adding that checkpoint directory to `PYTHONPATH`.
The upstream v1 requirements specify `torch>=2.8`, `transformers>=5.8`, and, on
Linux, `flash-linear-attention>=0.5.2` and `causal-conv1d>=1.7`. They also include
training and data dependencies, including `peft>=0.20`; these are upstream v1
requirements, distinct from the PEFT reference version for our direct head
adapters above. Follow the upstream CUDA installation instructions for compiled
kernels.

## Configuration and execution

Copy [`configs/native.example.json`](../configs/native.example.json) to
`configs/native.local.json`, change `model` and `backend`, and point `MODEL_PATH`
to the local checkpoint:

```json
{
  "api": "native",
  "model": "AutoTrust-JEV-9B",
  "backend": "jev",
  "checkpoint_env": "MODEL_PATH",
  "device": "cuda:0",
  "max_input_tokens": 16384
}
```

`checkpoint_env` names an environment variable containing the checkpoint path.
You may instead set `checkpoint` directly in a private configuration file.
`model` is the identifier recorded with predictions. `max_input_tokens` is an
input limit, not a generation budget; overlong inputs are rejected without
truncation.

```bash
cp configs/native.example.json configs/native.local.json
```

After editing that file, run a small trial on the inputs prepared in the
[quick start](../README.md#quick-start):

```bash
export MODEL_PATH=/absolute/path/to/checkpoint
python3 -m jev_eval run \
  --questions data/questions.jsonl \
  --datasets supergpqa lexam --limit 5 \
  --config configs/native.local.json \
  --output runs/native-trial.jsonl \
  --workers 1
```

Native backends require `--workers 1`. They select the highest-probability
candidate from the native decision head and map it back to its original key.
Remove `--limit 5` for the full primary collection, or add `--dry-run` to inspect
task payloads without loading weights.

Source attribution for adapted code appears in
[`third_party/CODE_NOTICES.md`](../third_party/CODE_NOTICES.md).
