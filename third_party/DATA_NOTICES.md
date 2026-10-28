# Third-party dataset notices

The evaluation manifests select and describe records from the datasets below. Each dataset remains subject to its upstream license and applicable source-content terms. The reconstruction script selects the listed records, converts their fields to a common input schema, and applies the recorded initial candidate permutation for GPQA Diamond. It preserves question and option strings.

## SuperGPQA

Source: [m-a-p/SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA), revision `4430d4458112c7d4497fdcf94d7cc223313d6acf`.

The [pinned dataset card](https://huggingface.co/datasets/m-a-p/SuperGPQA/blob/4430d4458112c7d4497fdcf94d7cc223313d6acf/README.md) declares the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/) and states that it "asserts no copyright over the underlying content." The card requires appropriate credit to third-party creators and compliance with the licenses of its constituent datasets. Consult that card for the source-dataset references.

## LEXam

Source: [LEXam-Benchmark/LEXam](https://huggingface.co/datasets/LEXam-Benchmark/LEXam), revision `31e57ae395f92ed7284a4c28d278c55da892b898`.

The [pinned dataset card](https://huggingface.co/datasets/LEXam-Benchmark/LEXam/blob/31e57ae395f92ed7284a4c28d278c55da892b898/README.md) declares [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The evaluation uses selected test records from `mcq_4_choices`, `mcq_8_choices`, `mcq_16_choices`, and `mcq_32_choices`.

## MMLU-Pro

Source: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`.

The [pinned dataset card](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/b189ec765aa7ed75c8acfea42df31fdae71f97be/README.md) declares `license: mit`; see the [MIT license text](https://opensource.org/license/mit). No separate dataset copyright header is supplied at this revision, so none is added here. The card describes the underlying question sources. This dataset declaration is distinct from the Apache-2.0 license for the accompanying software repository.

## GPQA Diamond

Source: [idavidrein/gpqa](https://github.com/idavidrein/gpqa), `dataset/gpqa_diamond.csv` in the official archive identified in `data/sources.json`.

The publisher's dataset notice is reproduced verbatim in [`GPQA-license.txt`](GPQA-license.txt). The dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Its source README and [Hugging Face access page](https://huggingface.co/datasets/Idavidrein/gpqa) describe acquisition and request that examples not be posted online to reduce training contamination. The reconstruction script takes a user-supplied source CSV and does not publish examples.
