# Third-party code notices

## PPLX native decision interface

The PPLX prompt formatter and noncausal full-attention support in
`jev_eval/native.py` are adapted from the source implementation distributed with
[`perplexity-ai/pplx-decider-v1.1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b),
revision `14c9d90137962d300f3de95633f7b6535e80c5ce`:

- [Original source: `source/src/autojev/model.py`](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/source/src/autojev/model.py)
- [Apache License 2.0, as distributed upstream](PPLX-license.txt)
- [Unmodified upstream notice](PPLX-NOTICE.txt)

This repository adapts the implementation to single-question choice evaluation,
uses the common candidate-mapping interface, and supports the applicable
Transformers attention-mask interfaces. It does not distribute model weights.
The upstream notice is retained verbatim; its descriptions refer to the upstream
artifact.

StartLux backends are imported from their respective upstream packages. Model
and package sources are listed in [native model setup](../docs/NATIVE_MODELS.md).
Those packages and model weights retain their own licenses and are not bundled
here. Dataset notices are listed separately in [DATA_NOTICES.md](DATA_NOTICES.md).
