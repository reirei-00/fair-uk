# Fair-UK

**Fairness evaluation for language models, built on EleutherAI's evaluation harness.**

Fair-UK supports the **WarBias** study of bias involving veterans and internally displaced people, including age and gender profiles. BBQ, StereoSet and WinoBias provide complementary measurements. Each benchmark reports its own metrics, supplemented by subgroup and worst-group comparisons.

> **Pilot:** datasets are human-unvalidated. Results describe these benchmark cases and do not certify that a model is fair.

## Quick start

Requires Python 3.10 or newer:

```sh
git clone https://github.com/reirei-00/fair-uk.git
cd fair-uk
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[fair-uk]'

fair-uk-eval list --format table
fair-uk-eval metrics --task bbq_uk
fair-uk-eval validate --task warbias_intersectional_uk
fair-uk-eval suite
```

The last command previews the six-task Ukrainian suite without running a model. To execute against a compatible Hugging Face causal checkpoint, replace the placeholders:

```sh
fair-uk-eval suite --execute \
  --model hf \
  --model-args 'pretrained=YOUR_MODEL,revision=IMMUTABLE_40_CHARACTER_COMMIT,device=cuda' \
  --output results/my-model
```

Select benchmarks with `--benchmarks` and languages with `--languages uk en`. WarBias has registered Ukrainian and English tasks; the other English counterparts currently require a local dataset bundle. Hosted chat APIs support WarBias generation; the other protocols require candidate token scores.

Runs produce raw predictions, native and group metrics, provenance, and JSON/Markdown/CSV reports. Matched bilingual runs also produce UK–EN comparisons. Benchmarks and languages keep separate scores.

## Documentation

- [Usage guide](docs/fair-uk.md): datasets, model configuration, bilingual preparation, resume, rescoring and comparisons.
- [Metrics reference](docs/worst-group-metrics.md): each benchmark's definitions, group analysis and interpretation limits.
- [Draft WarBias expansion](docs/fair-uk.md#local-warbias-expansion): validated local bundles, matched comparisons and benign-request judging.

## Attribution

Fair-UK extends [EleutherAI's Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Use `fair-uk-eval` for these protocols; upstream `lm-eval` remains available. See the [upstream README](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad8737ae7fad24cf64e50fc7fc31397bff586b9e/README.md) for general harness features. Software retains the [MIT license](LICENSE.md) and [harness citation](CITATION.bib); datasets retain their own licenses, listed in the usage guide.
