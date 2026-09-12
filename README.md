# Fair-UK

**Fairness evaluation for language models, with reproducible benchmarks and worst-group reporting.**

Fair-UK supports **WarBias**, a study of bias involving veterans and internally displaced people, including age and gender profiles. WarBias is evaluated in Ukrainian and English; BBQ-UK, StereoSet-UK and WinoBias-UK provide complementary measurements.

[Quick start](#quick-start) · [Benchmarks](#benchmarks) · [Metrics and reports](#metrics-and-reports) · [Documentation](#documentation)

> **Pilot release.** The registered datasets are human-unvalidated. Results describe performance on these benchmark cases; they do not certify that a model is fair.

## Quick start

Requires Python 3.10 or newer. Install in a dedicated environment:

```sh
git clone https://github.com/reirei-00/fair-uk.git
cd fair-uk
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[fair-uk]'

fair-uk-eval list
fair-uk-eval validate --task warbias_intersectional_uk
```

These commands list the tasks and validate a pinned dataset without running a model. For model evaluation, start with the [run guide](docs/fair-uk.md#install-and-run).

## Benchmarks

| Benchmark | Language | Task names |
| --- | --- | --- |
| **WarBias — base** | Ukrainian, English | `warbias_uk`, `warbias_en` |
| **WarBias — intersectional** | Ukrainian, English | `warbias_intersectional_uk`, `warbias_intersectional_en` |
| BBQ-UK | Ukrainian | `bbq_uk` |
| StereoSet-UK | Ukrainian | `stereoset_uk` |
| WinoBias-UK Natural | Ukrainian | `winobias_uk_natural` |
| WinoBias-UK Controlled | Ukrainian | `winobias_uk_controlled` |

Each task has a pinned HF revision, file checksum and its own scoring protocol. See [dataset sizes and protocols](docs/fair-uk.md#registered-datasets-and-protocols).

## Metrics and reports

- **Native benchmark scores** alongside accuracy, stereotypical responses and invalid-answer rates where applicable.
- **Worst-group performance**, group gaps and min/max rate ratios, with group identities, case counts and exploratory source-case bootstrap intervals.
- **Paired UK–EN WarBias comparisons** that keep translations and demographic variants of each case together.
- **Reproducible outputs:** raw predictions, model/data/code fingerprints, Markdown reports, JSON results and CSV tables. Completed predictions can be resumed or rescored.

Benchmarks and languages retain separate scores. A parity ratio of one means equal measured rates; it can still accompany poor performance. See the [metric definitions and limitations](docs/worst-group-metrics.md).

## Documentation

| I want to… | Guide |
| --- | --- |
| Evaluate Gemini models on WarBias | [Gemini pilot guide](docs/gemini-warbias-pilot.md) |
| Evaluate OpenAI models on WarBias | [OpenAI pilot guide](docs/openai-warbias-pilot.md) |
| Run a model or rescore predictions | [Installation and usage](docs/fair-uk.md) |
| Compare Ukrainian and English results | [Paired language comparisons](docs/fair-uk.md#paired-warbias-uken-comparisons) |
| Combine reports into model-by-task tables | [Experiment tables](docs/fair-uk.md#experiment-tables) |
| Understand group metrics and uncertainty | [Worst-group comparisons](docs/worst-group-metrics.md) |
| Check validation and release limitations | [Pilot release notes](docs/fair-uk-release.md) |
| See planned work | [Development roadmap](docs/fair-uk-roadmap.md) |

## Built on EleutherAI's evaluation harness

Fair-UK extends the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), reusing its model backends. Use `fair-uk-eval` for this toolkit's protocols; the upstream `lm-eval` command is also available.

General backend configuration and upstream features are covered in the [upstream README](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad8737ae7fad24cf64e50fc7fc31397bff586b9e/README.md). Software retains the [MIT license and EleutherAI attribution](LICENSE.md); the [harness citation](CITATION.bib) is included. Datasets retain their [separate licenses and provenance](docs/fair-uk.md#validation-and-attribution).
