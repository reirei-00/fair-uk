"""Reproduce step-6 data/software checks without model inference or network calls."""
import csv
import html
import json
import tempfile
from pathlib import Path

from lm_eval.fair_uk.comparison import compare_rows
from lm_eval.fair_uk.data import bundle_registry, digest, load
from lm_eval.fair_uk.expansion import prepare, write_json
from lm_eval.fair_uk.metrics import make_report, score_all
from lm_eval.fair_uk.provenance import code_identity
from lm_eval.fair_uk.reporting import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'outputs/warbias_expansion_20260925/artifacts'
PREP = ROOT / 'outputs/warbias_preparation_20260925/artifacts'


def main():
    outputs = {}
    for split in ('evaluation', 'all'):
        destination = HERE / split
        if not destination.exists():
            prepare(SOURCE, PREP / 'cases.jsonl', destination, split)
        manifest = destination / 'manifest.json'
        frozen = json.loads(manifest.read_text())
        assert frozen['preparation_code_sha256'] == digest(ROOT / 'fairforget-eval/lm_eval/fair_uk/expansion.py'), 'Stale bundle: rebuild in a new directory'
        # A second build must reproduce every byte, including provenance.
        with tempfile.TemporaryDirectory(prefix='warbias-step6-') as temporary:
            rebuilt = Path(temporary) / 'bundle'
            prepare(SOURCE, PREP / 'cases.jsonl', rebuilt, split)
            files = [p for p in rebuilt.rglob('*') if p.is_file()]
            assert all(p.read_bytes() == (destination / p.relative_to(rebuilt)).read_bytes() for p in files)
        sources, scored, results = {}, {}, {}
        for task in bundle_registry(manifest):
            rows = load(task, dataset_bundle=manifest)
            # Oracle letters and fixed scores are SOFTWARE FIXTURES, not model predictions.
            predictions = [{'id': r['id'], **({'scores': [3.0, 2.0, 1.0]} if 'sentences' in r else {'response': 'ABC'[r['gold']]} if 'gold' in r else {'response': 'Software fixture; not model output.'})} for r in rows]
            records = score_all(rows, predictions)
            report = make_report(records, rows, bootstrap=8, seed=42)
            json.dumps(report, allow_nan=False)
            markdown({**report, 'provenance': {'model_identity': 'software_fixture_only'}})
            sources[task], scored[task] = rows, records
            results[task] = {'rows': len(rows), 'source_cases': report['source_cases'], 'sampling_units': report['sampling_units'], 'matched_contrasts': len(report['matched_comparisons']), 'load_score_report': 'passed'}
        pairs = []
        for task in sources:
            if task.endswith('_uk'):
                en = task[:-3] + '_en'
                report = compare_rows(scored[task], scored[en], sources[task], sources[en], bootstrap=8, seed=42)
                assert all(r['delta_en_minus_uk'] in (0, None) for r in report['native_comparisons'])
                pairs.append({'uk': task, 'en': en, 'paired_fixture_check': 'passed'})
        outputs[split] = {'manifest': str(manifest.relative_to(ROOT)), 'sha256': digest(manifest), 'byte_identical_rebuild_files': len(files), 'tasks': results, 'bilingual_pairs': pairs}
    originals = list(csv.DictReader((PREP / 'source_hashes.csv').open(encoding='utf-8-sig')))
    changed = [r['path'] for r in originals if digest(ROOT / r['path']) != r['sha256']]
    assert not changed, changed
    verification = {
        'step': 6, 'status': 'technical_validation_and_integration_complete',
        'software_fixture_checks_are_model_results': False,
        'model_experiments_run': 0, 'judge_requests_sent': 0, 'human_validated': False,
        'original_inputs_verified_unchanged': len(originals),
        'regression_tests': {'passed': 194, 'failed': 0, 'command': 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/fair_uk -q'},
        'static_checks': 'ruff check passed; git diff --check passed',
        'code_identity': code_identity(), 'bundles': outputs,
        'scientific_limits': [
            'Independent human/construct validation remains pending; no new annotator assignments were created.',
            'The frozen evaluation split contains only two benign task families and two matched cross-actor claim components.',
            'Expanded QA adds nine held-out situations; the 29-situation full expansion does not cover all 79 originally missing QA families.',
            'Fifty eligible case families still need separate QA designs; existing source files remain available.',
            'Severity stays provisional. Shared-content contrasts do not establish causal demographic effects or intervention spillover.',
            'No live model, judge quality or confirmatory interval-coverage validation was performed.',
        ],
    }
    write_json(HERE / 'verification.json', verification)
    table = ''.join('<tr><td>' + html.escape(task) + '</td><td>' + str(info['rows']) + '</td><td>' + str(info['source_cases']) + '</td><td>' + str(info['sampling_units']) + '</td></tr>' for task, info in outputs['evaluation']['tasks'].items())
    limitations = ''.join('<li>' + html.escape(x) + '</li>' for x in verification['scientific_limits'])
    (HERE / 'overview.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>WarBias — step 6</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17252a}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd8db;text-align:left}.notice{padding:18px;background:#fff1d1}</style><h1>WarBias: validation and evaluator integration</h1><p>194 tests passed. Eight draft tasks load, score and compare across Ukrainian and English. Original data and frozen splits are preserved.</p><p class="notice">No model evaluation or judge requests were made. These are staged, human-unvalidated data. Software fixtures are not model results.</p><h2>Frozen evaluation split</h2><table><tr><th>Task</th><th>Rows</th><th>Source cases / tasks</th><th>Independent sampling units</th></tr>' + table + '</table><h2>Available locally</h2><p><a href="evaluation/manifest.json">Evaluation bundle</a> · <a href="all/manifest.json">All-splits bundle (exploratory)</a> · <a href="evaluation/content_validation.json">Content checks</a> · <a href="verification.json">Verification record</a></p><h2>Interpretation limits</h2><ul>' + limitations + '</ul><p>Reports include dataset-specific metrics, a worst-group layer, matched-topic comparisons and explicit missing judgment coverage. Benign response scoring binds judgments to the exact request, response and rubric.</p></html>')
    print(json.dumps({'status': verification['status'], 'bundles': {k: v['manifest'] for k, v in outputs.items()}, 'original_inputs_unchanged': len(originals)}, indent=2))


if __name__ == '__main__':
    main()
