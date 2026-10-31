"""Collector subprocesses must retain the selected virtual environment."""

import json
import os
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lm_eval.fair_uk import all_metrics


class CollectorInterpreterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.collector = self.root / "collector"
        self.collector.mkdir()
        self.analysis = self.root / "analysis"
        self.analysis.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.pair = {
            "task": "warbias_triplets_uk",
            "before_run": "before",
            "after_run": "after",
            "dataset_bundle": "manifest.json",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_real_venv_symlink_keeps_venv_only_distribution_metadata(self):
        environment = self.root / "collector-venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        python = environment / "bin/python"
        self.assertTrue(python.is_symlink())
        self.assertNotEqual(python, python.resolve())
        probe = subprocess.run(  # noqa: S603 - Only this test-created local interpreter runs.
            [
                str(python),
                "-c",
                "import sysconfig;print(sysconfig.get_path('purelib'))",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        metadata = Path(probe.stdout.strip()) / "fair_uk_runtime_fixture-0.1.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: fair-uk-runtime-fixture\nVersion: 0.1\n"
        )
        # The underlying interpreter cannot see this distribution; no package install is performed.
        absent = subprocess.run(  # noqa: S603 - Only the qualified local base interpreter runs.
            [
                str(python.resolve()),
                "-c",
                "import importlib.metadata;print(importlib.metadata.version('fair-uk-runtime-fixture'))",
            ],
            cwd=self.collector,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(absent.returncode, 0)
        self.assertIn("PackageNotFoundError", absent.stderr)
        script = self.analysis / "collector_export.py"
        script.write_text(
            "import argparse,importlib.metadata,json,sys\n"
            "from pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()\n"
            "request=json.loads(Path(a.request).read_text())\n"
            "value={'kind':'collector_validated_saved_pair','task':request['task'],'runtime':{'executable':sys.executable,'prefix':sys.prefix,'fixture_version':importlib.metadata.version('fair-uk-runtime-fixture')}}\n"
            "Path(a.output).write_text(json.dumps(value))\n"
        )
        with patch.object(
            all_metrics, "__file__", str(self.analysis / "all_metrics.py")
        ):
            exported, reference = all_metrics.export_pair(
                self.pair,
                {"python": str(python), "evaluator_root": str(self.collector)},
                self.root,
                self.output,
            )
        self.assertEqual(
            exported["runtime"],
            {
                "executable": str(python),
                "prefix": str(environment),
                "fixture_version": "0.1",
            },
        )
        self.assertEqual(
            reference["sha256"], all_metrics.digest(self.output / "validated_pair.json")
        )

    def test_only_python_argument_preserves_lexical_path_and_expands_home(self):
        real = self.root / "real-data"
        real.mkdir()
        (self.root / "linked-data").symlink_to(real, target_is_directory=True)
        (self.root / "linked-collector").symlink_to(
            self.collector, target_is_directory=True
        )
        pair = {
            **self.pair,
            "before_run": "linked-data/before",
            "after_run": "linked-data/after",
            "dataset_bundle": "linked-data/bundle.json",
        }
        arguments = (
            "linked-data/../linked-data/venv/bin/python",
            "~/fair-uk-test-collector-venv/bin/python",
        )
        for index, selected in enumerate(arguments):
            output = self.root / ("binding-" + str(index))
            output.mkdir()

            def run(command, selected=selected, output=output, **kwargs):
                expected = os.path.abspath(self.root / Path(selected).expanduser())
                self.assertEqual(command[0], expected)
                self.assertEqual(
                    command[2:4], ["--request", str(output / "export_request.json")]
                )
                request = json.loads((output / "export_request.json").read_text())
                self.assertEqual(request["evaluator_root"], str(self.collector))
                self.assertEqual(request["before_run"], str(real / "before"))
                self.assertEqual(request["after_run"], str(real / "after"))
                self.assertEqual(request["dataset_bundle"], str(real / "bundle.json"))
                self.assertEqual(kwargs["cwd"], str(self.collector))
                self.assertEqual(kwargs["env"]["PYTHONPATH"], str(self.collector))
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertFalse(kwargs["check"])
                (output / "validated_pair.json").write_text(
                    json.dumps(
                        {"kind": "collector_validated_saved_pair", "task": pair["task"]}
                    )
                )
                return SimpleNamespace(returncode=0)

            with (
                self.subTest(python=selected),
                patch.object(all_metrics.subprocess, "run", side_effect=run),
            ):
                all_metrics.export_pair(
                    pair,
                    {"python": selected, "evaluator_root": "linked-collector"},
                    self.root,
                    output,
                )


if __name__ == "__main__":
    unittest.main()
