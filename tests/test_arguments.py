import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arguments import (
    OptimizationParams,
    _parse_cfg_namespace,
    evaluation_output_name,
    validate_aux_loss_options,
)


class ConfigParserTests(unittest.TestCase):
    def test_aux_loss_defaults_preserve_legacy_behavior(self):
        parser = ArgumentParser()
        parameters = OptimizationParams(parser)
        resolved = parameters.extract(parser.parse_args([]))
        self.assertEqual(resolved.aux_loss_version, "legacy")
        self.assertEqual(resolved.edge_filter_kernel, 5)
        self.assertEqual(resolved.edge_filter_sigma, 1.0)

    def test_filtered_edge_options_parse_and_validate(self):
        parser = ArgumentParser()
        parameters = OptimizationParams(parser)
        resolved = parameters.extract(
            parser.parse_args(
                [
                    "--aux_loss_version",
                    "filtered_edge",
                    "--edge_filter_kernel",
                    "7",
                    "--edge_filter_sigma",
                    "1.5",
                    "--lambda_edge",
                    "0.001",
                ]
            )
        )
        self.assertEqual(resolved.aux_loss_version, "filtered_edge")
        self.assertEqual(resolved.edge_filter_kernel, 7)
        self.assertEqual(resolved.edge_filter_sigma, 1.5)

    def test_invalid_aux_loss_options_are_rejected_before_training(self):
        invalid = (
            ({"aux_loss_version": "unknown"}, "aux_loss_version"),
            ({"edge_filter_kernel": 4}, "positive odd"),
            ({"edge_filter_sigma": 0.0}, "positive"),
            (
                {"aux_loss_version": "filtered_edge", "lambda_thermal": 0.1},
                "requires lambda_thermal",
            ),
            (
                {"aux_loss_version": "filtered_edge", "lambda_smooth": 0.1},
                "requires lambda_thermal",
            ),
        )
        for options, message in invalid:
            with self.subTest(options=options):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    validate_aux_loss_options(options)

    def test_legacy_cfg_without_new_fields_remains_loadable(self):
        parsed = _parse_cfg_namespace(
            "Namespace(source_path='scene', lambda_edge=0.01)"
        )
        validate_aux_loss_options(parsed)
        self.assertFalse(hasattr(parsed, "aux_loss_version"))

    def test_evaluation_output_name_tracks_manifest_partition(self):
        self.assertEqual(evaluation_output_name("dataset.json", "val"), "val")
        self.assertEqual(evaluation_output_name("dataset.json", "test"), "test")
        self.assertEqual(evaluation_output_name("", "val"), "test")
        with self.assertRaisesRegex(ValueError, "evaluation_partition"):
            evaluation_output_name("dataset.json", "validation")

    def test_namespace_literals_are_parsed(self):
        parsed = _parse_cfg_namespace(
            "Namespace(source_path='scene', eval=True, resolution=-1, values=[1, 2])"
        )
        self.assertEqual(parsed.source_path, "scene")
        self.assertTrue(parsed.eval)
        self.assertEqual(parsed.resolution, -1)
        self.assertEqual(parsed.values, [1, 2])

    def test_executable_expression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a literal"):
            _parse_cfg_namespace("Namespace(value=__import__('os').getcwd())")

    def test_non_namespace_expression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Namespace"):
            _parse_cfg_namespace("{'source_path': 'scene'}")


if __name__ == "__main__":
    unittest.main()
