import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arguments import _parse_cfg_namespace, evaluation_output_name


class ConfigParserTests(unittest.TestCase):
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
