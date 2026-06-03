from __future__ import annotations

import unittest
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import main_generate
import main_evaluate
import main_training


class EntrypointTests(unittest.TestCase):
    def test_training_parser_defaults_to_autoencoder(self) -> None:
        args = main_training.build_parser().parse_args([])
        self.assertEqual(args.profile, "gtx1660")
        self.assertEqual(args.stage, "autoencoder")

    def test_generate_output_paths(self) -> None:
        out_json, out_png = main_generate.resolve_output_paths(Path("outputs"), "sample", "тест")
        self.assertEqual(out_json, Path("outputs/sample.json"))
        self.assertEqual(out_png, Path("outputs/sample.png"))

    def test_evaluate_parser_defaults_to_all(self) -> None:
        args = main_evaluate.build_parser().parse_args([])
        self.assertEqual(args.profile, "gtx1660")
        self.assertEqual(args.stage, "all")


if __name__ == "__main__":
    unittest.main()
