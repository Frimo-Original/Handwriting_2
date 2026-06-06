from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from handwriting_ai.checkpoint import save_checkpoint
from handwriting_ai.generator_checkpoint import CURRENT_GENERATOR_TRAINING_VERSION
import main_generate
import main_evaluate
import main_training


class EntrypointTests(unittest.TestCase):
    def test_training_parser_defaults_to_recognizer(self) -> None:
        args = main_training.build_parser().parse_args([])
        self.assertEqual(args.profile, "gtx1660")
        self.assertEqual(args.stage, "recognizer")

    def test_generate_output_paths(self) -> None:
        out_json, out_png = main_generate.resolve_output_paths(Path("outputs"), "sample", "тест")
        self.assertEqual(out_json, Path("outputs/sample.json"))
        self.assertEqual(out_png, Path("outputs/sample.png"))

    def test_evaluate_parser_defaults_to_all(self) -> None:
        args = main_evaluate.build_parser().parse_args([])
        self.assertEqual(args.profile, "gtx1660")
        self.assertEqual(args.stage, "all")
        self.assertEqual(args.generator_selection, "best")

    def test_generator_selection_parser(self) -> None:
        eval_args = main_evaluate.build_parser().parse_args(["--generator-selection", "train_best"])
        self.assertEqual(eval_args.generator_selection, "train_best")

        generate_args = main_generate.build_parser().parse_args(["Тест", "--generator-selection", "last"])
        self.assertEqual(generate_args.generator_selection, "last")

    def test_generate_point_length_aliases(self) -> None:
        generate_args = main_generate.build_parser().parse_args(
            ["Тест", "--point-length", "123", "--max-point-length", "456"]
        )
        self.assertEqual(generate_args.latent_length, 123)
        self.assertEqual(generate_args.max_latent_length, 456)

        legacy_args = main_generate.build_parser().parse_args(
            ["Тест", "--latent-length", "12", "--max-latent-length", "34"]
        )
        self.assertEqual(legacy_args.latent_length, 12)
        self.assertEqual(legacy_args.max_latent_length, 34)

    def test_current_generator_checkpoint_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generator.pt"
            save_checkpoint(path, model_type="latent_flow")
            self.assertFalse(main_training.is_current_generator_checkpoint(path))
            save_checkpoint(
                path,
                model_type="latent_regressor",
                latent_normalization={"mean": [0.0], "std": [1.0]},
            )
            self.assertFalse(main_training.is_current_generator_checkpoint(path))
            save_checkpoint(
                path,
                model_type="latent_regressor",
                latent_normalization={"mean": [0.0], "std": [1.0]},
                generator_training_version=CURRENT_GENERATOR_TRAINING_VERSION,
            )
            self.assertFalse(main_training.is_current_generator_checkpoint(path))
            save_checkpoint(
                path,
                model_type="trajectory_generator",
                generator_training_version=CURRENT_GENERATOR_TRAINING_VERSION,
            )
            self.assertFalse(main_training.is_current_generator_checkpoint(path))
            save_checkpoint(
                path,
                model_type="content_aligned_latent_flow",
                generator_training_version=CURRENT_GENERATOR_TRAINING_VERSION,
            )
            self.assertTrue(main_training.is_current_generator_checkpoint(path))

    def test_generate_rejects_old_latent_regressor_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generator.pt"
            save_checkpoint(
                path,
                model_type="latent_regressor",
                latent_normalization={"mean": [0.0], "std": [1.0]},
            )

            with self.assertRaisesRegex(ValueError, "outdated"):
                main_generate.validate_generator_checkpoint(path, allow_legacy_flow=False)

            save_checkpoint(
                path,
                model_type="latent_regressor",
                latent_normalization={"mean": [0.0], "std": [1.0]},
                generator_training_version=CURRENT_GENERATOR_TRAINING_VERSION,
            )
            with self.assertRaisesRegex(ValueError, "outdated"):
                main_generate.validate_generator_checkpoint(path, allow_legacy_flow=False)

            save_checkpoint(
                path,
                model_type="trajectory_generator",
                generator_training_version=CURRENT_GENERATOR_TRAINING_VERSION,
            )
            with self.assertRaisesRegex(ValueError, "outdated"):
                main_generate.validate_generator_checkpoint(path, allow_legacy_flow=False)

            save_checkpoint(
                path,
                model_type="content_aligned_latent_flow",
                generator_training_version=CURRENT_GENERATOR_TRAINING_VERSION,
            )
            main_generate.validate_generator_checkpoint(path, allow_legacy_flow=False)


if __name__ == "__main__":
    unittest.main()
