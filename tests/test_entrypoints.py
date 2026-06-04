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
            main_generate.validate_generator_checkpoint(path, allow_legacy_flow=False)


if __name__ == "__main__":
    unittest.main()
