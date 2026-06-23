from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from handwriting_ai.checkpoint import save_checkpoint
from handwriting_ai.alignment import ctc_forced_alignment, fit_durations
from handwriting_ai.data.codec import VOCAB_TOKENS
from handwriting_ai.latent_stats import LatentNormalizationStats
from handwriting_ai.inference import generate_points
from handwriting_ai.models import (
    AlignedLatentFlow,
    ContentAlignedLatentFlow,
    InkAutoencoder,
    LatentFlowTransformer,
    LatentRegressorTransformer,
    LocalInkAutoencoder,
    TrajectoryGenerator,
    TrajectoryRecognizer,
)
from handwriting_ai.models.aligned_flow import maximum_path
from handwriting_ai.training.losses import (
    aligned_flow_matching_loss,
    autoencoder_loss,
    content_aligned_flow_loss,
    flow_matching_loss,
    latent_regression_loss,
    recognizer_ctc_loss,
    trajectory_generator_loss,
)


class ModelTests(unittest.TestCase):
    def test_ctc_forced_alignment_handles_blanks_and_repeats(self) -> None:
        blank = 5
        targets = torch.tensor([1, 1, 2])
        path = [blank, 1, blank, 1, blank, 2, blank]
        logits = torch.full((len(path), 6), -8.0)
        for frame, token in enumerate(path):
            logits[frame, token] = 8.0
        alignment = ctc_forced_alignment(
            logits.log_softmax(dim=-1),
            targets,
            blank_id=blank,
        )
        self.assertEqual(int(alignment.durations.sum()), len(path))
        self.assertEqual(alignment.durations.numel(), 3)
        self.assertTrue(bool(torch.all(alignment.token_frames[1:] >= alignment.token_frames[:-1])))
        fitted = fit_durations(alignment.durations, target_length=10)
        self.assertEqual(int(fitted.sum()), 10)
        self.assertTrue(bool(torch.all(fitted >= 1)))

    def test_local_autoencoder_shapes_and_backward(self) -> None:
        model = LocalInkAutoencoder(
            hidden_dim=32,
            latent_dim=16,
            downsample_factor=4,
            bottleneck_layers=2,
            local_kernel_size=5,
            dropout=0.0,
        )
        points = torch.randn(2, 65, 5)
        points[..., 2] = (points[..., 2] > 0).float()
        lengths = torch.tensor([65, 48], dtype=torch.long)
        mask = torch.arange(65).unsqueeze(0) < lengths.unsqueeze(1)
        output = model(points, lengths)
        self.assertEqual(output.reconstruction.shape, points.shape)
        self.assertEqual(output.latent.shape[1], 17)
        loss, metrics = autoencoder_loss(
            output,
            points,
            mask,
            kl_weight=0.0,
            pen_weight=0.3,
            curvature_weight=0.01,
            render_weight=0.0,
        )
        loss.backward()
        self.assertIn("jump", metrics)

    def test_maximum_path_is_monotonic_and_complete(self) -> None:
        scores = torch.tensor(
            [
                [
                    [5.0, 0.0, 0.0],
                    [4.0, 1.0, 0.0],
                    [0.0, 5.0, 0.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 0.0, 5.0],
                    [0.0, 0.0, 4.0],
                ]
            ]
        )
        path = maximum_path(
            scores,
            latent_lengths=torch.tensor([6]),
            text_lengths=torch.tensor([3]),
        )
        self.assertEqual(path.shape, scores.shape)
        self.assertTrue(torch.equal(path.sum(dim=2), torch.ones(1, 6)))
        self.assertTrue(torch.equal(path.sum(dim=1), torch.tensor([[2.0, 2.0, 2.0]])))
        token_indices = path[0].argmax(dim=1)
        self.assertTrue(bool(torch.all(token_indices[1:] >= token_indices[:-1])))

    def test_autoencoder_shapes_and_backward(self) -> None:
        model = InkAutoencoder(
            hidden_dim=32,
            latent_dim=16,
            downsample_factor=4,
            bottleneck_layers=1,
            n_heads=2,
            dropout=0.0,
        )
        points = torch.randn(2, 64, 3)
        points[..., 2] = (points[..., 2] > 0).float()
        lengths = torch.tensor([64, 48], dtype=torch.long)
        mask = torch.arange(64).unsqueeze(0) < lengths.unsqueeze(1)
        output = model(points, lengths)
        self.assertEqual(output.reconstruction.shape, points.shape)
        loss, metrics = autoencoder_loss(
            output,
            points,
            mask,
            kl_weight=0.001,
            pen_weight=0.3,
            curvature_weight=0.01,
            render_weight=0.0,
        )
        loss.backward()
        self.assertIn("xy", metrics)

    def test_flow_shapes_and_backward(self) -> None:
        flow = LatentFlowTransformer(
            latent_dim=16,
            hidden_dim=32,
            text_dim=32,
            layers=2,
            n_heads=4,
            dropout=0.0,
        )
        latents = torch.randn(2, 12, 16)
        latent_lengths = torch.tensor([12, 9], dtype=torch.long)
        latent_mask = torch.arange(12).unsqueeze(0) < latent_lengths.unsqueeze(1)
        text = torch.tensor([[10, 11, 12, 89], [13, 14, 89, 90]], dtype=torch.long)
        text_mask = text != 90
        loss, metrics = flow_matching_loss(
            flow,
            latents,
            latent_mask,
            latent_lengths,
            text,
            text_mask,
            length_loss_weight=0.1,
        )
        loss.backward()
        self.assertIn("velocity", metrics)

    def test_aligned_flow_shapes_alignment_and_backward(self) -> None:
        model = AlignedLatentFlow(
            latent_dim=8,
            hidden_dim=32,
            text_dim=32,
            layers=2,
            n_heads=4,
            dropout=0.0,
        )
        latents = torch.randn(2, 12, 8)
        latent_lengths = torch.tensor([12, 9], dtype=torch.long)
        latent_mask = torch.arange(12).unsqueeze(0) < latent_lengths.unsqueeze(1)
        text = torch.tensor([[10, 11, 12, 89], [13, 14, 89, 90]], dtype=torch.long)
        text_mask = text != 90
        text_lengths = text_mask.sum(dim=1)
        loss, metrics = aligned_flow_matching_loss(
            model,
            latents,
            latent_mask,
            latent_lengths,
            text,
            text_mask,
            text_lengths,
            alignment_loss_weight=0.25,
            duration_loss_weight=0.2,
        )
        loss.backward()
        self.assertIn("alignment", metrics)
        self.assertIn("duration", metrics)
        sampled, sampled_mask = model.sample(
            text,
            text_mask,
            latent_length=10,
            steps=2,
            temperature=0.0,
        )
        self.assertEqual(sampled.shape, (2, 10, 8))
        self.assertEqual(sampled_mask.shape, (2, 10))

    def test_content_aligned_flow_endpoint_losses_and_backward(self) -> None:
        model = ContentAlignedLatentFlow(
            latent_dim=8,
            hidden_dim=32,
            text_dim=32,
            layers=2,
            n_heads=4,
            dropout=0.0,
        )
        autoencoder = LocalInkAutoencoder(
            hidden_dim=16,
            latent_dim=8,
            downsample_factor=4,
            bottleneck_layers=1,
            local_kernel_size=5,
            dropout=0.0,
        )
        recognizer = TrajectoryRecognizer(hidden_dim=16, layers=1, dropout=0.0)
        for module in (autoencoder, recognizer):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        latent_lengths = torch.tensor([8, 7])
        latent_mask = torch.arange(8).unsqueeze(0) < latent_lengths.unsqueeze(1)
        latents = torch.randn(2, 8, 8) * latent_mask.unsqueeze(-1)
        text = torch.tensor([[10, 11, 89, 90], [12, 13, 89, 90]])
        text_mask = text != 90
        text_lengths = text_mask.sum(dim=1)
        durations = torch.tensor([[4, 4, 0, 0], [3, 4, 0, 0]])
        points = torch.randn(2, 32, 5)
        points[..., 2] = (points[..., 2] > 0).float()
        point_lengths = latent_lengths * 4
        point_mask = torch.arange(32).unsqueeze(0) < point_lengths.unsqueeze(1)
        loss, metrics = content_aligned_flow_loss(
            model,
            latents,
            latent_mask,
            latent_lengths,
            durations,
            text,
            text_mask,
            text_lengths,
            points,
            point_mask,
            point_lengths,
            autoencoder=autoencoder,
            latent_normalization=LatentNormalizationStats(
                mean=[0.0] * 8,
                std=[1.0] * 8,
            ),
            recognizer=recognizer,
            duration_loss_weight=0.2,
            endpoint_latent_weight=0.1,
            decoded_xy_weight=0.1,
            decoded_path_weight=0.1,
            decoded_pen_weight=0.1,
            decoded_pen_pos_weight=2.0,
            decoded_curvature_weight=0.1,
            semantic_weight=0.01,
            blank_id=90,
        )
        loss.backward()
        self.assertIn("semantic", metrics)
        sampled, sampled_mask = model.sample(
            text,
            text_mask,
            latent_length=8,
            steps=2,
        )
        self.assertEqual(sampled.shape, (2, 8, 8))
        self.assertEqual(sampled_mask.shape, (2, 8))

    def test_latent_regressor_shapes_and_backward(self) -> None:
        model = LatentRegressorTransformer(
            latent_dim=16,
            hidden_dim=32,
            text_dim=32,
            layers=2,
            n_heads=4,
            dropout=0.0,
        )
        latents = torch.randn(2, 12, 16)
        latent_lengths = torch.tensor([12, 9], dtype=torch.long)
        latent_mask = torch.arange(12).unsqueeze(0) < latent_lengths.unsqueeze(1)
        text = torch.tensor([[10, 11, 12, 89], [13, 14, 89, 90]], dtype=torch.long)
        text_mask = text != 90
        autoencoder = InkAutoencoder(
            hidden_dim=32,
            latent_dim=16,
            downsample_factor=4,
            bottleneck_layers=0,
            n_heads=1,
            dropout=0.0,
        )
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        target_points = torch.randn(2, 48, 3)
        target_points[..., 2] = (target_points[..., 2] > 0).float()
        point_lengths = latent_lengths * autoencoder.downsample_factor
        point_mask = torch.arange(48).unsqueeze(0) < point_lengths.unsqueeze(1)
        loss, metrics = latent_regression_loss(
            model,
            latents,
            latent_mask,
            latent_lengths,
            text,
            text_mask,
            length_loss_weight=0.1,
            autoencoder=autoencoder,
            latent_normalization=LatentNormalizationStats(mean=[0.0] * 16, std=[1.0] * 16),
            target_points=target_points,
            point_mask=point_mask,
            latent_loss_weight=0.25,
            latent_std_weight=0.1,
            decoded_xy_weight=1.0,
            decoded_path_weight=0.5,
            decoded_pen_weight=0.5,
            decoded_pen_pos_weight=4.0,
            decoded_curvature_weight=0.1,
        )
        loss.backward()
        self.assertIn("latent", metrics)
        self.assertIn("decoded_xy", metrics)
        self.assertIn("decoded_path", metrics)
        self.assertIn("latent_std", metrics)
        sampled, sampled_mask = model.sample(text, text_mask, latent_length=7)
        self.assertEqual(sampled.shape, (2, 7, 16))
        self.assertEqual(sampled_mask.shape, (2, 7))

    def test_recognizer_shapes_and_backward(self) -> None:
        recognizer = TrajectoryRecognizer(hidden_dim=32, layers=1, dropout=0.0)
        points = torch.randn(2, 96, 3)
        lengths = torch.tensor([96, 80], dtype=torch.long)
        text = torch.tensor([[10, 11, 89], [12, 13, 89]], dtype=torch.long)
        text_lengths = torch.tensor([3, 3], dtype=torch.long)
        log_probs, output_lengths = recognizer(points, lengths)
        loss = recognizer_ctc_loss(
            log_probs,
            output_lengths,
            text,
            text_lengths,
            blank_id=90,
        )
        loss.backward()
        self.assertEqual(log_probs.shape[1], 2)

    def test_trajectory_generator_shapes_and_backward(self) -> None:
        model = TrajectoryGenerator(
            hidden_dim=32,
            text_dim=32,
            layers=2,
            n_heads=4,
            dropout=0.0,
        )
        points = torch.randn(2, 64, 5)
        points[..., 2] = (points[..., 2] > 0).float()
        point_lengths = torch.tensor([64, 48], dtype=torch.long)
        point_mask = torch.arange(64).unsqueeze(0) < point_lengths.unsqueeze(1)
        text = torch.tensor([[10, 11, 12, 89], [13, 14, 89, 90]], dtype=torch.long)
        text_mask = text != 90
        recognizer = TrajectoryRecognizer(hidden_dim=32, layers=1, dropout=0.0)
        for parameter in recognizer.parameters():
            parameter.requires_grad_(False)
        text_lengths = torch.tensor([3, 2], dtype=torch.long)
        loss, metrics = trajectory_generator_loss(
            model,
            points,
            point_mask,
            point_lengths,
            text,
            text_mask,
            length_loss_weight=0.1,
            xy_weight=1.0,
            path_weight=0.5,
            pen_weight=0.5,
            pen_pos_weight=4.0,
            curvature_weight=0.1,
            jump_weight=1.0,
            recognizer=recognizer,
            text_lengths=text_lengths,
            semantic_weight=0.1,
            blank_id=90,
        )
        loss.backward()
        self.assertIn("path", metrics)
        self.assertIn("jump", metrics)
        self.assertIn("semantic", metrics)
        sampled, sampled_mask = model.sample(text, text_mask, point_length=32)
        self.assertEqual(sampled.shape, (2, 32, 5))
        self.assertEqual(sampled_mask.shape, (2, 32))

    def test_inference_with_temporary_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae = InkAutoencoder(
                hidden_dim=16,
                latent_dim=8,
                downsample_factor=4,
                bottleneck_layers=0,
                n_heads=1,
                dropout=0.0,
            )
            flow = LatentFlowTransformer(
                latent_dim=8,
                hidden_dim=16,
                text_dim=16,
                layers=1,
                n_heads=1,
                dropout=0.0,
            )
            ae_path = tmp_path / "ae.pt"
            flow_path = tmp_path / "flow.pt"
            save_checkpoint(
                ae_path,
                model_state=ae.state_dict(),
                model_kwargs={
                    "input_dim": 3,
                    "hidden_dim": 16,
                    "latent_dim": 8,
                    "downsample_factor": 4,
                    "bottleneck_layers": 0,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                vocab_tokens=VOCAB_TOKENS,
            )
            save_checkpoint(
                flow_path,
                model_state=flow.state_dict(),
                model_kwargs={
                    "latent_dim": 8,
                    "hidden_dim": 16,
                    "text_dim": 16,
                    "layers": 1,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                vocab_tokens=VOCAB_TOKENS,
            )
            points = generate_points(
                text="тест",
                autoencoder_checkpoint=ae_path,
                generator_checkpoint=flow_path,
                device=torch.device("cpu"),
                steps=1,
                latent_length=4,
            )
            self.assertEqual(points.shape, (16, 3))
            self.assertEqual(int(points[-1, 2]), 1)

    def test_inference_with_trajectory_generator_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generator_path = tmp_path / "trajectory.pt"
            generator = TrajectoryGenerator(
                hidden_dim=16,
                text_dim=16,
                layers=1,
                n_heads=1,
                dropout=0.0,
            )
            save_checkpoint(
                generator_path,
                model_type="trajectory_generator",
                model_state=generator.state_dict(),
                model_kwargs={
                    "hidden_dim": 16,
                    "text_dim": 16,
                    "layers": 1,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                vocab_tokens=VOCAB_TOKENS,
            )
            points = generate_points(
                text="тест",
                autoencoder_checkpoint=tmp_path / "missing_ae.pt",
                generator_checkpoint=generator_path,
                device=torch.device("cpu"),
                latent_length=24,
                temperature=0.0,
            )
            self.assertEqual(points.shape, (24, 3))
            self.assertEqual(int(points[-1, 2]), 1)

    def test_inference_with_aligned_flow_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae_path = tmp_path / "ae.pt"
            generator_path = tmp_path / "aligned.pt"
            ae = InkAutoencoder(
                hidden_dim=16,
                latent_dim=8,
                downsample_factor=4,
                bottleneck_layers=0,
                n_heads=1,
                dropout=0.0,
            )
            generator = AlignedLatentFlow(
                latent_dim=8,
                hidden_dim=16,
                text_dim=16,
                layers=1,
                n_heads=1,
                dropout=0.0,
            )
            save_checkpoint(
                ae_path,
                model_state=ae.state_dict(),
                model_kwargs={
                    "input_dim": 3,
                    "hidden_dim": 16,
                    "latent_dim": 8,
                    "downsample_factor": 4,
                    "bottleneck_layers": 0,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                vocab_tokens=VOCAB_TOKENS,
            )
            save_checkpoint(
                generator_path,
                model_type="aligned_latent_flow",
                model_state=generator.state_dict(),
                model_kwargs={
                    "latent_dim": 8,
                    "hidden_dim": 16,
                    "text_dim": 16,
                    "layers": 1,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                latent_normalization={"mean": [0.0] * 8, "std": [1.0] * 8},
                normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                vocab_tokens=VOCAB_TOKENS,
            )
            points = generate_points(
                text="тест",
                autoencoder_checkpoint=ae_path,
                generator_checkpoint=generator_path,
                device=torch.device("cpu"),
                steps=2,
                latent_length=8,
                temperature=0.0,
            )
            self.assertEqual(points.shape, (32, 3))
            self.assertEqual(int(points[-1, 2]), 1)

    def test_inference_with_content_aligned_flow_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae_path = tmp_path / "ae.pt"
            generator_path = tmp_path / "content_flow.pt"
            ae = LocalInkAutoencoder(
                hidden_dim=16,
                latent_dim=8,
                downsample_factor=4,
                bottleneck_layers=1,
                local_kernel_size=5,
                dropout=0.0,
            )
            generator = ContentAlignedLatentFlow(
                latent_dim=8,
                hidden_dim=16,
                text_dim=16,
                layers=1,
                n_heads=1,
                dropout=0.0,
            )
            save_checkpoint(
                ae_path,
                model_type="local_content_autoencoder",
                model_state=ae.state_dict(),
                model_kwargs={
                    "input_dim": 5,
                    "hidden_dim": 16,
                    "latent_dim": 8,
                    "downsample_factor": 4,
                    "bottleneck_layers": 1,
                    "local_kernel_size": 5,
                    "dropout": 0.0,
                },
                normalization={
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                    "jump_mean": [0.0, 0.0],
                    "jump_std": [1.0, 1.0],
                },
                vocab_tokens=VOCAB_TOKENS,
            )
            save_checkpoint(
                generator_path,
                model_type="content_aligned_latent_flow",
                model_state=generator.state_dict(),
                model_kwargs={
                    "latent_dim": 8,
                    "hidden_dim": 16,
                    "text_dim": 16,
                    "layers": 1,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                latent_normalization={
                    "mean": [0.0] * 8,
                    "std": [1.0] * 8,
                },
                normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                vocab_tokens=VOCAB_TOKENS,
            )
            points = generate_points(
                text="тест",
                autoencoder_checkpoint=ae_path,
                generator_checkpoint=generator_path,
                device=torch.device("cpu"),
                steps=2,
                latent_length=8,
                temperature=1.0,
            )
            self.assertEqual(points.shape, (32, 3))
            self.assertEqual(int(points[-1, 2]), 1)

    def test_inference_with_latent_regressor_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae = InkAutoencoder(
                hidden_dim=16,
                latent_dim=8,
                downsample_factor=4,
                bottleneck_layers=0,
                n_heads=1,
                dropout=0.0,
            )
            generator = LatentRegressorTransformer(
                latent_dim=8,
                hidden_dim=16,
                text_dim=16,
                layers=1,
                n_heads=1,
                dropout=0.0,
            )
            ae_path = tmp_path / "ae.pt"
            generator_path = tmp_path / "regressor.pt"
            save_checkpoint(
                ae_path,
                model_state=ae.state_dict(),
                model_kwargs={
                    "input_dim": 3,
                    "hidden_dim": 16,
                    "latent_dim": 8,
                    "downsample_factor": 4,
                    "bottleneck_layers": 0,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                vocab_tokens=VOCAB_TOKENS,
            )
            save_checkpoint(
                generator_path,
                model_type="latent_regressor",
                model_state=generator.state_dict(),
                model_kwargs={
                    "latent_dim": 8,
                    "hidden_dim": 16,
                    "text_dim": 16,
                    "layers": 1,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                latent_normalization={"mean": [0.0] * 8, "std": [1.0] * 8},
                normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                vocab_tokens=VOCAB_TOKENS,
            )
            points = generate_points(
                text="тест",
                autoencoder_checkpoint=ae_path,
                generator_checkpoint=generator_path,
                device=torch.device("cpu"),
                latent_length=4,
                temperature=0.0,
            )
            self.assertEqual(points.shape, (16, 3))
            self.assertEqual(int(points[-1, 2]), 1)


if __name__ == "__main__":
    unittest.main()
