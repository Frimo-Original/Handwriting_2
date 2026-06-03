from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from handwriting_ai.checkpoint import save_checkpoint
from handwriting_ai.data.codec import VOCAB_TOKENS
from handwriting_ai.inference import generate_points
from handwriting_ai.models import InkAutoencoder, LatentFlowTransformer, TrajectoryRecognizer
from handwriting_ai.training.losses import autoencoder_loss, flow_matching_loss, recognizer_ctc_loss


class ModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
