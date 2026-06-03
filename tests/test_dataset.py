from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from handwriting_ai.config import DataConfig, HardwareConfig
from handwriting_ai.data.dataset import build_dataloaders, build_datasets, collate_batch


def _write_npz(path: Path) -> None:
    points = np.asarray(
        [
            np.asarray([[0, 0, 0], [4, 0, 0], [8, 1, 1], [12, 1, 0], [14, 2, 1]], dtype=np.float32),
            np.asarray([[0, 1, 0], [2, 2, 0], [4, 3, 1], [8, 3, 0], [9, 3, 1]], dtype=np.float32),
            np.asarray([[1, 0, 0], [2, 0, 0], [3, 1, 1], [5, 1, 0], [6, 2, 1]], dtype=np.float32),
        ],
        dtype=object,
    )
    texts = np.asarray(
        [
            np.asarray([1, 2, 3], dtype=np.int64),
            np.asarray([4, 5, 6], dtype=np.int64),
            np.asarray([7, 8, 9], dtype=np.int64),
        ],
        dtype=object,
    )
    np.savez_compressed(path, points=points, text_indices=texts)


class DatasetTests(unittest.TestCase):
    def test_build_dataset_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "tiny.npz"
            _write_npz(dataset_path)
            data = DataConfig(
                dataset_npz=dataset_path,
                val_fraction=0.34,
                split_seed=1,
                resample_spacing=1.0,
                smooth_window=1,
                max_points=128,
                max_text_len=8,
                batch_size=2,
            )
            train, val, stats = build_datasets(data)
            self.assertGreater(len(train), 0)
            self.assertGreater(len(val), 0)
            self.assertEqual(stats.mean.shape, (2,))
            batch = collate_batch([train[0], train[min(1, len(train) - 1)]])
            self.assertEqual(batch.points.ndim, 3)
            self.assertEqual(batch.text.ndim, 2)
            self.assertTrue(batch.point_mask.any())

    def test_dataloader_builds_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "tiny.npz"
            _write_npz(dataset_path)
            data = DataConfig(
                dataset_npz=dataset_path,
                val_fraction=0.34,
                split_seed=2,
                resample_spacing=None,
                smooth_window=1,
                max_points=128,
                max_text_len=8,
                batch_size=2,
            )
            hardware = HardwareConfig(
                device="cpu",
                num_workers=0,
                pin_memory=False,
                amp=False,
            )
            train_loader, _, _ = build_dataloaders(data, hardware)
            batch = next(iter(train_loader))
            self.assertEqual(batch.points.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()
