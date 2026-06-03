from __future__ import annotations

import unittest

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from handwriting_ai.data.transforms import deltas_to_points, points_to_deltas, resample_strokes, smooth_strokes


class TransformTests(unittest.TestCase):
    def test_delta_roundtrip_preserves_geometry_and_flags(self) -> None:
        points = np.asarray(
            [
                [10.0, 10.0, 0.0],
                [12.0, 11.0, 0.0],
                [15.0, 13.0, 1.0],
                [20.0, 8.0, 0.0],
                [22.0, 8.0, 1.0],
            ],
            dtype=np.float32,
        )
        deltas = points_to_deltas(points)
        restored = deltas_to_points(deltas)
        np.testing.assert_allclose(restored[:, :2], points[:, :2] - points[0, :2], atol=1e-5)
        np.testing.assert_array_equal(restored[:, 2], points[:, 2])

    def test_resample_and_smooth_keep_stroke_end_flags(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 1.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        smoothed = smooth_strokes(points, window=3)
        resampled = resample_strokes(smoothed, spacing=2.5)
        self.assertGreater(len(resampled), len(points))
        self.assertEqual(int(resampled[-1, 2]), 1)
        self.assertEqual(int(resampled[:, 2].sum()), 2)


if __name__ == "__main__":
    unittest.main()
