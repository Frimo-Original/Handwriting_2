import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

import text_codec

DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1]


def convert(json_path, txt_path, out_npz):
    with open(json_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    pts = np.asarray(trajectory, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"{json_path} must contain points shaped as (T, 3)")
    pts[:, 2] = (pts[:, 2] == 1).astype(np.float32)
    pts[-1, 2] = 1.0

    unsupported = text_codec.unsupported_tokens(text, normalize=True)
    if unsupported:
        printable = ", ".join(repr(token) for token in unsupported)
        print(
            f"Warning {txt_path.name}: unsupported symbols will be encoded as spaces: {printable}"
        )

    text_indices = np.asarray(
        text_codec.encode_text(
            text,
            append_eos=text_codec.append_eos_to_dataset,
            normalize=True,
        ),
        dtype=np.int64,
    )
    np.savez_compressed(
        out_npz,
        points=pts,
        text_indices=text_indices,
        vocab_tokens=np.asarray(text_codec.VOCAB_TOKENS),
        eos_token=np.asarray([text_codec.EOS_TOKEN]),
    )


def merge_npz(npz_files, out_npz):
    all_points = []
    all_texts = []

    for file_path in tqdm.tqdm(npz_files, desc="Merging"):
        data = np.load(file_path, allow_pickle=True)
        pts = data["points"]
        text_indices = data["text_indices"]
        if pts.dtype == np.object_:
            pts = pts[0]
        if text_indices.dtype == np.object_:
            text_indices = text_indices[0]
        all_points.append(np.asarray(pts, dtype=np.float32))
        all_texts.append(np.asarray(text_indices, dtype=np.int64).reshape(-1))

    np.savez_compressed(
        out_npz,
        points=np.asarray(all_points, dtype=object),
        text_indices=np.asarray(all_texts, dtype=object),
        vocab_tokens=np.asarray(text_codec.VOCAB_TOKENS),
        eos_token=np.asarray([text_codec.EOS_TOKEN]),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build all_trajectories.npz from trajectory JSON/TXT pairs."
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Folder that contains jsons/, texts/ and npzs/. Defaults to the parent of this scripts/ folder.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    json_dir = dataset_dir / "jsons"
    text_dir = dataset_dir / "texts"
    npz_dir = dataset_dir / "npzs"
    npz_dir.mkdir(parents=True, exist_ok=True)

    npz_files = []
    json_files = sorted(
        json_dir.glob("trajectory_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for json_path in tqdm.tqdm(json_files, desc="Converting json"):
        idx = json_path.stem.split("_")[-1]
        txt_path = text_dir / f"trajectory_{idx}.txt"
        if not txt_path.exists():
            print(f"Skip {json_path.name}: missing {txt_path.name}")
            continue
        out_npz = npz_dir / f"handwriting_trajectory_{idx}.npz"
        convert(json_path, txt_path, out_npz)
        npz_files.append(out_npz)

    if not npz_files:
        raise FileNotFoundError(f"No complete trajectory JSON/TXT pairs found in {dataset_dir}")

    merge_npz(npz_files, dataset_dir / "all_trajectories.npz")


if __name__ == "__main__":
    main()
