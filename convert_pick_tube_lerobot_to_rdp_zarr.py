#!/usr/bin/env python3
"""Convert the fixed pick_tube LeRobot v2.1 contract to an RDP replay buffer."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc
from PIL import Image


DEFAULT_DATASETS = ("pick_tube_01", "pick_tube_02", "pick_tube_03", "pick_tube_04")
CAMERA_KEYS = ("observation.images.camera0", "observation.images.camera1")
STATE_KEY = "observation.state"
ACTION_KEY = "actions"
PARQUET_KEYS = CAMERA_KEYS + (STATE_KEY, ACTION_KEY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/hillbot/datasets"),
        help="Directory containing pick_tube_01, pick_tube_02, ...",
    )
    parser.add_argument(
        "--tactile-cache-root",
        type=Path,
        default=Path("data/tactile_embeddings_encoder0809"),
        help="Root containing KaiyueChen/<dataset>/embeddings.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pick_tube_01_04_rdp_zarr"),
        help="Parent directory for replay_buffer.zarr",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help="Smoke-test limit; omit for the full conversion",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_episode_lengths(dataset_dir: Path) -> tuple[list[dict], dict[int, int]]:
    records = []
    with (dataset_dir / "meta" / "episodes.jsonl").open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))
    records.sort(key=lambda item: int(item["episode_index"]))

    offsets: dict[int, int] = {}
    offset = 0
    for record in records:
        episode_index = int(record["episode_index"])
        offsets[episode_index] = offset
        offset += int(record["length"])
    return records, offsets


def decode_image(value: object, dataset_dir: Path) -> np.ndarray:
    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload is None and value.get("path"):
            payload = (dataset_dir / value["path"]).read_bytes()
    else:
        payload = value
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"unsupported LeRobot image value: {type(value)!r}")
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def parquet_path(dataset_dir: Path, episode_index: int) -> Path:
    return dataset_dir / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def append(array: zarr.Array, values: np.ndarray) -> None:
    old_length = array.shape[0]
    array.resize((old_length + values.shape[0],) + array.shape[1:])
    array[old_length:] = values


def create_output(path: Path) -> tuple[zarr.Group, dict[str, zarr.Array]]:
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    arrays = {
        "camera1": data.create_dataset(
            "camera1", shape=(0, 224, 224, 3), chunks=(16, 224, 224, 3), dtype="u1", compressor=compressor
        ),
        "camera2": data.create_dataset(
            "camera2", shape=(0, 224, 224, 3), chunks=(16, 224, 224, 3), dtype="u1", compressor=compressor
        ),
        "observation_state": data.create_dataset(
            "observation_state", shape=(0, 20), chunks=(2048, 20), dtype="f4", compressor=compressor
        ),
        "tactile_embedding": data.create_dataset(
            "tactile_embedding", shape=(0, 2048), chunks=(256, 2048), dtype="f2", compressor=compressor
        ),
        "action": data.create_dataset(
            "action", shape=(0, 20), chunks=(2048, 20), dtype="f4", compressor=compressor
        ),
    }
    return root, arrays


def main() -> None:
    args = parse_args()
    zarr_path = args.output_dir / "replay_buffer.zarr"
    if zarr_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{zarr_path} already exists; pass --overwrite to replace it")
        shutil.rmtree(zarr_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root, arrays = create_output(zarr_path)
    episode_ends: list[int] = []
    total_frames = 0

    for dataset_name in args.datasets:
        dataset_dir = args.dataset_root / dataset_name
        records, offsets = load_episode_lengths(dataset_dir)
        cache_path = args.tactile_cache_root / "KaiyueChen" / dataset_name / "embeddings.npy"
        tactile_cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if tactile_cache.ndim != 3 or tactile_cache.shape[1:] != (4, 512):
            raise ValueError(f"{cache_path}: expected [N,4,512], got {tactile_cache.shape}")

        selected = records[: args.max_episodes_per_dataset]
        for record in selected:
            episode_index = int(record["episode_index"])
            expected_length = int(record["length"])
            table = pq.read_table(parquet_path(dataset_dir, episode_index), columns=list(PARQUET_KEYS))
            if table.num_rows != expected_length:
                raise ValueError(
                    f"{dataset_name} episode {episode_index}: metadata length {expected_length} != parquet {table.num_rows}"
                )

            camera1 = np.stack([decode_image(value, dataset_dir) for value in table[CAMERA_KEYS[0]].to_pylist()])
            camera2 = np.stack([decode_image(value, dataset_dir) for value in table[CAMERA_KEYS[1]].to_pylist()])
            state = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
            action = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
            start = offsets[episode_index]
            tactile = np.asarray(tactile_cache[start : start + expected_length], dtype=np.float16).reshape(expected_length, 2048)

            if camera1.shape != (expected_length, 224, 224, 3) or camera2.shape != camera1.shape:
                raise ValueError(f"{dataset_name} episode {episode_index}: RGB shape mismatch")
            if state.shape != (expected_length, 20) or action.shape != (expected_length, 20):
                raise ValueError(f"{dataset_name} episode {episode_index}: state/action must be [T,20]")
            if tactile.shape != (expected_length, 2048):
                raise ValueError(f"{dataset_name} episode {episode_index}: tactile shape mismatch")

            for key, values in (
                ("camera1", camera1),
                ("camera2", camera2),
                ("observation_state", state),
                ("tactile_embedding", tactile),
                ("action", action),
            ):
                append(arrays[key], values)
            total_frames += expected_length
            episode_ends.append(total_frames)
            print(f"{dataset_name} episode {episode_index}: {expected_length} frames -> total {total_frames}", flush=True)

    root["meta"].create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(max(1, min(1024, len(episode_ends))),),
        compressor=None,
    )
    print(f"wrote {len(episode_ends)} episodes / {total_frames} frames to {zarr_path}")


if __name__ == "__main__":
    main()
