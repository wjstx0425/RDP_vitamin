#!/usr/bin/env python3
"""Convert the fixed pick_tube LeRobot v2.1 contract to an RDP replay buffer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc
from PIL import Image
from tqdm.auto import tqdm

from reactive_diffusion_policy.common.pick_tube_action_contract import (
    ACTION_CONTRACT,
    ACTION_REPRESENTATION_VERSION,
    HIGH_GRIPPER_DELTA_M,
    HIGH_ROTATION_DELTA_DEG,
    HIGH_TRANSLATION_DELTA_M,
    IDLE_ENTRY_FRAMES,
    IDLE_EXIT_FRAMES,
    LOW_GRIPPER_DELTA_M,
    LOW_ROTATION_DELTA_DEG,
    LOW_TRANSLATION_DELTA_M,
    TERMINAL_ACTION_POLICY,
    canonicalize_episode_actions,
)
from reactive_diffusion_policy.model.tactile_pca import BimanualTactilePCA


DEFAULT_DATASETS = (
    "pick_tube_01",
    "pick_tube_02",
    "pick_tube_03",
    "pick_tube_04",
    "pick_tube_05",
    "pick_tube_06",
)
DEFAULT_DATASET_REPEATS = ("pick_tube_05=2", "pick_tube_06=2")
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
        default=Path("data/pick_tube_01_06_pca30_rdp_zarr"),
        help="Parent directory for replay_buffer.zarr",
    )
    parser.add_argument(
        "--tactile-pca-path",
        type=Path,
        default=Path("data/PCA_Transform_PickTube/tactile_pca_2x15.npz"),
        help="Two-arm PCA artifact produced by fit_pick_tube_tactile_pca.py",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--dataset-repeats",
        nargs="*",
        default=list(DEFAULT_DATASET_REPEATS),
        metavar="DATASET=REPEAT",
        help="Training-only episode repeat factors stored as Zarr metadata.",
    )
    parser.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help="Smoke-test limit; omit for the full conversion",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_dataset_repeats(items: list[str]) -> dict[str, int]:
    repeats: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid repeat specification {item!r}; expected DATASET=REPEAT")
        dataset, value = item.rsplit("=", 1)
        repeat = int(value)
        if repeat < 1:
            raise ValueError(f"repeat factor must be positive, got {item!r}")
        repeats[dataset] = repeat
    return repeats


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


def extract_float32_matrix(
    column: pa.ChunkedArray, *, expected_width: int, name: str
) -> np.ndarray:
    """Extract a list-valued Arrow column without casting its physical values."""
    column_type = column.type
    if pa.types.is_fixed_size_list(column_type):
        if column_type.list_size != expected_width:
            raise ValueError(
                f"{name} must have width {expected_width}, got {column_type.list_size}"
            )
    elif not (pa.types.is_list(column_type) or pa.types.is_large_list(column_type)):
        raise ValueError(f"{name} must be an Arrow list column, got {column_type}")

    if not pa.types.is_float32(column_type.value_type):
        raise ValueError(
            f"{name} must contain float32 values, got {column_type.value_type}"
        )
    if column.null_count:
        raise ValueError(f"{name} must not contain null rows")

    matrices = []
    for chunk in column.chunks:
        if not pa.types.is_fixed_size_list(column_type):
            lengths = np.diff(chunk.offsets.to_numpy(zero_copy_only=False))
            if not np.all(lengths == expected_width):
                raise ValueError(f"{name} rows must all have width {expected_width}")
        flat = chunk.flatten()
        if flat.null_count:
            raise ValueError(f"{name} must not contain null values")
        values = flat.to_numpy(zero_copy_only=False)
        if values.dtype != np.float32:
            raise ValueError(f"{name} extraction changed dtype to {values.dtype}")
        matrices.append(values.reshape(len(chunk), expected_width))

    if not matrices:
        return np.empty((0, expected_width), dtype=np.float32)
    return np.concatenate(matrices, axis=0)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_array_content(array: zarr.Array) -> str:
    """Hash array metadata and canonical C-order content in bounded chunks."""
    digest = hashlib.sha256()
    digest.update(str(np.dtype(array.dtype)).encode("ascii"))
    digest.update(stable_json_sha256(list(array.shape)).encode("ascii"))
    chunk_rows = max(1, int(array.chunks[0]))
    for start in range(0, int(array.shape[0]), chunk_rows):
        values = np.asarray(array[start : start + chunk_rows])
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def converter_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_v2_manifest(
    *,
    arrays: dict[str, zarr.Array],
    pca_path: Path,
    tactile_cache_paths: dict[str, Path],
    tactile_embedding_dim: int,
    episode_manifest: list[dict],
    repair_counts: dict[str, int],
    idle_coverage_by_source: dict[str, dict[str, float]],
    git_commit: str,
) -> dict:
    """Build the JSON-serializable audit manifest for a completed conversion."""
    canonical_action_sha256 = sha256_array_content(arrays["action"])
    source_action_sha256 = sha256_array_content(arrays["action_raw"])
    dataset_digest = stable_json_sha256(
        {
            "source_episodes": episode_manifest,
            "canonical_action_sha256": canonical_action_sha256,
            "source_action_sha256": source_action_sha256,
        }
    )
    return {
        "action_representation_version": ACTION_REPRESENTATION_VERSION,
        "action_contract": ACTION_CONTRACT,
        "normalizer_version": "zero_centered_v2",
        "terminal_action_policy": TERMINAL_ACTION_POLICY,
        "idle_thresholds": {
            "low": {
                "translation_m": LOW_TRANSLATION_DELTA_M,
                "rotation_deg": LOW_ROTATION_DELTA_DEG,
                "gripper_m": LOW_GRIPPER_DELTA_M,
            },
            "high": {
                "translation_m": HIGH_TRANSLATION_DELTA_M,
                "rotation_deg": HIGH_ROTATION_DELTA_DEG,
                "gripper_m": HIGH_GRIPPER_DELTA_M,
            },
            "entry_frames": IDLE_ENTRY_FRAMES,
            "exit_frames": IDLE_EXIT_FRAMES,
        },
        "repair_counts": dict(repair_counts),
        "idle_coverage_by_source": idle_coverage_by_source,
        "arrays": {
            key: {"shape": list(array.shape), "dtype": str(np.dtype(array.dtype))}
            for key, array in sorted(arrays.items())
        },
        "pca_sha256": sha256_file(Path(pca_path)),
        "tactile_cache_sha256": stable_json_sha256(
            {
                dataset_name: sha256_file(Path(cache_path))
                for dataset_name, cache_path in sorted(tactile_cache_paths.items())
            }
        ),
        "pca_output_dim": int(tactile_embedding_dim),
        "pca_sensor_to_arm_order": ["left", "right"],
        "source_episodes": episode_manifest,
        "canonical_action_sha256": canonical_action_sha256,
        "source_action_sha256": source_action_sha256,
        "dataset_digest": dataset_digest,
        "converter_git_commit": git_commit,
    }


def create_output(
    path: Path, tactile_embedding_dim: int
) -> tuple[zarr.Group, dict[str, zarr.Array]]:
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
            "tactile_embedding",
            shape=(0, tactile_embedding_dim),
            chunks=(2048, tactile_embedding_dim),
            dtype="f4",
            compressor=compressor,
        ),
        "action": data.create_dataset(
            "action", shape=(0, 20), chunks=(2048, 20), dtype="f4", compressor=compressor
        ),
        "action_raw": data.create_dataset(
            "action_raw", shape=(0, 20), chunks=(2048, 20), dtype="f4", compressor=compressor
        ),
        "action_valid": data.create_dataset(
            "action_valid", shape=(0,), chunks=(2048,), dtype="bool", compressor=compressor
        ),
        "idle_arm_mask": data.create_dataset(
            "idle_arm_mask", shape=(0, 2), chunks=(2048, 2), dtype="bool", compressor=compressor
        ),
    }
    return root, arrays


def main() -> None:
    args = parse_args()
    dataset_repeats = parse_dataset_repeats(args.dataset_repeats)
    zarr_path = args.output_dir / "replay_buffer.zarr"
    tactile_pca = BimanualTactilePCA.from_npz(args.tactile_pca_path)
    tactile_embedding_dim = tactile_pca.output_dim
    if zarr_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{zarr_path} already exists; pass --overwrite to replace it")
        shutil.rmtree(zarr_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conversion_frame_count = 0
    for dataset_name in args.datasets:
        records, _ = load_episode_lengths(args.dataset_root / dataset_name)
        selected = records[: args.max_episodes_per_dataset]
        conversion_frame_count += sum(int(record["length"]) for record in selected)

    root, arrays = create_output(zarr_path, tactile_embedding_dim)
    episode_ends: list[int] = []
    episode_repeats: list[int] = []
    episode_dataset_ids: list[int] = []
    episode_manifest: list[dict] = []
    source_idle_counts = {
        dataset_name: np.zeros(2, dtype=np.int64) for dataset_name in args.datasets
    }
    source_valid_counts = {dataset_name: 0 for dataset_name in args.datasets}
    tactile_cache_paths = {
        dataset_name: args.tactile_cache_root
        / "KaiyueChen"
        / dataset_name
        / "embeddings.npy"
        for dataset_name in args.datasets
    }
    total_frames = 0

    progress = tqdm(
        total=conversion_frame_count,
        desc=f"Converting to PCA{tactile_embedding_dim} Zarr",
        unit="frame",
        unit_scale=True,
        dynamic_ncols=True,
    )

    for dataset_id, dataset_name in enumerate(args.datasets):
        dataset_dir = args.dataset_root / dataset_name
        records, offsets = load_episode_lengths(dataset_dir)
        cache_path = tactile_cache_paths[dataset_name]
        tactile_cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if tactile_cache.ndim != 3 or tactile_cache.shape[1:] != (4, 512):
            raise ValueError(f"{cache_path}: expected [N,4,512], got {tactile_cache.shape}")

        selected = records[: args.max_episodes_per_dataset]
        for record in selected:
            episode_index = int(record["episode_index"])
            expected_length = int(record["length"])
            progress.set_postfix(dataset=dataset_name, episode=episode_index)
            table = pq.read_table(parquet_path(dataset_dir, episode_index), columns=list(PARQUET_KEYS))
            if table.num_rows != expected_length:
                raise ValueError(
                    f"{dataset_name} episode {episode_index}: metadata length {expected_length} != parquet {table.num_rows}"
                )

            camera1 = np.stack([decode_image(value, dataset_dir) for value in table[CAMERA_KEYS[0]].to_pylist()])
            camera2 = np.stack([decode_image(value, dataset_dir) for value in table[CAMERA_KEYS[1]].to_pylist()])
            state = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
            action = extract_float32_matrix(
                table[ACTION_KEY], expected_width=20, name=ACTION_KEY
            )
            start = offsets[episode_index]
            tactile_raw = np.asarray(
                tactile_cache[start : start + expected_length], dtype=np.float32
            )
            tactile = tactile_pca.transform_numpy(tactile_raw)

            if camera1.shape != (expected_length, 224, 224, 3) or camera2.shape != camera1.shape:
                raise ValueError(f"{dataset_name} episode {episode_index}: RGB shape mismatch")
            if state.shape != (expected_length, 20) or action.shape != (expected_length, 20):
                raise ValueError(f"{dataset_name} episode {episode_index}: state/action must be [T,20]")
            if tactile.shape != (expected_length, tactile_embedding_dim):
                raise ValueError(f"{dataset_name} episode {episode_index}: tactile shape mismatch")

            canonical_actions = canonicalize_episode_actions(state, action)

            for key, values in (
                ("camera1", camera1),
                ("camera2", camera2),
                ("observation_state", state),
                ("tactile_embedding", tactile),
                ("action_raw", canonical_actions.action_raw),
                ("action", canonical_actions.action),
                ("action_valid", canonical_actions.action_valid),
                ("idle_arm_mask", canonical_actions.idle_arm_mask),
            ):
                append(arrays[key], values)
            source_idle_counts[dataset_name] += canonical_actions.idle_arm_mask.sum(axis=0)
            source_valid_counts[dataset_name] += int(canonical_actions.action_valid.sum())
            total_frames += expected_length
            episode_ends.append(total_frames)
            repeat = dataset_repeats.get(dataset_name, 1)
            episode_repeats.append(repeat)
            episode_dataset_ids.append(dataset_id)
            episode_manifest.append(
                {
                    "dataset": dataset_name,
                    "dataset_id": dataset_id,
                    "episode_index": episode_index,
                    "length": expected_length,
                    "repeat": repeat,
                }
            )
            progress.update(expected_length)

    progress.close()

    root["meta"].create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(max(1, min(1024, len(episode_ends))),),
        compressor=None,
    )
    root["meta"].create_dataset(
        "episode_repeats",
        data=np.asarray(episode_repeats, dtype=np.int16),
        chunks=(max(1, min(1024, len(episode_repeats))),),
        compressor=None,
    )
    root["meta"].create_dataset(
        "episode_dataset_ids",
        data=np.asarray(episode_dataset_ids, dtype=np.int16),
        chunks=(max(1, min(1024, len(episode_dataset_ids))),),
        compressor=None,
    )
    root["meta"].attrs["dataset_names"] = list(args.datasets)
    root["meta"].attrs["tactile_pca_path"] = str(args.tactile_pca_path.resolve())
    root["meta"].attrs["tactile_embedding_dim"] = tactile_embedding_dim
    idle_coverage_by_source = {}
    for dataset_name in args.datasets:
        denominator = source_valid_counts[dataset_name]
        counts = source_idle_counts[dataset_name]
        idle_coverage_by_source[dataset_name] = {
            "left": float(counts[0] / denominator) if denominator else 0.0,
            "right": float(counts[1] / denominator) if denominator else 0.0,
        }
    repair_counts = {
        "terminal_actions": len(episode_manifest),
        "invalid_nonterminal_actions": 0,
        "idle_frames_left": int(sum(counts[0] for counts in source_idle_counts.values())),
        "idle_frames_right": int(sum(counts[1] for counts in source_idle_counts.values())),
    }
    manifest = build_v2_manifest(
        arrays=arrays,
        pca_path=args.tactile_pca_path,
        tactile_cache_paths=tactile_cache_paths,
        tactile_embedding_dim=tactile_embedding_dim,
        episode_manifest=episode_manifest,
        repair_counts=repair_counts,
        idle_coverage_by_source=idle_coverage_by_source,
        git_commit=converter_git_commit(),
    )
    root["meta"].attrs["v2_manifest_json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    )
    starts = [0, *episode_ends[:-1]]
    effective_frames = sum(
        (end - start) * repeat
        for start, end, repeat in zip(starts, episode_ends, episode_repeats)
    )
    print(
        f"wrote {len(episode_ends)} episodes / {total_frames} physical frames / "
        f"{effective_frames} effective training frames to {zarr_path}"
    )


if __name__ == "__main__":
    main()
