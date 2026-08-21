#!/usr/bin/env python3
"""Precompute four 512-D tactile features from pick_tube LeRobot v2.1 Parquet."""

from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image

from reactive_diffusion_policy.model.tactile_encoder_jax import load_tactile_encoder
from reactive_diffusion_policy.model.tactile_encoder_jax import encode_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pick_tube_tactile_cache_0809.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "Dataset names or repo IDs to process. When provided, --dataset-root "
            "must point to the directory containing their local folders. Bare "
            "names are interpreted as KaiyueChen/<name>."
        ),
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-episodes-per-dataset", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-after-episodes", type=int)
    return parser.parse_args()


def decode_image(value: object) -> np.ndarray:
    payload = value.get("bytes") if isinstance(value, dict) else value
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"unsupported LeRobot image value: {type(value)!r}")
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_episodes(dataset_dir: Path) -> list[dict]:
    with (dataset_dir / "meta" / "episodes.jsonl").open(encoding="utf-8") as file:
        episodes = [json.loads(line) for line in file]
    return sorted(episodes, key=lambda item: int(item["episode_index"]))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache_config = config["tactile_embedding_cache"]
    model_config = config["model"]
    if args.datasets is not None:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required when --datasets is provided")
        config["datasets"] = [
            {
                "repo_id": value if "/" in value else f"KaiyueChen/{value}",
                "root": str(args.dataset_root / value.rsplit("/", 1)[-1]),
            }
            for value in args.datasets
        ]
    if args.dataset_root is not None:
        for source in config["datasets"]:
            source["root"] = str(args.dataset_root / str(source["repo_id"]).rsplit("/", 1)[-1])
    if args.cache_root is not None:
        cache_config["root"] = str(args.cache_root)
    if args.encoder_path is not None:
        model_config["tactile_encoder_path"] = str(args.encoder_path)
    cache_root = Path(cache_config["root"])
    tactile_keys = tuple(model_config["tactile_keys"])
    batch_size = args.batch_size or int(cache_config["precompute_batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cache_config["precompute_num_workers"])
    embedding_dim = int(model_config["tactile_embedding_dim"])
    image_size = int(model_config["tactile_image_size"])

    bundle = load_tactile_encoder(model_config["tactile_encoder_path"])
    checkpoint_config = bundle.metadata["tactile_clip_config"]
    if embedding_dim != int(checkpoint_config["embedding_dim"]) or image_size != int(checkpoint_config["tactile_image_size"]):
        raise ValueError("config and encoder embedding/image dimensions differ")
    resnet_params = bundle.params["tactile_resnet"]

    @jax.jit
    def encode(images: jax.Array) -> jax.Array:
        values, _ = encode_resnet18(
            resnet_params,
            images,
            train=False,
            embedding_dim=embedding_dim,
        )
        return values

    print(f"JAX devices={jax.devices()}", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for source in config["datasets"]:
            repo_id = str(source["repo_id"])
            dataset_dir = Path(source["root"])
            episodes = load_episodes(dataset_dir)
            if args.max_episodes_per_dataset is not None:
                episodes = episodes[: args.max_episodes_per_dataset]
            total_frames = sum(int(item["length"]) for item in episodes)
            output_dir = cache_root.joinpath(*repo_id.split("/"))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "embeddings.npy"
            metadata_path = output_dir / "metadata.json"
            progress_path = output_dir / "progress.json"
            if output_path.exists() and metadata_path.exists() and not args.overwrite:
                print(f"exists, skipping: {output_path}", flush=True)
                continue

            completed_episodes = 0
            if output_path.exists() and not args.overwrite:
                if progress_path.exists():
                    completed_episodes = int(json.loads(progress_path.read_text())["completed_episodes"])
                elif args.resume_after_episodes is not None:
                    completed_episodes = args.resume_after_episodes
                    args.resume_after_episodes = None
                else:
                    raise ValueError(
                        f"partial cache has no progress marker: {output_path}; "
                        "pass --resume-after-episodes or --overwrite"
                    )
                write_index = sum(int(item["length"]) for item in episodes[:completed_episodes])
                embeddings = np.lib.format.open_memmap(output_path, mode="r+")
                expected_shape = (total_frames, len(tactile_keys), embedding_dim)
                if embeddings.shape != expected_shape or embeddings.dtype != np.float16:
                    raise ValueError(
                        f"{output_path}: expected {expected_shape} float16, "
                        f"got {embeddings.shape} {embeddings.dtype}"
                    )
                print(
                    f"resuming: {repo_id} after episode {completed_episodes}, frame {write_index}",
                    flush=True,
                )
            else:
                embeddings = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(total_frames, len(tactile_keys), embedding_dim),
                )
                write_index = 0
            started = time.perf_counter()
            for episode_number, episode in enumerate(
                episodes[completed_episodes:], start=completed_episodes + 1
            ):
                episode_index = int(episode["episode_index"])
                episode_path = (
                    dataset_dir
                    / "data"
                    / f"chunk-{episode_index // 1000:03d}"
                    / f"episode_{episode_index:06d}.parquet"
                )
                table = pq.read_table(episode_path, columns=list(tactile_keys))
                if table.num_rows != int(episode["length"]):
                    raise ValueError(f"{episode_path}: episode length mismatch")
                columns = [table[key].to_pylist() for key in tactile_keys]
                for start in range(0, table.num_rows, batch_size):
                    end = min(start + batch_size, table.num_rows)
                    values = [columns[sensor][frame] for frame in range(start, end) for sensor in range(len(tactile_keys))]
                    images = np.stack(list(executor.map(decode_image, values)))
                    if images.shape[1:] != (image_size, image_size, 3):
                        raise ValueError(f"{episode_path}: tactile image shape {images.shape}")
                    encoded = encode(jnp.asarray(images, dtype=jnp.float32) * (1.0 / 255.0))
                    count = end - start
                    encoded = np.asarray(jax.device_get(encoded), dtype=np.float32).reshape(
                        count, len(tactile_keys), embedding_dim
                    )
                    embeddings[write_index : write_index + count] = encoded.astype(np.float16)
                    write_index += count
                progress_path.write_text(
                    json.dumps({"completed_episodes": episode_number, "frames": write_index}) + "\n",
                    encoding="utf-8",
                )
                if episode_number % 10 == 0 or episode_number == len(episodes):
                    embeddings.flush()
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"{repo_id}: episode {episode_number}/{len(episodes)} "
                        f"frames {write_index}/{total_frames} ({write_index / elapsed:.1f} frames/s)",
                        flush=True,
                    )

            metadata = {
                "repo_id": repo_id,
                "total_frames": total_frames,
                "tactile_keys": list(tactile_keys),
                "shape": [total_frames, len(tactile_keys), embedding_dim],
                "dtype": "float16",
                "encoder_path": str(Path(model_config["tactile_encoder_path"]).resolve()),
            }
            embeddings.flush()
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            progress_path.unlink(missing_ok=True)
            print(f"completed: {output_path}", flush=True)


if __name__ == "__main__":
    main()
