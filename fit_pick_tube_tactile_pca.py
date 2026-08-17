#!/usr/bin/env python3
"""Fit two 1024-to-15 PCA projections from cached tactile embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

from reactive_diffusion_policy.model.tactile_pca import (
    ARM_COUNT,
    COMPONENTS_PER_ARM,
    TACTILE_SENSOR_ORDER,
    group_tactile_embeddings,
    save_tactile_pca,
)


DEFAULT_DATASETS = tuple(f"pick_tube_{index:02d}" for index in range(1, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tactile-cache-root",
        type=Path,
        default=Path("data/tactile_embeddings_encoder0809"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/PCA_Transform_PickTube/tactile_pca_2x15.npz"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def iter_batches(path: Path, batch_size: int):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 3 or values.shape[1:] != (len(TACTILE_SENSOR_ORDER), 512):
        raise ValueError(f"{path}: expected [N,4,512], got {values.shape}")
    if values.shape[0] < COMPONENTS_PER_ARM:
        raise ValueError(
            f"{path}: PCA requires at least {COMPONENTS_PER_ARM} samples, got {values.shape[0]}"
        )
    start = 0
    while start < values.shape[0]:
        end = min(start + batch_size, values.shape[0])
        if 0 < values.shape[0] - end < COMPONENTS_PER_ARM:
            end = values.shape[0]
        yield group_tactile_embeddings(np.asarray(values[start:end], dtype=np.float32))
        start = end


def main() -> None:
    args = parse_args()
    if args.batch_size < COMPONENTS_PER_ARM:
        raise ValueError(f"batch-size must be at least {COMPONENTS_PER_ARM}")

    cache_paths = {
        dataset: args.tactile_cache_root / "KaiyueChen" / dataset / "embeddings.npy"
        for dataset in args.datasets
    }
    total_frames = sum(
        int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])
        for path in cache_paths.values()
    )

    models = [
        IncrementalPCA(n_components=COMPONENTS_PER_ARM, batch_size=args.batch_size)
        for _ in range(ARM_COUNT)
    ]
    sample_count = 0
    with tqdm(
        total=total_frames,
        desc="Fitting tactile PCA",
        unit="frame",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for dataset in args.datasets:
            dataset_count = 0
            progress.set_postfix(dataset=dataset)
            for grouped in iter_batches(cache_paths[dataset], args.batch_size):
                for arm, model in enumerate(models):
                    model.partial_fit(grouped[:, arm, :])
                batch_frames = grouped.shape[0]
                dataset_count += batch_frames
                sample_count += batch_frames
                progress.update(batch_frames)
            progress.set_postfix(dataset=dataset, dataset_frames=dataset_count)

    means = np.stack([model.mean_ for model in models])
    components = np.stack([model.components_ for model in models])
    explained = np.stack([model.explained_variance_ratio_ for model in models])
    save_tactile_pca(
        args.output,
        means=means,
        components=components,
        explained_variance_ratio=explained,
        sample_count=sample_count,
    )
    for arm in range(ARM_COUNT):
        print(
            f"arm {arm}: 15-component explained variance={explained[arm].sum():.6f}",
            flush=True,
        )
    print(f"saved tactile PCA to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
