"""Convert a UMI (bi_taccap) LeRobot dataset to first-frame-relative coordinates.

For every episode, the first frame's left and right TCP poses (position +
orientation) define per-arm reference frames; all `observation.state` and
`action` poses in that episode are re-expressed in those frames. Gripper dims
pass through untouched.

The dataset keeps the native UMI per-side-grouped layout
[left_tcp(0-8), left_gripper(9), right_tcp(10-18), right_gripper(19)].
Regrouping to the BiFlexiv layout reported by the robot driver happens at
inference time in the client-side adapter
(examples/umi_bi_flexiv_rizon4_rt/frame_transform.py).

The output is a regular LeRobot v3.0 dataset (videos hard-linked, data parquets
rewritten, meta/stats.json and the per-episode stats in meta/episodes
recomputed for the converted columns).

Example:

    python scripts/convert_umi_first_frame_relative.py \
        --repo-id TacVerse/taccap-g1-sort-defective-parts-0710 \
        --skip-hub-sync
"""

import json
import logging
import os
import pathlib
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

# Repo root on sys.path so `scripts` is importable no matter where this script is invoked from.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.compute_norm_stats import ensure_lerobot_meta_and_parquet

logger = logging.getLogger(__name__)

STATE_COL = "observation.state"
ACTION_COL = "action"

# UMI per-side-grouped dim slices (layout is preserved by the conversion).
_SRC_LEFT_TCP = slice(0, 9)
_SRC_LEFT_GRIPPER = slice(9, 10)
_SRC_RIGHT_TCP = slice(10, 19)
_SRC_RIGHT_GRIPPER = slice(19, 20)

# Stat fields present in meta/stats.json and meta/episodes parquet stat columns.
_STAT_FIELDS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}

_IDENTITY_POSE9 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _pose9_to_matrix(pose: np.ndarray) -> np.ndarray:
    """[x, y, z, r1..r6] -> 4x4 homogeneous matrix (batched).

    The 6D rotation is re-orthonormalized with Gram-Schmidt (normalize the
    first column, orthogonalize the second against it, third = cross product),
    so slightly non-orthonormal values still yield a valid rotation.
    """
    pose = np.asarray(pose)
    if pose.ndim == 0 or pose.shape[-1] != 9:
        raise ValueError(f"Expected pose last dimension 9, got {pose.shape}")

    first = pose[..., 3:6]
    second = pose[..., 6:9]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Invalid 6D rotation: first column has zero norm")
    first = first / first_norm

    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Invalid 6D rotation: columns are linearly dependent")
    second = second / second_norm
    third = np.cross(first, second)

    matrix = np.zeros((*pose.shape[:-1], 4, 4), dtype=np.result_type(pose.dtype, np.float32))
    matrix[..., :3, :3] = np.stack((first, second, third), axis=-1)
    matrix[..., :3, 3] = pose[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def _matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (matrix[..., :3, 3], matrix[..., :3, 0], matrix[..., :3, 1]),
        axis=-1,
    )


def _transform_pose9(pose: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    source_pose = _pose9_to_matrix(pose)
    target_pose = np.matmul(target_from_source, source_pose)
    return _matrix_to_pose9(target_pose).astype(np.result_type(np.asarray(pose).dtype, np.float32), copy=False)


def _feature_stats(values: np.ndarray) -> dict:
    """Recompute one feature's stats dict (same shape as LeRobot meta/stats.json)."""
    values = np.asarray(values, dtype=np.float64)
    stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }
    for key, q in _QUANTILES.items():
        stats[key] = np.quantile(values, q, axis=0).tolist()
    return stats


def _copy_tree_hardlink(src: pathlib.Path, dst: pathlib.Path) -> list[pathlib.Path]:
    """Hard-link (fallback: copy) every file under src into dst; returns data parquet paths."""
    data_parquets: list[pathlib.Path] = []
    for src_file in sorted(src.rglob("*")):
        if not src_file.is_file() or ".cache" in src_file.relative_to(src).parts:
            continue
        dst_file = dst / src_file.relative_to(src)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src_file, dst_file)
        except OSError:
            shutil.copy2(src_file, dst_file)
        if dst_file.suffix == ".parquet" and dst_file.relative_to(dst).parts[0] == "data":
            data_parquets.append(dst_file)
    return data_parquets


def _atomic_replace(path: pathlib.Path, write_fn) -> None:
    """Write via a temp sibling + rename so hard-linked source files stay intact."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def convert_dataset(src_root: pathlib.Path, dst_root: pathlib.Path) -> None:
    if dst_root.exists():
        raise FileExistsError(f"Output directory already exists: {dst_root}")

    logger.info("Copying dataset tree %s -> %s (hard links) ...", src_root, dst_root)
    data_parquets = _copy_tree_hardlink(src_root, dst_root)
    if not data_parquets:
        raise FileNotFoundError(f"No data parquets found under {src_root}/data")

    # ---- Load all data rows, remembering the source file of each row. ----
    file_ids: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {"episode": [], "frame": [], "state": [], "action": []}
    tables: list[pa.Table] = []
    for file_idx, parquet_path in enumerate(tqdm.tqdm(data_parquets, desc="Reading parquets")):
        table = pq.read_table(parquet_path)
        tables.append(table)
        n = table.num_rows
        file_ids.append(np.full(n, file_idx, dtype=np.int64))
        parts["episode"].append(np.asarray(table.column("episode_index").to_pylist()))
        parts["frame"].append(np.asarray(table.column("frame_index").to_pylist()))
        parts["state"].append(np.asarray(table.column(STATE_COL).to_pylist(), dtype=np.float32))
        parts["action"].append(np.asarray(table.column(ACTION_COL).to_pylist(), dtype=np.float32))

    file_id = np.concatenate(file_ids)
    episode = np.concatenate(parts["episode"])
    frame = np.concatenate(parts["frame"])
    states = np.concatenate(parts["state"])
    actions = np.concatenate(parts["action"])
    logger.info("Loaded %d frames from %d parquet files.", len(episode), len(data_parquets))

    # ---- Per-episode first-frame reference poses (source UMI layout). ----
    unique_episodes = np.unique(episode)
    ref_left_inv: dict[int, np.ndarray] = {}
    ref_right_inv: dict[int, np.ndarray] = {}
    for ep in unique_episodes:
        ep_rows = np.flatnonzero(episode == ep)
        first_row = ep_rows[np.argmin(frame[ep_rows])]
        first_state = states[first_row]
        ref_left_inv[ep] = np.linalg.inv(_pose9_to_matrix(first_state[_SRC_LEFT_TCP]))
        ref_right_inv[ep] = np.linalg.inv(_pose9_to_matrix(first_state[_SRC_RIGHT_TCP]))

    def _convert(values: np.ndarray) -> np.ndarray:
        """First-frame-relative transform (per arm); layout stays UMI per-side-grouped."""
        rows_left_inv = np.stack([ref_left_inv[ep] for ep in episode])
        rows_right_inv = np.stack([ref_right_inv[ep] for ep in episode])
        left_rel = _transform_pose9(values[:, _SRC_LEFT_TCP], rows_left_inv)
        right_rel = _transform_pose9(values[:, _SRC_RIGHT_TCP], rows_right_inv)
        return np.concatenate(
            (left_rel, values[:, _SRC_LEFT_GRIPPER], right_rel, values[:, _SRC_RIGHT_GRIPPER]),
            axis=-1,
        ).astype(np.float32)

    new_states = _convert(states)
    new_actions = _convert(actions)

    # Sanity check: every episode's first frame must map to the identity pose.
    for ep in unique_episodes:
        ep_rows = np.flatnonzero(episode == ep)
        first_row = ep_rows[np.argmin(frame[ep_rows])]
        first = new_states[first_row]
        np.testing.assert_allclose(first[0:9], _IDENTITY_POSE9, atol=1e-4, err_msg=f"ep {ep} left first pose")
        np.testing.assert_allclose(first[10:19], _IDENTITY_POSE9, atol=1e-4, err_msg=f"ep {ep} right first pose")
        np.testing.assert_allclose(first[9], states[first_row][_SRC_LEFT_GRIPPER], atol=1e-6)
        np.testing.assert_allclose(first[19], states[first_row][_SRC_RIGHT_GRIPPER], atol=1e-6)

    # ---- Rewrite data parquets with the converted columns. ----
    for file_idx, parquet_path in enumerate(tqdm.tqdm(data_parquets, desc="Writing parquets")):
        rows = np.flatnonzero(file_id == file_idx)
        table = tables[file_idx]
        for col_name, values in ((STATE_COL, new_states), (ACTION_COL, new_actions)):
            col_idx = table.schema.get_field_index(col_name)
            field = table.schema.field(col_idx)
            flat = pa.array(values[rows].reshape(-1), type=pa.float32())
            list_array = pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
            table = table.set_column(col_idx, field, list_array)
        pq.write_table(table, parquet_path.with_suffix(".tmp"))
        os.replace(parquet_path.with_suffix(".tmp"), parquet_path)

    # ---- meta/stats.json: recompute stats for the converted columns. ----
    stats_path = dst_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[STATE_COL] = _feature_stats(new_states)
    stats[ACTION_COL] = _feature_stats(new_actions)
    _atomic_replace(stats_path, lambda tmp: tmp.write_text(json.dumps(stats, indent=2)))

    # ---- meta/episodes parquets: recompute per-episode stats for converted columns. ----
    episodes_parquets = sorted((dst_root / "meta" / "episodes").rglob("*.parquet"))
    for ep_parquet in episodes_parquets:
        table = pq.read_table(ep_parquet)
        ep_indices = np.asarray(table.column("episode_index").to_pylist())
        per_ep_stats: dict[str, dict[str, list]] = {}
        for ep in ep_indices:
            rows = np.flatnonzero(episode == ep)
            per_ep_stats[ep] = {
                STATE_COL: _feature_stats(new_states[rows]),
                ACTION_COL: _feature_stats(new_actions[rows]),
            }
        for col_name, feature in ((f"stats/{STATE_COL}", STATE_COL), (f"stats/{ACTION_COL}", ACTION_COL)):
            for stat_field in _STAT_FIELDS:
                full_name = f"{col_name}/{stat_field}"
                col_idx = table.schema.get_field_index(full_name)
                if col_idx == -1:
                    continue
                field = table.schema.field(col_idx)
                values = [per_ep_stats[ep][feature][stat_field] for ep in ep_indices]
                element_type = pa.int64() if stat_field == "count" else pa.float64()
                table = table.set_column(col_idx, field, pa.array(values, type=pa.list_(element_type)))
        pq.write_table(table, ep_parquet.with_suffix(".tmp"))
        os.replace(ep_parquet.with_suffix(".tmp"), ep_parquet)

    logger.info(
        "Done. Converted %d episodes / %d frames -> %s",
        len(unique_episodes),
        len(episode),
        dst_root,
    )


def main(
    repo_id: str,
    output_repo_id: str | None = None,
    dataset_root: pathlib.Path | None = None,
    output_root: pathlib.Path | None = None,
    skip_hub_sync: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    output_repo_id = output_repo_id or f"{repo_id}-ffr"
    src_root = ensure_lerobot_meta_and_parquet(repo_id, dataset_root=dataset_root, skip_hub_sync=skip_hub_sync)
    if output_root is None:
        home = src_root.parents[len(repo_id.split("/")) - 1]
        output_root = home.joinpath(*output_repo_id.split("/"))

    logger.info("Source: %s", src_root)
    logger.info("Output: %s (%s)", output_root, output_repo_id)
    convert_dataset(src_root, output_root)


if __name__ == "__main__":
    tyro.cli(main)
