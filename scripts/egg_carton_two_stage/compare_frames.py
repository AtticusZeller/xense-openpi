"""Compare decoded frames of matching episodes in two LeRobot datasets.

Used after LeRobot's delete / merge operations to check that every kept episode still decodes to the
same images and the same state / action rows. For each ``A_EP:B_EP`` pair it samples frames evenly
(first and last included) and reports, per camera, the maximum absolute pixel difference (0-255) and the
minimum PSNR. Exits non-zero if a numeric column differs or a pixel difference exceeds ``--max-abs-diff``.

Usage:
    python scripts/egg_carton_two_stage/compare_frames.py \
        --a-root <root> --a-repo-id <id> --b-root <root> --b-repo-id <id> --pairs 0:0 109:108 \
        [--frames-per-episode 5] [--max-abs-diff 0]
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

_NUMERIC_KEYS = ("observation.state", "action", "frame_index", "timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-root", type=pathlib.Path, required=True)
    parser.add_argument("--a-repo-id", required=True)
    parser.add_argument("--b-root", type=pathlib.Path, required=True)
    parser.add_argument("--b-repo-id", required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="Episode pairs A_EP:B_EP.")
    parser.add_argument("--frames-per-episode", type=int, default=5)
    parser.add_argument("--max-abs-diff", type=float, default=0.0, help="Allowed pixel difference (0-255).")
    return parser.parse_args()


def load(repo_id: str, root: pathlib.Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=root, video_backend="pyav")


def psnr(max_abs: float, mse: float) -> float:
    return float("inf") if max_abs == 0 else 10 * np.log10(255.0**2 / mse)


def compare_frame(a: dict, b: dict, video_keys: list[str]) -> tuple[list[str], dict[str, tuple[float, float]]]:
    """Numeric mismatches and per-camera (max abs diff, PSNR) for one frame."""
    mismatches = [k for k in _NUMERIC_KEYS if not torch.equal(a[k], b[k])]
    images = {}
    for key in video_keys:
        diff = (a[key].float() - b[key].float()) * 255.0
        max_abs = float(diff.abs().max())
        images[key] = (max_abs, psnr(max_abs, float((diff**2).mean())))
    return mismatches, images


def main() -> int:
    args = parse_args()
    ds_a, ds_b = load(args.a_repo_id, args.a_root), load(args.b_repo_id, args.b_root)
    video_keys = ds_a.meta.video_keys
    failed = False
    for pair in args.pairs:
        ep_a, ep_b = (int(x) for x in pair.split(":"))
        ea, eb = ds_a.meta.episodes[ep_a], ds_b.meta.episodes[ep_b]
        if ea["length"] != eb["length"]:
            print(f"{pair}: length {ea['length']} vs {eb['length']}")
            failed = True
            continue
        frames = sorted({round(x) for x in np.linspace(0, ea["length"] - 1, args.frames_per_episode)})
        worst = {k: (0.0, float("inf")) for k in video_keys}
        numeric_bad = set()
        for frame in frames:
            mismatches, images = compare_frame(
                ds_a[ea["dataset_from_index"] + frame], ds_b[eb["dataset_from_index"] + frame], video_keys
            )
            numeric_bad.update(mismatches)
            for k, (max_abs, p) in images.items():
                worst[k] = (max(worst[k][0], max_abs), min(worst[k][1], p))
        max_abs_all = max(v[0] for v in worst.values())
        min_psnr = min(v[1] for v in worst.values())
        bad = bool(numeric_bad) or max_abs_all > args.max_abs_diff
        failed |= bad
        print(
            f"{pair}: frames={frames} numeric_mismatch={sorted(numeric_bad) or 'none'} "
            f"max_abs_diff={max_abs_all:.1f} min_psnr={min_psnr:.1f} dB {'FAIL' if bad else 'ok'}"
        )
        if max_abs_all > 0:
            for k, (m, p) in worst.items():
                print(f"    {k}: max_abs_diff={m:.1f} min_psnr={p:.1f} dB")
    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
