"""Makes a copy of a dataset of initial demonstrations that LeRobot can merge with rollout datasets.

LeRobot only merges datasets whose features are identical. Rollout datasets carry three per-frame columns
(`is_intervention`, `episode_success`, `round`) and record only the cameras the policy sees. This script
writes a copy of the demonstrations with `is_intervention = False`, `episode_success = True` and
`round = 0` on every frame and without the features named by `--remove-features` (for the bi_flexiv
demonstrations: the four tactile video streams), using LeRobot's own `modify_features`. The source dataset
is not modified; the copy holds a full copy of the videos it keeps.

The video codec is not changed: rollouts have to be recorded with the codec of the demonstrations
(`--args.record-vcodec` of examples/bi_flexiv_rizon4_rt/main.py).

Usage:
    python scripts/recap/mark_initial_dataset.py --src <lerobot dataset dir> --dst <new dir> \
        --remove-features observation.images.left_tactile_left observation.images.left_tactile_right \
        observation.images.right_tactile_left observation.images.right_tactile_right
"""

import dataclasses
import logging
import pathlib

import tyro

from openpi.recap import dataset_edit as _dataset_edit


@dataclasses.dataclass(frozen=True)
class Args:
    # Root directory of the LeRobot v3.0 dataset of initial demonstrations.
    src: pathlib.Path
    # Root directory of the marked copy; must not exist.
    dst: pathlib.Path
    # Features of the demonstrations that rollout datasets do not have; the copy leaves them out.
    remove_features: tuple[str, ...] = ()


def main(args: Args) -> None:
    _dataset_edit.mark_initial_demonstrations(args.src, args.dst, args.remove_features)
    logging.info(
        "Wrote %s: %s with the collection columns of round 0, without %s.",
        args.dst, args.src.name, list(args.remove_features) or "any feature removed",
    )  # fmt: skip


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
