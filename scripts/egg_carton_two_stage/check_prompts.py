"""Print the prompt that the training input pipeline feeds the model for sampled frames.

Builds the dataset exactly as training does (`create_torch_dataset` + the transform list of
`transform_dataset`), runs the transforms one by one, and reports for each sampled frame:

* `task_index` stored in the frame,
* the `task` string LeRobot attaches (`meta.tasks.iloc[task_index]`, positional lookup),
* the `prompt` from `PromptFromLeRobotTask` (lookup by the `task_index` column),
* the `prompt` right before tokenization (after `InjectDefaultPrompt`),
* the decoded text of the tokenized prompt, i.e. what the model actually reads.

Usage (from the xense-openpi root):
    python scripts/egg_carton_two_stage/check_prompts.py --config <train-config> \
        [--repo-id ...] [--dataset-root ...] [--episodes 0 5 121] [--cuts cuts.csv]
"""

import argparse
import csv
import dataclasses
import pathlib
import sys

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

_TOKENIZE_TYPES = (_transforms.TokenizePrompt, _transforms.TokenizeFASTInputs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Train config name (configs/<name>.yaml).")
    parser.add_argument("--repo-id", help="Override data.repo_id.")
    parser.add_argument("--dataset-root", help="Override data.base_config.dataset_root.")
    parser.add_argument("--episodes", type=int, nargs="*", help="Episode indices; default: 4 evenly spaced.")
    parser.add_argument(
        "--positions",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.9, 1.0],
        help="Fractions of each episode to sample (0 = first frame, 1 = last frame).",
    )
    parser.add_argument(
        "--cuts",
        type=pathlib.Path,
        help="CSV with episode_index,cut_s; adds frames cut-1 and cut = round(cut_s * fps) per episode.",
    )
    return parser.parse_args()


def _load_train_config(args: argparse.Namespace) -> _config.TrainConfig:
    train_config = _config.get_config(args.config)
    factory = train_config.data
    if args.repo_id is not None:
        factory = dataclasses.replace(factory, repo_id=args.repo_id)
    if args.dataset_root is not None:
        factory = dataclasses.replace(
            factory, base_config=dataclasses.replace(factory.base_config, dataset_root=args.dataset_root)
        )
    return dataclasses.replace(train_config, data=factory)


def _read_cuts(path: pathlib.Path, fps: int) -> dict[int, int]:
    """Map episode index to the first frame of the second stage, matching relabel.py."""
    with path.open() as f:
        return {int(row["episode_index"]): round(float(row["cut_s"]) * fps) for row in csv.DictReader(f)}


def _sample_frames(length: int, positions: list[float], cut: int | None) -> list[int]:
    frames = {round(p * (length - 1)) for p in positions}
    if cut is not None:
        frames |= {cut - 1, cut}
    return sorted(f for f in frames if 0 <= f < length)


def _decode_prompt_tokens(tokenize: _transforms.DataTransformFn, data: dict) -> str:
    """Decode the tokenized prompt and drop the discretized state digits for readability."""
    sp = tokenize.tokenizer._tokenizer  # SentencePiece processor owned by the tokenizer.
    tokens = np.asarray(data["tokenized_prompt"])[np.asarray(data["tokenized_prompt_mask"], dtype=bool)]
    text = sp.decode(tokens.tolist())
    head, sep, _ = text.partition(", State:")
    return f"{head}{sep} <state ...>" if sep else text


def _run_pipeline(item: dict, pipeline: list[_transforms.DataTransformFn]) -> tuple[str | None, str | None]:
    """Apply the training transforms; return (prompt before tokenization, decoded model text)."""
    data = item
    prompt_before_tokenize = decoded = None
    for transform in pipeline:
        if isinstance(transform, _TOKENIZE_TYPES):
            prompt = data.get("prompt")
            prompt_before_tokenize = None if prompt is None else str(np.asarray(prompt).item())
            data = transform(data)
            if isinstance(transform, _transforms.TokenizePrompt):
                decoded = _decode_prompt_tokens(transform, data)
            continue
        data = transform(data)
    return prompt_before_tokenize, decoded


def main() -> int:
    args = _parse_args()
    train_config = _load_train_config(args)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    print(
        f"config={args.config} repo_id={data_config.repo_id} dataset_root={_config.resolve_dataset_root(data_config)}"
    )
    print(f"prompt_from_task={data_config.prompt_from_task} default_prompt={train_config.data.default_prompt!r}")

    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    # Same transform list as `_data_loader.transform_dataset`; normalization does not touch the prompt.
    pipeline = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    dataset_root = _config.resolve_dataset_root(data_config)
    meta = LeRobotDatasetMetadata(data_config.repo_id, root=dataset_root)
    episodes = meta.episodes
    print(f"total_episodes={meta.total_episodes} total_frames={meta.total_frames} tasks:")
    for task, row in meta.tasks.iterrows():
        print(f"  task_index={int(row['task_index'])}: {task!r}")

    episode_ids = args.episodes
    if not episode_ids:
        episode_ids = sorted({round(x) for x in np.linspace(0, meta.total_episodes - 1, 4)})
    cuts = _read_cuts(args.cuts, meta.fps) if args.cuts else {}

    mismatches = 0
    for ep in episode_ids:
        start = int(episodes[ep]["dataset_from_index"])
        length = int(episodes[ep]["length"])
        print(f"\nepisode {ep}: length={length} global_index=[{start}, {start + length}) tasks={episodes[ep]['tasks']}")
        for frame in _sample_frames(length, args.positions, cuts.get(ep)):
            item = dataset[start + frame]
            if int(item["frame_index"]) != frame or int(item["episode_index"]) != ep:
                raise RuntimeError(f"index mapping broken at ep={ep} frame={frame}: got {item['frame_index']}")
            task_index = int(item["task_index"])
            task, prompt_column = item.get("task"), item.get("prompt")
            final_prompt, decoded = _run_pipeline(item, pipeline)
            consistent = task == final_prompt and (prompt_column is None or prompt_column == final_prompt)
            mismatches += not consistent
            print(
                f"  frame {frame:5d} task_index={task_index} final_prompt={final_prompt!r}"
                f"{'' if consistent else f'  MISMATCH task={task!r} prompt_from_task={prompt_column!r}'}"
            )
            print(f"             model text: {decoded!r}")
    print(f"\nmismatches={mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
