"""Trains the RECAP value model on one or more LeRobot datasets with sidecar returns.

Needs the optional `gemma` package (see `openpi.recap.gemma3`). 10% of the episodes of every dataset are
held out; the script logs train/held-out loss and MAE, saves the params at every new best held-out metric,
stops when that metric stops improving, and writes the held-out predictions of the best params. With
several datasets every dataset is sampled equally often and the held-out metrics are averaged over datasets.

Usage:
    python scripts/recap/train_value.py --repo-id Xense/cylinder-stack-0909 \
        --returns <returns.parquet> --gemma-params <dir> --gemma-tokenizer <tokenizer.model> \
        --output-dir <dir> --exp-name <name>
"""

import dataclasses
import itertools
import logging
import pathlib
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import sentencepiece
import torch
import tyro
import wandb

from openpi.models import model as _model
import openpi.models.siglip as _siglip
from openpi.recap import checkpoints as _checkpoints
from openpi.recap import gemma3 as _gemma3
from openpi.recap import value_data as _value_data
from openpi.recap import value_model as _value_model
import openpi.shared.download as download
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class Args:
    # LeRobot repo ids of the datasets.
    repo_id: tuple[str, ...]
    # Sidecar parquets written by scripts/recap/compute_returns.py, one per repo id in the same order.
    returns: tuple[pathlib.Path, ...]
    # Local copy of the Gemma3-270M checkpoint (`openpi.recap.gemma3.PARAMS_PATH`).
    gemma_params: pathlib.Path
    # Local copy of the Gemma3 sentencepiece model (`openpi.recap.gemma3.TOKENIZER_PATH`).
    gemma_tokenizer: pathlib.Path
    # Directory for checkpoints and predictions; the run goes to <output_dir>/<exp_name>.
    output_dir: pathlib.Path
    exp_name: str
    # Roots of the datasets, one per repo id. Defaults to the LeRobot cache.
    dataset_root: tuple[pathlib.Path, ...] = ()
    # `params` directory of an earlier run to start from, instead of the SigLIP and Gemma3 pretrained weights.
    init_params: pathlib.Path | None = None
    # openpi checkpoint providing the SigLIP So400m weights.
    siglip_params: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    val_fraction: float = 0.1
    max_token_len: int = 48
    batch_size: int = 128
    num_train_steps: int = 5000
    peak_lr: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    log_interval: int = 50
    # Every `eval_interval` steps, evaluate on `eval_batches` fixed held-out batches, split evenly over datasets.
    eval_interval: int = 250
    eval_batches: int = 16
    # Held-out metric that selects the saved params: "mae" (of the expected value) or "loss" (cross-entropy).
    best_metric: Literal["mae", "loss"] = "mae"
    # Stop after this many evaluations in a row without a new best.
    patience: int = 3
    # Held-out episodes per dataset predicted in full at the end. Defaults to all of them.
    final_episodes: int | None = None
    num_workers: int = 32
    seed: int = 0
    wandb_enabled: bool = True
    project_name: str = "pi06-recap-value"


def build_model() -> _value_model.ValueModel:
    vision = _siglip.Module(num_classes=None, variant="So400m/14", pool_type="none", scan=True, dtype_mm="float32")
    return _value_model.ValueModel(
        vision=vision, language=_gemma3.Gemma3Backbone(dtype=jnp.float32), width=_gemma3.WIDTH
    )


def init_params(model: _value_model.ValueModel, args: Args) -> dict:
    """Initializes the model, then loads either `args.init_params` or the pretrained SigLIP and Gemma3 weights."""
    images = jnp.zeros((1, 3, _value_data.IMAGE_SIZE, _value_data.IMAGE_SIZE, 3), jnp.float32)
    tokens = jnp.zeros((1, args.max_token_len), jnp.int32)
    params = model.init(jax.random.key(args.seed), images, tokens, tokens > 0)["params"]
    if args.init_params is not None:
        return _checkpoints.match_params(params, _checkpoints.restore_params(args.init_params), "init_params")

    openpi_params = _model.restore_params(download.maybe_download(args.siglip_params), restore_type=np.ndarray)
    siglip = {key: value for key, value in openpi_params["PaliGemma"]["img"].items() if key != "head"}
    gemma = _gemma3.load_pretrained_params(str(args.gemma_params))
    return {
        **params,
        "vision": _checkpoints.match_params(params["vision"], siglip, "vision"),
        "language": _checkpoints.match_params(params["language"], gemma, "language"),
    }


def make_steps(model: _value_model.ValueModel, optimizer: optax.GradientTransformation):
    def forward(params, batch):
        images = batch["images"].astype(jnp.float32) / 127.5 - 1.0
        logits = model.apply({"params": params}, images, batch["tokens"], batch["token_mask"])
        values = _value_model.expected_value(logits)
        return _value_model.value_loss(logits, batch["target"]), values

    def loss_fn(params, batch):
        losses, values = forward(params, batch)
        return losses.mean(), jnp.abs(values - batch["target"]).mean()

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, mae), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, {"loss": loss, "mae": mae}

    return train_step, jax.jit(forward)


def to_device(batch: dict, sharding: jax.sharding.Sharding) -> dict:
    return {key: jax.make_array_from_process_local_data(sharding, np.asarray(value)) for key, value in batch.items()}


def predict(eval_step, params, frames: _value_data.ValueFrames, args: Args, sharding, max_batches: int | None = None):
    """Runs the model over `frames` in order; returns per-frame index, target, value and loss arrays."""
    loader = torch.utils.data.DataLoader(
        frames,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda items: {key: np.stack([item[key] for item in items]) for key in items[0]},
    )
    outputs = {"index": [], "target": [], "value": [], "loss": []}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        size = len(batch["index"])
        if size < args.batch_size:
            # Pad the last batch so the compiled step and the device sharding keep their shapes.
            batch = {
                key: np.concatenate([value, np.repeat(value[-1:], args.batch_size - size, 0)])
                for key, value in batch.items()
            }
        losses, values = eval_step(params, to_device(batch, sharding))
        outputs["index"].append(batch["index"][:size])
        outputs["target"].append(batch["target"][:size])
        outputs["value"].append(np.asarray(values)[:size])
        outputs["loss"].append(np.asarray(losses)[:size])
    return {key: np.concatenate(value) for key, value in outputs.items()}


def summarize(outputs: dict) -> dict:
    error = outputs["value"] - outputs["target"]
    return {
        "loss": float(outputs["loss"].mean()),
        "mae": float(np.abs(error).mean()),
        "corr": float(np.corrcoef(outputs["value"], outputs["target"])[0, 1]),
    }


def load_split(
    repo_id: str, returns: pathlib.Path, root: pathlib.Path | None, tokenizer, args: Args
) -> _value_data.Split:
    dataset = _data_loader.SelectiveVideoLeRobotDataset(repo_id, root=root, tolerance_s=1e-2)
    dataset.decode_video_keys = frozenset(_value_data.BI_FLEXIV_CAMERA_KEYS)
    episodes = _value_data.load_episode_table(pathlib.Path(dataset.root))
    starts, lengths = episodes["dataset_from_index"].to_numpy(), episodes["length"].to_numpy()
    tasks = _data_loader._convert_tasks_to_dict(dataset.meta.tasks)
    train_episodes, val_episodes = _value_data.split_episodes(len(episodes), args.val_fraction, args.seed)
    split = _value_data.Split(
        repo_id=repo_id,
        dataset=dataset,
        returns=_value_data.align_returns(pd.read_parquet(returns), starts, lengths),
        task_tokens=_value_data.tokenize_tasks(
            lambda text: [tokenizer.bos_id(), *tokenizer.encode(text)], tasks, args.max_token_len
        ),
        episodes=episodes,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
    )
    logging.info(
        "%s: train %d episodes / %d frames; held out %d episodes / %d frames %s",
        split.name,
        len(train_episodes),
        len(split.indices(train_episodes)),
        len(val_episodes),
        len(split.indices(val_episodes)),
        val_episodes.tolist(),
    )
    return split


def evaluate(eval_step, params, frames: list[_value_data.ValueFrames], args: Args, sharding) -> list[dict]:
    """Predicts over the datasets in one pass; returns the per-frame outputs of each dataset."""
    outputs = predict(eval_step, params, torch.utils.data.ConcatDataset(frames), args, sharding)
    bounds = np.cumsum([0, *(len(f) for f in frames)])
    return [{key: value[lo:hi] for key, value in outputs.items()} for lo, hi in itertools.pairwise(bounds)]


def mean_metrics(per_dataset: list[dict]) -> dict:
    """Averages metrics over datasets, so every dataset weighs the same whatever its size."""
    return {key: float(np.mean([metrics[key] for metrics in per_dataset])) for key in per_dataset[0]}


def log_metrics(prefix: str, step: int, splits: list[_value_data.Split], per_dataset: list[dict], args: Args) -> dict:
    """Logs per-dataset metrics (when there are several datasets) and their mean; returns the mean."""
    mean = mean_metrics(per_dataset)
    if len(splits) > 1:
        for split, metrics in zip(splits, per_dataset, strict=True):
            logging.info(
                "Step %d [%s]: %s_loss=%.4f %s_mae=%.4f %s_corr=%.4f",
                step, split.name, prefix, metrics["loss"], prefix, metrics["mae"], prefix, metrics["corr"],
            )  # fmt: skip
            if args.wandb_enabled:
                wandb.log({f"{prefix}_{split.name}/{key}": value for key, value in metrics.items()}, step=step)
    logging.info(
        "Step %d: %s_loss=%.4f %s_mae=%.4f %s_corr=%.4f",
        step, prefix, mean["loss"], prefix, mean["mae"], prefix, mean["corr"],
    )  # fmt: skip
    if args.wandb_enabled:
        wandb.log({f"{prefix}/{key}": value for key, value in mean.items()}, step=step)
    return mean


def main(args: Args) -> None:
    roots = args.dataset_root or (None,) * len(args.repo_id)
    if not (len(args.repo_id) == len(args.returns) == len(roots)):
        raise ValueError("--repo-id, --returns and --dataset-root (when given) need one value per dataset.")
    run_dir = args.output_dir / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.gemma_tokenizer))
    splits = [
        load_split(repo_id, returns, root, tokenizer, args)
        for repo_id, returns, root in zip(args.repo_id, args.returns, roots, strict=True)
    ]
    train_frames = [split.frames(split.indices(split.train_episodes)) for split in splits]
    # A fixed, evenly spread subset of each dataset's held-out frames for the periodic evaluation.
    subset_size = max(args.batch_size, args.eval_batches * args.batch_size // len(splits))
    val_subsets = [split.frames(_value_data.spread(split.indices(split.val_episodes), subset_size)) for split in splits]

    mesh = jax.make_mesh((jax.device_count(),), ("batch",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    model = build_model()
    params = jax.device_put(init_params(model, args), replicated)
    num_params = sum(x.size for x in jax.tree.leaves(params))
    logging.info("value model parameters: %.1fM", num_params / 1e6)

    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.peak_lr, args.warmup_steps, max(args.num_train_steps, args.warmup_steps + 1), args.peak_lr * 0.1
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=args.weight_decay))
    opt_state = jax.device_put(optimizer.init(params), replicated)
    train_step, eval_step = make_steps(model, optimizer)

    if args.wandb_enabled:
        wandb.init(
            project=args.project_name, name=args.exp_name, config=dataclasses.asdict(args) | {"num_params": num_params}
        )

    loader = _data_loader.TorchDataLoader(
        torch.utils.data.ConcatDataset(train_frames),
        args.batch_size,
        sharding=data_sharding,
        shuffle=True,
        # With one dataset the default sampler shuffles its frames; with several, every dataset is drawn
        # equally often whatever its size.
        sampler=_value_data.BalancedSampler([len(f) for f in train_frames], seed=args.seed)
        if len(splits) > 1
        else None,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    data_iter = iter(loader)
    infos, start = [], time.monotonic()
    best_value, best_step, best_params = float("inf"), 0, params
    step = 0
    for step in range(args.num_train_steps + 1):
        if step % args.eval_interval == 0 or step == args.num_train_steps:
            outputs = evaluate(eval_step, params, val_subsets, args, data_sharding)
            metrics = log_metrics("val", step, splits, [summarize(o) for o in outputs], args)
            if metrics[args.best_metric] < best_value:
                best_value, best_step, best_params = metrics[args.best_metric], step, params
                # Written at every new best so an interrupted run still leaves usable params.
                _checkpoints.save_params(run_dir / "params", params)
                logging.info("Step %d: new best val_%s=%.4f, params saved", step, args.best_metric, best_value)
            elif step - best_step >= args.patience * args.eval_interval:
                logging.info(
                    "Step %d: early stop, best val_%s=%.4f at step %d", step, args.best_metric, best_value, best_step
                )
                break
            start = time.monotonic()
        if step == args.num_train_steps:
            break
        batch = next(data_iter)
        batch = {key: value for key, value in batch.items() if key != "index"}
        params, opt_state, info = train_step(params, opt_state, batch)
        infos.append(info)
        if (step + 1) % args.log_interval == 0:
            mean = {key: float(np.mean([float(i[key]) for i in infos])) for key in infos[0]}
            step_time = (time.monotonic() - start) / len(infos)
            logging.info("Step %d: loss=%.4f mae=%.4f step_time=%.3fs", step + 1, mean["loss"], mean["mae"], step_time)
            if args.wandb_enabled:
                wandb.log(
                    {
                        "train/loss": mean["loss"],
                        "train/mae": mean["mae"],
                        "train/step_time": step_time,
                        "train/lr": float(schedule(step)),
                    },
                    step=step + 1,
                )
            infos, start = [], time.monotonic()
    loader.close()

    # Whole held-out episodes with the best params, for the value-curve plots.
    final_frames = [split.frames(split.indices(split.val_episodes[: args.final_episodes])) for split in splits]
    outputs = evaluate(eval_step, best_params, final_frames, args, data_sharding)
    logging.info("held-out episodes, params of step %d:", best_step)
    log_metrics("final", step, splits, [summarize(o) for o in outputs], args)
    pd.concat(
        [s.predictions(o["index"], o["target"], o["value"]) for s, o in zip(splits, outputs, strict=True)],
        ignore_index=True,
    ).to_parquet(run_dir / "heldout_predictions.parquet", index=False)
    if args.wandb_enabled:
        wandb.log({"final/best_step": best_step}, step=step)
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(tyro.cli(Args))
