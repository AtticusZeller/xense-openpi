"""Trains the RECAP value model on a LeRobot dataset with sidecar returns.

Needs the optional `gemma` package (see `openpi.recap.gemma3`). 10% of the episodes are held out; the
script logs train/held-out loss and MAE, saves the params at every new best held-out metric, stops when
that metric stops improving, and writes the held-out predictions of the best params.

Usage:
    python scripts/recap/train_value.py --repo-id Xense/cylinder-stack-0909 \
        --returns <returns.parquet> --gemma-params <dir> --gemma-tokenizer <tokenizer.model> \
        --output-dir <dir> --exp-name <name>
"""

import dataclasses
import logging
import pathlib
import time
from typing import Literal

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
import sentencepiece
import torch
import tyro
import wandb

from openpi.models import model as _model
import openpi.models.siglip as _siglip
from openpi.recap import gemma3 as _gemma3
from openpi.recap import value_data as _value_data
from openpi.recap import value_model as _value_model
import openpi.shared.download as download
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class Args:
    # LeRobot repo id of the dataset.
    repo_id: str
    # Sidecar parquet written by scripts/recap/compute_returns.py.
    returns: pathlib.Path
    # Local copy of the Gemma3-270M checkpoint (`openpi.recap.gemma3.PARAMS_PATH`).
    gemma_params: pathlib.Path
    # Local copy of the Gemma3 sentencepiece model (`openpi.recap.gemma3.TOKENIZER_PATH`).
    gemma_tokenizer: pathlib.Path
    # Directory for checkpoints and predictions; the run goes to <output_dir>/<exp_name>.
    output_dir: pathlib.Path
    exp_name: str
    # Root of the dataset. Defaults to the LeRobot cache.
    dataset_root: pathlib.Path | None = None
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
    # Every `eval_interval` steps, evaluate on `eval_batches` fixed held-out batches.
    eval_interval: int = 250
    eval_batches: int = 16
    # Held-out metric that selects the saved params: "mae" (of the expected value) or "loss" (cross-entropy).
    best_metric: Literal["mae", "loss"] = "mae"
    # Stop after this many evaluations in a row without a new best.
    patience: int = 3
    num_workers: int = 32
    seed: int = 0
    wandb_enabled: bool = True
    project_name: str = "pi06-recap-value"


def build_model() -> _value_model.ValueModel:
    vision = _siglip.Module(num_classes=None, variant="So400m/14", pool_type="none", scan=True, dtype_mm="float32")
    return _value_model.ValueModel(
        vision=vision, language=_gemma3.Gemma3Backbone(dtype=jnp.float32), width=_gemma3.WIDTH
    )


def replace_subtree(params: dict, name: str, pretrained: dict) -> dict:
    """Replaces `params[name]` with pretrained weights after checking that names and shapes match."""
    expected = flax.traverse_util.flatten_dict(params[name], sep="/")
    loaded = flax.traverse_util.flatten_dict(pretrained, sep="/")
    if expected.keys() != loaded.keys():
        raise ValueError(
            f"{name}: missing {sorted(expected.keys() - loaded.keys())[:5]}, "
            f"unexpected {sorted(loaded.keys() - expected.keys())[:5]}"
        )
    for key, value in expected.items():
        if value.shape != loaded[key].shape:
            raise ValueError(f"{name}/{key}: expected shape {value.shape}, got {loaded[key].shape}")
    converted = {key: jnp.asarray(value, dtype=expected[key].dtype) for key, value in loaded.items()}
    return {**params, name: flax.traverse_util.unflatten_dict(converted, sep="/")}


def init_params(model: _value_model.ValueModel, args: Args) -> dict:
    """Initializes the model and loads the pretrained SigLIP and Gemma3 weights."""
    images = jnp.zeros((1, 3, _value_data.IMAGE_SIZE, _value_data.IMAGE_SIZE, 3), jnp.float32)
    tokens = jnp.zeros((1, args.max_token_len), jnp.int32)
    params = model.init(jax.random.key(args.seed), images, tokens, tokens > 0)["params"]

    openpi_params = _model.restore_params(download.maybe_download(args.siglip_params), restore_type=np.ndarray)
    siglip = {key: value for key, value in openpi_params["PaliGemma"]["img"].items() if key != "head"}
    params = replace_subtree(params, "vision", siglip)
    return replace_subtree(params, "language", _gemma3.load_pretrained_params(str(args.gemma_params)))


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


def save_params(run_dir: pathlib.Path, params: dict) -> None:
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save((run_dir / "params").resolve(), {"params": jax.device_get(params)}, force=True)


def main(args: Args) -> None:
    run_dir = args.output_dir / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root
    dataset = _data_loader.SelectiveVideoLeRobotDataset(args.repo_id, root=dataset_root, tolerance_s=1e-2)
    dataset.decode_video_keys = frozenset(_value_data.BI_FLEXIV_CAMERA_KEYS)
    episodes = _value_data.load_episode_table(pathlib.Path(dataset.root))
    starts, lengths = episodes["dataset_from_index"].to_numpy(), episodes["length"].to_numpy()
    returns = _value_data.align_returns(pd.read_parquet(args.returns), starts, lengths)

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.gemma_tokenizer))
    tasks = _data_loader._convert_tasks_to_dict(dataset.meta.tasks)
    task_tokens = _value_data.tokenize_tasks(
        lambda text: [tokenizer.bos_id(), *tokenizer.encode(text)], tasks, args.max_token_len
    )

    train_episodes, val_episodes = _value_data.split_episodes(len(episodes), args.val_fraction, args.seed)
    train_frames = _value_data.ValueFrames(
        dataset, _value_data.frame_indices(starts, lengths, train_episodes), returns, task_tokens
    )
    val_indices = _value_data.frame_indices(starts, lengths, val_episodes)
    val_frames = _value_data.ValueFrames(dataset, val_indices, returns, task_tokens)
    # A fixed, evenly spread subset of the held-out frames for the periodic evaluation.
    subset = val_indices[np.linspace(0, len(val_indices) - 1, args.eval_batches * args.batch_size).astype(int)]
    val_subset = _value_data.ValueFrames(dataset, subset, returns, task_tokens)
    logging.info(
        "train: %d episodes / %d frames; held out: %d episodes / %d frames %s",
        len(train_episodes),
        len(train_frames),
        len(val_episodes),
        len(val_frames),
        val_episodes.tolist(),
    )

    mesh = jax.make_mesh((jax.device_count(),), ("batch",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    model = build_model()
    params = jax.device_put(init_params(model, args), replicated)
    num_params = sum(x.size for x in jax.tree.leaves(params))
    logging.info("value model parameters: %.1fM", num_params / 1e6)

    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.peak_lr, args.warmup_steps, args.num_train_steps, args.peak_lr * 0.1
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=args.weight_decay))
    opt_state = jax.device_put(optimizer.init(params), replicated)
    train_step, eval_step = make_steps(model, optimizer)

    if args.wandb_enabled:
        wandb.init(
            project=args.project_name, name=args.exp_name, config=dataclasses.asdict(args) | {"num_params": num_params}
        )

    loader = _data_loader.TorchDataLoader(
        train_frames,
        args.batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    data_iter = iter(loader)
    infos, start = [], time.monotonic()
    best_value, best_step, best_params = float("inf"), 0, params
    step = 0
    for step in range(args.num_train_steps + 1):
        if step % args.eval_interval == 0 or step == args.num_train_steps:
            metrics = summarize(predict(eval_step, params, val_subset, args, data_sharding))
            logging.info(
                "Step %d: val_loss=%.4f val_mae=%.4f val_corr=%.4f",
                step,
                metrics["loss"],
                metrics["mae"],
                metrics["corr"],
            )
            if args.wandb_enabled:
                wandb.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
            if metrics[args.best_metric] < best_value:
                best_value, best_step, best_params = metrics[args.best_metric], step, params
                # Written at every new best so an interrupted run still leaves usable params.
                save_params(run_dir, params)
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

    outputs = predict(eval_step, best_params, val_frames, args, data_sharding)
    final = summarize(outputs)
    logging.info(
        "held-out episodes, all frames, params of step %d: loss=%.4f mae=%.4f corr=%.4f",
        best_step,
        final["loss"],
        final["mae"],
        final["corr"],
    )
    episode_of = np.repeat(episodes["episode_index"].to_numpy(), lengths)
    pd.DataFrame(
        {
            "episode_index": episode_of[outputs["index"]],
            "frame_index": outputs["index"] - starts[episode_of[outputs["index"]]],
            "return": outputs["target"],
            "value": outputs["value"],
        }
    ).to_parquet(run_dir / "heldout_predictions.parquet", index=False)
    if args.wandb_enabled:
        wandb.log({f"final/{key}": value for key, value in final.items()} | {"final/best_step": best_step}, step=step)
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(tyro.cli(Args))
