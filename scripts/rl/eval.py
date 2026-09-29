"""Labeled evaluation of one frozen policy on the real robot.

Runs rounds over the same robot session as online RLT training - the robot host runs
``examples/bi_flexiv_rizon4_rt`` with ``--args.rlt`` - and the operator uses the same
buttons: open a window at the critical phase, label it success or failure. Every
labeled window is one trial; the run ends once ``--trials`` trials count. Nothing is
trained. ``openpi.rl.eval`` defines how trials are counted.

    mamba activate lerobot-xense
    python scripts/rl/eval.py <rlt_config> --exp-name <run> --arm vla --trials 20 --eval-name <name>
    python scripts/rl/eval.py <rlt_config> --exp-name <run> --arm rlt [--actor <rl/<round> | snapshot>] \\
        --trials 20 --eval-name <name>

``--arm vla`` is the frozen VLA alone; ``--arm rlt`` puts the trained actor in the
windows (default: the run's latest round). Outputs under
``<checkpoint_base_dir>/<config>/<exp-name>/eval/<eval-name>/``: ``trials.jsonl``, one
line per trial, and ``events.jsonl``; W&B logs the trial and round axes as run
``<config>/<exp-name>/eval/<eval-name>``.
"""

import argparse
import dataclasses
import json
import logging
import shutil

import wandb

from openpi.rl import eval as _eval
from openpi.rl import run_logger as _run_logger
from openpi.rl.algos.rlt import config as _rlt_config
from openpi.rl.algos.rlt import eval_arm as _eval_arm
from openpi.rl.env import protocol as env_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="RLT config name (configs/rl/rlt/<name>.yaml)")
    parser.add_argument("--exp-name", required=True, help="RLT run whose token (and actor) checkpoints to evaluate")
    parser.add_argument("--arm", required=True, choices=_eval_arm.ARMS)
    parser.add_argument("--actor", help="--arm rlt: an rl/<round> dir or actor_critic_round*.pkl snapshot")
    parser.add_argument("--trials", required=True, type=int, help="labeled trials to collect")
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.actor is not None and args.arm != "rlt":
        parser.error("--actor only applies to --arm rlt")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    config = dataclasses.replace(_rlt_config.get_config(args.config), exp_name=args.exp_name)
    out_dir = config.checkpoint_dir / "eval" / args.eval_name
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite or choose another --eval-name.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    arm = _eval_arm.create_arm(config, args.arm, args.actor)
    run = wandb.init(
        project=config.project_name,
        name=f"{config.name}/{config.exp_name}/eval/{args.eval_name}",
        config={"rlt": dataclasses.asdict(config), "arm": args.arm, "actor": args.actor, "trials": args.trials},
        mode=None if config.wandb_enabled else "disabled",
    )
    logger = _run_logger.RunLogger(run if config.wandb_enabled else None, out_dir, axes=_eval.AXES)

    rl = config.rl
    host, port = rl.listen.rsplit(":", 1)
    env = env_protocol.RemoteEnv(host, int(port), state_dim=arm.space.state_dim, action_dim=arm.space.action_dim)
    evaluation = _eval.Evaluation(
        env,
        arm,
        action_dim=arm.space.action_dim,
        takeover_position_m=rl.takeover_position_m,
        takeover_rotation_deg=rl.takeover_rotation_deg,
        capture_stride=rl.replay_stride,
        trials=args.trials,
        out_dir=out_dir,
        logger=logger,
    )
    try:
        summary = evaluation.run()
    finally:
        env.close()
    logging.info("Evaluation %s (%s arm):\n%s", args.eval_name, arm.name, json.dumps(summary, indent=2))
    if config.wandb_enabled:
        run.summary.update(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
