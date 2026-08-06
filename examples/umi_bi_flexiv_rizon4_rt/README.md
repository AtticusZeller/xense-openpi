# UMI checkpoint on BiFlexiv Rizon4 RT

This client keeps the robot and action brokers in the native BiFlexiv 20D
layout while converting WebSocket requests/responses to the policy's
first-frame-relative coordinate space (see
`scripts/convert_umi_first_frame_relative.py`). The head camera is not
connected or sent; only the two wrist cameras are used.

## Start the UMI policy server

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_umi_sort_defective_parts_0710 \
    --policy.dir=<umi_checkpoint_dir> \
    --port=8000
```

The checkpoint must have been trained with the current UMI data conventions:
`base_0_rgb` black/masked, left wrist in the left slot, right wrist in the
right slot, and states/actions expressed per arm in the episode's first-frame
TCP frame with the BiFlexiv dim layout (grippers at dims 18-19).

## Dry-run client

Activate the `lerobot-xense` conda environment, then run:

```bash
python -m examples.umi_bi_flexiv_rizon4_rt.main \
    --args.host 192.168.142.220 \
    --args.port 8000 \
    --args.bi-mount-type forward \
    --args.inner-control-hz 1000 \
    --args.interpolate-cmds \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.dry-run
```

`forward` currently resolves to the `forward-06` hardware preset used by the
installed `lerobot-xense`. A station-qualified preset such as `forward-04`,
`forward-05`, or `forward-06` can also be passed directly.

Dry-run suppresses policy actions, but the robot still connects and the normal
episode reset can move it to the configured start pose. Add
`--args.no-go-to-start` when the connection itself must not perform that move.

## First-frame-relative coordinates

The policy was trained on first-frame-relative data: every training episode
expresses all of its states/actions in the frame of the episode's first-frame
TCP pose, per arm. The client reproduces this by capturing each arm's TCP pose
from the episode's first observation as the reference frame (re-captured on
every episode reset), converting outgoing states and incoming absolute action
chunks with that frame. No extrinsic UMI↔Flexiv calibration is required.

The conversion is applied independently to left/right state, policy actions,
and RTC leftover actions.
