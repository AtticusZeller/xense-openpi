# UMI checkpoint on BiFlexiv Rizon4 RT

This client keeps the robot and action brokers in the native BiFlexiv 20D
layout while converting WebSocket requests/responses to the UMI training
layout. The head camera is not connected or sent; only the two wrist cameras
are used.

## Start the UMI policy server

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_umi_sort_defective_parts_0710 \
    --policy.dir=<umi_checkpoint_dir> \
    --port=8000
```

The checkpoint must have been trained with the current UMI camera mapping:
`base_0_rgb` black/masked, left wrist in the left slot, and right wrist in the
right slot.

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

BiFlexiv reports each TCP in its own arm coordinate frame, while the UMI data
uses a shared Pico4 world frame. Real execution therefore uses two separate
rigid transforms. An identity template is provided at
`examples/umi_bi_flexiv_rizon4_rt/frame_calibration_identity.json`:

```json
{
  "left_flexiv_from_umi": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "right_flexiv_from_umi": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
}
```

The identity values mean that the corresponding UMI and Flexiv arm axes and
origins are assumed equal. Replace them after measuring non-identity offsets.
Load this file with:

```bash
--args.frame-calibration examples/umi_bi_flexiv_rizon4_rt/frame_calibration_identity.json
```

The conversion is applied independently to left/right state, policy actions,
and RTC leftover actions. Real execution is blocked when the calibration file
is absent.
