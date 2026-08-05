import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_umi_example() -> dict:
    """Creates a random input example for the UMI (bi_taccap) policy.

    State format (20D, per-side grouped):
        left_tcp.{x, y, z, r1-r6} (9D) + left_gripper.pos (1D) = 10D
        right_tcp.{x, y, z, r1-r6} (9D) + right_gripper.pos (1D) = 10D
    """
    return {
        "state": np.ones((20,)),
        "images": {
            "left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """Inputs for the UMI (bi_taccap_gripper) bimanual policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. Both wrist cameras are required.
      A head camera is accepted for client compatibility but is ignored.
    - state: [20] = [left_tcp.x, left_tcp.y, left_tcp.z, left_tcp.r1..r6, left_gripper.pos,
                     right_tcp.x, right_tcp.y, right_tcp.z, right_tcp.r1..r6, right_gripper.pos]
      (per-side grouped: left TCP dims 0-8, left gripper dim 9, right TCP dims 10-18, right gripper dim 19)
    - actions: [action_horizon, 20]

    UMI datasets have no third-person camera. The model base_0_rgb slot is filled with a black
    image and masked out. Each wrist camera is kept in its corresponding wrist slot.

    The 6D rotation representation (r1-r6) consists of the first two columns of the rotation matrix:
    - [r1, r2, r3]: First column of rotation matrix
    - [r4, r5, r6]: Second column of rotation matrix
    """

    # Both wrist cameras are required. A head camera may be supplied by a compatible robot client,
    # but it is deliberately ignored because UMI training data has no third-person view.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist", "right_wrist")
    OPTIONAL_CAMERAS: ClassVar[tuple[str, ...]] = ("head",)

    def __call__(self, data: dict) -> dict:
        data = _decode_umi(data)

        in_images = data["images"]
        unexpected_cameras = set(in_images) - set(self.EXPECTED_CAMERAS) - set(self.OPTIONAL_CAMERAS)
        if unexpected_cameras:
            raise ValueError(f"Unexpected cameras {tuple(sorted(unexpected_cameras))}; got {tuple(in_images)}")

        missing_cameras = set(self.EXPECTED_CAMERAS) - set(in_images)
        if missing_cameras:
            raise ValueError(f"Missing required wrist cameras {tuple(sorted(missing_cameras))}; got {tuple(in_images)}")

        left_wrist = in_images["left_wrist"]
        right_wrist = in_images["right_wrist"]

        images = {
            "base_0_rgb": np.zeros_like(left_wrist),
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        image_masks = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # No conversion needed - 6D rotation is already a continuous representation.
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOutputs(transforms.DataTransformFn):
    """Outputs for the UMI (bi_taccap) policy.

    Model output format (20 dims, per-side grouped):
        left_tcp.{x, y, z, r1-r6} (9D) + left_gripper.pos (1D) = 10D
        right_tcp.{x, y, z, r1-r6} (9D) + right_gripper.pos (1D) = 10D

    No conversion needed - 6D rotation is already in the correct format.
    """

    def __call__(self, data: dict) -> dict:
        # Return 20 dims (in case model outputs padded actions).
        actions = np.asarray(data["actions"][:, :20])
        return {"actions": actions}


def _decode_umi(data: dict) -> dict:
    """Decode UMI data format.

    Processing steps:
    1. Convert images from [C, H, W] to [H, W, C].

    Args:
        data: Input data dict containing 'state' and 'images'.

    Returns:
        Modified data dict with converted images.
    """
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data
