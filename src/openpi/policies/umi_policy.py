import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_umi_example() -> dict:
    """Creates a random input example for the UMI (bi_taccap) policy.

    State format (20D, BiFlexiv layout after conversion):
        left_tcp.{x, y, z, r1-r6} (9D, dims 0-8) + right_tcp.{x, y, z, r1-r6} (9D, dims 9-17)
        left_gripper.pos (1D, dim 18) + right_gripper.pos (1D, dim 19)
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
    - images: dict[name, img] where img is [channel, height, width]. Both wrist cameras are
      required. A head camera is accepted for client compatibility; it is used for the
      base_0_rgb slot only when `use_head_camera` is True, otherwise it is ignored.
    - state: [20] = [left_tcp.x, left_tcp.y, left_tcp.z, left_tcp.r1..r6,
                     right_tcp.x, right_tcp.y, right_tcp.z, right_tcp.r1..r6,
                     left_gripper.pos, right_gripper.pos]
      (BiFlexiv layout: left TCP dims 0-8, right TCP dims 9-17, grippers dims 18-19;
      poses are first-frame-relative, see scripts/convert_umi_first_frame_relative.py)
    - actions: [action_horizon, 20]

    By default UMI datasets have no third-person camera: the model base_0_rgb slot is
    filled with a black image and masked out. Each wrist camera is kept in its
    corresponding wrist slot.

    The 6D rotation representation (r1-r6) consists of the first two columns of the rotation matrix:
    - [r1, r2, r3]: First column of rotation matrix
    - [r4, r5, r6]: Second column of rotation matrix
    """

    # Both wrist cameras are required. A head camera may be supplied by a compatible robot client,
    # but it is ignored unless `use_head_camera` is True.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist", "right_wrist")
    OPTIONAL_CAMERAS: ClassVar[tuple[str, ...]] = ("head",)

    # If True, the head camera is required and fills the base_0_rgb slot (image_mask=True).
    # If False (default), base_0_rgb is filled with a black image and masked out.
    use_head_camera: bool = False

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

        if self.use_head_camera:
            if "head" not in in_images:
                raise ValueError(f"use_head_camera=True but no 'head' image; got {tuple(in_images)}")
            base_image = in_images["head"]
            base_mask = np.True_
        else:
            base_image = np.zeros_like(left_wrist)
            base_mask = np.False_

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        image_masks = {
            "base_0_rgb": base_mask,
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

    Model output format (20 dims, BiFlexiv layout):
        left_tcp.{x, y, z, r1-r6} (9D, dims 0-8) + right_tcp.{x, y, z, r1-r6} (9D, dims 9-17)
        left_gripper.pos (1D, dim 18) + right_gripper.pos (1D, dim 19)

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
