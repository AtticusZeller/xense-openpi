"""Per-arm first-frame-relative coordinate conversion for BiFlexiv <-> policy space.

Layout convention (both the robot driver and the policy use the native BiFlexiv
layout; gripper values are always passed through untouched):

    [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]

Each 9D TCP pose is [x, y, z, r1..r6] where r1-r3 / r4-r6 are the first two
COLUMNS of the rotation matrix ("On the Continuity of Rotation Representations
in Neural Networks") — the same convention as the BiFlexiv lerobot driver and
openpi's umi_policy.UmiInputs.

Frame convention: training data is first-frame-relative — every episode expresses
all of its states/actions in the frame of the episode's first-frame TCP pose, per
arm (see scripts/convert_umi_first_frame_relative.py). At inference the same
convention is reproduced by capturing the episode's first observation as the
reference frame and re-capturing on every episode reset.
"""

import numpy as np


def _pose9_to_matrix(pose: np.ndarray) -> np.ndarray:
    """[x, y, z, r1..r6] -> 4x4 homogeneous matrix (batched).

    The 6D rotation is re-orthonormalized with Gram-Schmidt (normalize the
    first column, orthogonalize the second against it, third = cross product),
    so slightly non-orthonormal network outputs still yield a valid rotation.
    """
    pose = np.asarray(pose)
    if pose.ndim == 0 or pose.shape[-1] != 9:
        raise ValueError(f"Expected pose last dimension 9, got {pose.shape}")

    first = pose[..., 3:6]
    second = pose[..., 6:9]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Invalid 6D rotation: first column has zero norm")
    first = first / first_norm

    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Invalid 6D rotation: columns are linearly dependent")
    second = second / second_norm
    third = np.cross(first, second)

    matrix = np.zeros((*pose.shape[:-1], 4, 4), dtype=np.result_type(pose.dtype, np.float32))
    matrix[..., :3, :3] = np.stack((first, second, third), axis=-1)
    matrix[..., :3, 3] = pose[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def _matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (matrix[..., :3, 3], matrix[..., :3, 0], matrix[..., :3, 1]),
        axis=-1,
    )


def _transform_pose9(pose: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    source_pose = _pose9_to_matrix(pose)
    target_pose = np.matmul(target_from_source, source_pose)
    return _matrix_to_pose9(target_pose).astype(np.result_type(np.asarray(pose).dtype, np.float32), copy=False)


class PerArmFirstFrameTransform:
    """Per-arm conversion between native BiFlexiv poses and first-frame-relative policy poses.

    The reference frame of each arm is captured from the first state passed after
    construction or `reset()` — i.e. the episode's first observation — and held
    fixed for the rest of the episode. Both sides use the native BiFlexiv layout
    [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]; only
    the TCP coordinates change frame, gripper dims pass through untouched.
    """

    def __init__(self) -> None:
        # Per-arm reference pose (first frame) and its inverse, or None until captured.
        self._left_ref: np.ndarray | None = None
        self._left_ref_inv: np.ndarray | None = None
        self._right_ref: np.ndarray | None = None
        self._right_ref_inv: np.ndarray | None = None

    def reset(self) -> None:
        """Drop the captured reference frames; the next state re-captures them."""
        self._left_ref = self._left_ref_inv = None
        self._right_ref = self._right_ref_inv = None

    @property
    def is_captured(self) -> bool:
        return self._left_ref is not None

    def _capture(self, state: np.ndarray) -> None:
        self._left_ref = _pose9_to_matrix(state[..., :9])
        self._left_ref_inv = np.linalg.inv(self._left_ref)
        self._right_ref = _pose9_to_matrix(state[..., 9:18])
        self._right_ref_inv = np.linalg.inv(self._right_ref)

    def flexiv_to_policy(self, state: np.ndarray) -> np.ndarray:
        """Convert a native BiFlexiv state/action to first-frame-relative policy space.

        Captures the reference frames on the first call after construction/reset.
        Only absolute poses are valid inputs — the server-side DeltaActions
        transform turns them into deltas against the current state, matching
        training.
        """
        state = np.asarray(state)
        if state.ndim == 0 or state.shape[-1] != 20:
            raise ValueError(f"Expected BiFlexiv state last dimension 20, got {state.shape}")
        if not self.is_captured:
            self._capture(state)
        left = _transform_pose9(state[..., :9], self._left_ref_inv)
        right = _transform_pose9(state[..., 9:18], self._right_ref_inv)
        return np.concatenate(
            (left, right, state[..., 18:19], state[..., 19:20]),
            axis=-1,
        )

    def policy_to_flexiv(self, actions: np.ndarray) -> np.ndarray:
        """Convert first-frame-relative policy actions back to native BiFlexiv poses.

        Only valid because the policy server's output pipeline includes
        AbsoluteActions: the model emits deltas, the server re-absolutizes them
        against the current state, and what arrives here is an absolute
        first-frame-relative pose. Applying a rigid transform (with translation)
        to raw deltas would be wrong — the translation must only act on absolute
        poses.
        """
        actions = np.asarray(actions)
        if actions.ndim == 0 or actions.shape[-1] != 20:
            raise ValueError(f"Expected policy actions last dimension 20, got {actions.shape}")
        if not self.is_captured:
            raise RuntimeError("Reference frames not captured; call flexiv_to_policy with the first state first")
        left = _transform_pose9(actions[..., :9], self._left_ref)
        right = _transform_pose9(actions[..., 9:18], self._right_ref)
        return np.concatenate(
            (left, right, actions[..., 18:19], actions[..., 19:20]),
            axis=-1,
        )
