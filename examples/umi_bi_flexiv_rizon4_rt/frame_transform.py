"""Per-arm first-frame-relative coordinate conversion for BiFlexiv <-> policy space.

Layout convention (gripper values are always passed through untouched):

- Robot side (native BiFlexiv driver layout):
    [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)]
- Policy side (native UMI per-side-grouped layout, as kept by the dataset
  conversion script):
    [left_tcp(0-8), left_gripper(9), right_tcp(10-18), right_gripper(19)]

The UMI <-> BiFlexiv layout regrouping happens here, at the inference boundary;
training data is stored in the UMI layout.

Each 9D TCP pose is [x, y, z, r1..r6] where r1-r3 / r4-r6 are the first two
COLUMNS of the rotation matrix ("On the Continuity of Rotation Representations
in Neural Networks") — the same convention as the BiFlexiv lerobot driver and
openpi's umi_policy.UmiInputs.

Gripper end-frame convention: the two sides also use different gripper TCP
axis conventions for the same physical TCP point:

- UMI gripper frame (policy/dataset side): x along the fingertips (forward),
  y to the left, z up.
- BiFlexiv gripper frame (robot driver): z along the fingertips (forward),
  y to the right, x up.

So UMI x = flexiv +z, UMI y = flexiv -y, UMI z = flexiv +x. `flexiv_to_policy`
/`policy_to_flexiv` right-multiply each TCP orientation by the (self-inverse)
change-of-basis rotation when `align_gripper_frames` is enabled (default);
translation is unchanged (same physical TCP point).

Frame convention: training data is first-frame-relative — every episode expresses
all of its states/actions in the frame of the episode's first-frame TCP pose, per
arm (see scripts/convert_umi_first_frame_relative.py). At inference the same
convention is reproduced by capturing the episode's first observation as the
reference frame and re-capturing on every episode reset.
"""

import numpy as np

# Layout permutations between the robot-side BiFlexiv layout and the policy-side
# UMI per-side-grouped layout.
_FLEXIV_TO_UMI = [*range(9), 18, *range(9, 18), 19]
_UMI_TO_FLEXIV = [*range(9), *range(10, 19), 9, 19]

# Change-of-basis between the two gripper end-frame conventions (columns are the
# UMI gripper axes expressed in the BiFlexiv gripper frame: UMI x = flexiv +z,
# UMI y = flexiv -y, UMI z = flexiv +x). det = +1, and the matrix is symmetric
# and self-inverse (M @ M = I), so both conversion directions right-multiply by
# the same rotation. Translation is zero — both conventions share the TCP point.
_FLEXIV_GRIPPER_FROM_UMI_GRIPPER = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


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


def _reexpress_pose9(pose: np.ndarray, offset_in_pose_frame: np.ndarray) -> np.ndarray:
    """Right-multiply each pose by a constant offset expressed in the pose's own frame.

    Used for change-of-basis of the TCP end frame: same physical point, rotated
    axes — i.e. pose @ offset, NOT offset @ pose.
    """
    source_pose = _pose9_to_matrix(pose)
    target_pose = np.matmul(source_pose, offset_in_pose_frame)
    return _matrix_to_pose9(target_pose).astype(np.result_type(np.asarray(pose).dtype, np.float32), copy=False)


class PerArmFirstFrameTransform:
    """Per-arm conversion between native BiFlexiv poses and first-frame-relative policy poses.

    The reference frame of each arm is captured from the first state passed after
    construction or `reset()` — i.e. the episode's first observation — and held
    fixed for the rest of the episode. The robot side uses the native BiFlexiv
    layout [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)];
    the policy side uses the UMI per-side-grouped layout [left_tcp(0-8),
    left_gripper(9), right_tcp(10-18), right_gripper(19)]. The regrouping between
    the two layouts is applied on top of the per-arm frame transform; gripper
    dims pass through untouched.

    Args:
        align_gripper_frames: If True (default), rotate each TCP orientation
            between the BiFlexiv gripper end-frame convention (z forward, y
            right, x up) and the UMI one (x forward, y left, z up). Disable
            only when the connected robot driver already reports poses in the
            UMI gripper frame.
    """

    def __init__(self, align_gripper_frames: bool = True) -> None:
        self._align_gripper_frames = align_gripper_frames
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
        # `state` is already in the policy-side UMI layout.
        self._left_ref = _pose9_to_matrix(state[..., :9])
        self._left_ref_inv = np.linalg.inv(self._left_ref)
        self._right_ref = _pose9_to_matrix(state[..., 10:19])
        self._right_ref_inv = np.linalg.inv(self._right_ref)

    def _rotate_gripper_frames(self, state: np.ndarray) -> np.ndarray:
        """Right-multiply both TCP orientations by the gripper change-of-basis (UMI layout).

        The basis rotation is self-inverse, so the same helper serves both
        conversion directions. The offset acts in each pose's own frame (pose @
        M, not M @ pose). Gripper dims are untouched.
        """
        if not self._align_gripper_frames:
            return state
        return np.concatenate(
            (
                _reexpress_pose9(state[..., :9], _FLEXIV_GRIPPER_FROM_UMI_GRIPPER),
                state[..., 9:10],
                _reexpress_pose9(state[..., 10:19], _FLEXIV_GRIPPER_FROM_UMI_GRIPPER),
                state[..., 19:20],
            ),
            axis=-1,
        )

    def flexiv_to_policy(self, state: np.ndarray) -> np.ndarray:
        """Convert a native BiFlexiv state/action to first-frame-relative policy space.

        Regroups the BiFlexiv layout to the UMI per-side-grouped layout and (when
        `align_gripper_frames` is on) rotates the TCP orientations into the UMI
        gripper frame, then captures the reference frames on the first call after
        construction/reset. Only absolute poses are valid inputs — the
        server-side DeltaActions transform turns them into deltas against the
        current state, matching training.
        """
        state = np.asarray(state)
        if state.ndim == 0 or state.shape[-1] != 20:
            raise ValueError(f"Expected BiFlexiv state last dimension 20, got {state.shape}")
        state = self._rotate_gripper_frames(state[..., _FLEXIV_TO_UMI])
        if not self.is_captured:
            self._capture(state)
        left = _transform_pose9(state[..., :9], self._left_ref_inv)
        right = _transform_pose9(state[..., 10:19], self._right_ref_inv)
        return np.concatenate(
            (left, state[..., 9:10], right, state[..., 19:20]),
            axis=-1,
        )

    def policy_to_flexiv(self, actions: np.ndarray) -> np.ndarray:
        """Convert first-frame-relative policy actions back to native BiFlexiv poses.

        Takes UMI-layout policy actions, undoes the per-arm frame transform,
        rotates the TCP orientations back into the BiFlexiv gripper frame (when
        `align_gripper_frames` is on), and regroups back to the BiFlexiv layout
        the robot driver expects.

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
        right = _transform_pose9(actions[..., 10:19], self._right_ref)
        umi_layout = np.concatenate(
            (left, actions[..., 9:10], right, actions[..., 19:20]),
            axis=-1,
        )
        # Back in the flexiv gripper frame (the basis rotation is self-inverse),
        # then regrouped to the BiFlexiv layout the robot driver expects.
        return self._rotate_gripper_frames(umi_layout)[..., _UMI_TO_FLEXIV]
