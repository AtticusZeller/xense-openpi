"""Per-arm coordinate conversion between UMI world and Flexiv arm frames."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def _rigid_transform(value: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix, got {matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def _pose9_to_matrix(pose: np.ndarray) -> np.ndarray:
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


@dataclass(frozen=True)
class PerArmUmiBiFlexivTransform:
    """Two independent transforms from the shared UMI frame to each arm frame."""

    left_flexiv_from_umi: np.ndarray
    right_flexiv_from_umi: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_flexiv_from_umi",
            _rigid_transform(self.left_flexiv_from_umi, name="left_flexiv_from_umi"),
        )
        object.__setattr__(
            self,
            "right_flexiv_from_umi",
            _rigid_transform(self.right_flexiv_from_umi, name="right_flexiv_from_umi"),
        )

    @classmethod
    def identity(cls) -> "PerArmUmiBiFlexivTransform":
        return cls(np.eye(4), np.eye(4))

    @classmethod
    def from_json(cls, path: str | Path) -> "PerArmUmiBiFlexivTransform":
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)
        required = {"left_flexiv_from_umi", "right_flexiv_from_umi"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"Calibration file is missing keys: {tuple(sorted(missing))}")
        return cls(data["left_flexiv_from_umi"], data["right_flexiv_from_umi"])

    def flexiv_state_to_umi(self, state: np.ndarray) -> np.ndarray:
        """Convert native BiFlexiv state to per-side-grouped UMI state."""
        state = np.asarray(state)
        if state.ndim == 0 or state.shape[-1] != 20:
            raise ValueError(f"Expected BiFlexiv state last dimension 20, got {state.shape}")
        left = _transform_pose9(state[..., :9], np.linalg.inv(self.left_flexiv_from_umi))
        right = _transform_pose9(state[..., 9:18], np.linalg.inv(self.right_flexiv_from_umi))
        return np.concatenate(
            (left, state[..., 18:19], right, state[..., 19:20]),
            axis=-1,
        )

    def flexiv_actions_to_umi(self, actions: np.ndarray) -> np.ndarray:
        return self.flexiv_state_to_umi(actions)

    def umi_actions_to_flexiv(self, actions: np.ndarray) -> np.ndarray:
        """Convert per-side-grouped UMI actions to native BiFlexiv actions."""
        actions = np.asarray(actions)
        if actions.ndim == 0 or actions.shape[-1] != 20:
            raise ValueError(f"Expected UMI actions last dimension 20, got {actions.shape}")
        left = _transform_pose9(actions[..., :9], self.left_flexiv_from_umi)
        right = _transform_pose9(actions[..., 10:19], self.right_flexiv_from_umi)
        return np.concatenate(
            (left, right, actions[..., 9:10], actions[..., 19:20]),
            axis=-1,
        )
