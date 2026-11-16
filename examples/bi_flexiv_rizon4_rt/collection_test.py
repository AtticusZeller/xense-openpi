"""RECAP rollout collection, without robot or VR hardware.

The real Runtime, ActionChunkBroker, intervention agent, takeover controller and
recorder run against a fake bench, a fake policy and a scripted Pico4, and write a
real LeRobotDataset that the assertions read back.
"""

from collections import deque
import logging
from types import SimpleNamespace

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.utils.robot_utils as _robot_utils
import numpy as np
import pytest

if not hasattr(_robot_utils, "get_logger"):
    # The training environment ships upstream lerobot, which lacks the xense fork's logger.
    _robot_utils.get_logger = logging.getLogger

from xense_client import action_chunk_broker
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

from examples.bi_flexiv_rizon4_rt import collection
from examples.bi_flexiv_rizon4_rt import intervention
from examples.bi_flexiv_rizon4_rt import recorder as _recorder

ACTION_HORIZON = 5
STEP_LIMIT = 12
ROUND = 2
_IDENTITY = [0.0, 0.0, 0.0, 1.0]  # qx, qy, qz, qw
_CAMERAS = ("head", "left_wrist", "right_wrist")


class Episode:
    """One scripted episode: the step each input happens on (None: never)."""

    def __init__(self, press=None, grip_from=None, move_from=None, release_at=None, stale=(), gate=("A",)):
        self.press = press or {}  # step -> face button read on that step
        self.grip_from = grip_from  # left grip held from this step...
        self.release_at = release_at  # ...until this step
        self.move_from = move_from  # left controller displaced 6 mm from this step
        self.stale = list(stale)  # presses made while the arms home
        self.gate = list(gate)  # presses made at the gate, one poll each


class Bench:
    """Fake robot, cameras and Pico4, all driven by the scripted episode and its step counter."""

    def __init__(self, episodes):
        self._episodes = list(episodes)
        self.episode = Episode()
        self.step = 0  # control steps applied in the running episode
        self.applied = []  # actions sent to the robot, per episode
        self.button_queue = deque()  # what each drain of the button monitor returns
        self.resyncs = 0

    # --- xense_client Environment ---
    def reset(self):
        self.episode = self._episodes.pop(0) if self._episodes else Episode()
        self.step = 0
        self.applied.append([])
        self.button_queue = deque([self.episode.stale, *[[button] for button in self.episode.gate]])

    def is_episode_complete(self):
        return False

    def get_observation(self):
        image = np.full((48, 64, 3), self.step, dtype=np.uint8)
        return {"state": np.full(20, self.step, dtype=np.float32), "images_raw": dict.fromkeys(_CAMERAS, image)}

    def apply_action(self, action):
        self.applied[-1].append(np.asarray(action["actions"], dtype=np.float32).copy())
        self.step += 1
        if self.step in self.episode.press:
            self.button_queue.append([self.episode.press[self.step]])

    # --- Pico4ButtonMonitor ---
    def poll(self):
        pass

    def consume_events(self):
        return self.button_queue.popleft() if self.button_queue else []

    # --- XenseVR SDK ---
    def _gripping(self):
        ep = self.episode
        return ep.grip_from is not None and ep.grip_from <= self.step < ep.release_at

    def get_left_grip(self):
        return 0.9 if self._gripping() else 0.0

    def get_right_grip(self):
        return 0.0

    def get_left_controller_pose(self):
        moved = self.episode.move_from is not None and self.step >= self.episode.move_from
        return np.array([0.1 + (0.006 if moved else 0.0), 0.2, 0.3, *_IDENTITY])

    def get_right_controller_pose(self):
        return np.array([-0.1, 0.2, 0.3, *_IDENTITY])

    def get_left_trigger(self):
        return 0.0

    def get_right_trigger(self):
        return 0.0

    # --- BiPico4 teleop ---
    def teleop_action(self):
        return dict.fromkeys(intervention._ACTION_KEYS_IN_ORDER, -float(self.step + 1))

    def reset_to_pose(self, *_):
        self.resyncs += 1


class FakePolicy:
    """Chunk ``n`` holds the value ``100 * n + i`` at index ``i``, so an action names its inference."""

    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        steps = np.arange(ACTION_HORIZON, dtype=np.float32)[:, None]
        return {"actions": np.broadcast_to(100.0 * self.calls + steps, (ACTION_HORIZON, 20)).copy()}

    def reset(self):
        pass

    def warmup(self, obs):
        pass


def _collect(tmp_path, episodes, *, run=True):
    """Wire the collection stack the way main.py does and run it over the scripted episodes."""
    bench = Bench(episodes)
    config = SimpleNamespace(
        grip_enable_threshold=0.5, grip_disable_threshold=0.3, left_gripper_width=1.0, right_gripper_width=1.0
    )
    teleop = SimpleNamespace(
        _xrt=bench, config=config, reset_to_pose=bench.reset_to_pose, get_action=bench.teleop_action
    )
    tcp = np.array([0.0, 0, 0, 1, 0, 0, 0, 1.0])
    robot = SimpleNamespace(get_current_tcp_pose_quat=lambda: (tcp, tcp))
    controller = collection.CollectionController(
        teleop, SimpleNamespace(_env=SimpleNamespace(robot=robot)), takeover_motion=collection.TAKEOVER_MOTION
    )
    controller._buttons = bench  # No sampler thread: the script decides what each drain returns.

    rec = _recorder.make_collection_recorder_subscriber(
        repo_id="test/collection",
        task="stack the cylinders",
        round_index=ROUND,
        root=tmp_path / "dataset",
        image_height=48,
        image_width=64,
    )
    environment = collection.EpisodeControlEnvironment(bench, controller, on_label=rec.label_episode)
    policy = FakePolicy()
    broker = action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=ACTION_HORIZON)
    agent = intervention.InterventionPolicyAgent(
        inner_agent=_policy_agent.PolicyAgent(policy=broker), controller=controller, broker=broker
    )
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[rec],
        num_episodes=len(episodes),
        max_episode_steps=STEP_LIMIT,
    )
    if run:
        runtime.run()
        rec.finalize()
    return SimpleNamespace(
        bench=bench, runtime=runtime, environment=environment, recorder=rec, root=tmp_path / "dataset"
    )


def _read(root):
    dataset = LeRobotDataset("test/collection", root=root)
    table = dataset.hf_dataset.with_format("numpy")
    columns = {name: np.asarray(values) for name, values in table[:].items()}
    episodes = []
    for index in range(dataset.meta.total_episodes):
        rows = columns["episode_index"] == index
        episodes.append({name: values[rows] for name, values in columns.items()})
    return dataset, episodes


# Saved episodes sit between unsaved ones, so a dropped episode that still took an index would show.
SCRIPT = [
    Episode(press={6: "B"}, stale=["A"], gate=["X", "A"]),
    Episode(press={3: "X"}),
    Episode(press={10: "B"}, grip_from=3, move_from=4, release_at=7),
    Episode(),  # runs into the step limit
    Episode(press={5: "Y"}),
    Episode(press={4: "A"}),
]


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    run = _collect(tmp_path_factory.mktemp("collection"), SCRIPT)
    dataset, episodes = _read(run.root)
    return SimpleNamespace(run=run, dataset=dataset, episodes=episodes)


def test_only_labeled_episodes_are_saved_and_indices_stay_contiguous(collected):
    assert collected.dataset.meta.total_episodes == 3
    # The frame on which B / Y is read is the last one.
    assert [len(episode["frame_index"]) for episode in collected.episodes] == [7, 11, 6]
    # The bench ran all six; the unsaved ones still moved the robot.
    assert [len(actions) for actions in collected.run.bench.applied[:6]] == [7, 4, 11, STEP_LIMIT + 1, 6, 5]


def test_episode_success_is_constant_and_matches_the_button(collected):
    for episode, success in zip(collected.episodes, [True, True, False], strict=True):
        assert episode["episode_success"].dtype == np.bool_
        assert episode["episode_success"].tolist() == [success] * len(episode["frame_index"])


def test_round_is_the_collection_round(collected):
    for episode in collected.episodes:
        assert episode["round"].dtype == np.int64
        assert set(episode["round"].tolist()) == {ROUND}


def test_is_intervention_marks_exactly_the_taken_over_steps(collected):
    autonomous, taken_over, failed = collected.episodes
    assert not autonomous["is_intervention"].any()
    assert not failed["is_intervention"].any()
    # Grip at step 3 only arms the takeover, the controller moves at step 4, the grip opens at step 7.
    assert np.flatnonzero(taken_over["is_intervention"]).tolist() == [4, 5, 6]


def test_recorded_actions_are_what_the_robot_was_sent(collected):
    saved = [0, 2, 4]  # positions of the saved episodes in SCRIPT
    for episode, index in zip(collected.episodes, saved, strict=True):
        np.testing.assert_array_equal(episode["action"], np.stack(collected.run.bench.applied[index]))
        np.testing.assert_array_equal(episode["observation.state"][:, 0], np.arange(len(episode["frame_index"])))


def test_takeover_records_the_teleop_action_and_release_reinfers(collected):
    actions = collected.episodes[1]["action"]
    chunk = int(actions[0, 0]) // 100
    # Steps 0-3 come from one chunk, in order.
    np.testing.assert_array_equal(actions[:4], np.broadcast_to(100.0 * chunk + np.arange(4)[:, None], (4, 20)))
    # Steps 4-6: the left hand drives, so every left-arm, right-arm and left-gripper target is the
    # teleop's; the right hand's grip is open, so its gripper keeps the opening last commanded.
    for step in (4, 5, 6):
        np.testing.assert_array_equal(actions[step, :19], np.full(19, -(step + 1.0)))
        assert actions[step, 19] == actions[3, 19]
    # Step 7 is index 0 of a new chunk: the rest of the interrupted one (its index 4) is never executed.
    np.testing.assert_array_equal(actions[7:], np.broadcast_to(100.0 * (chunk + 1) + np.arange(4)[:, None], (4, 20)))
    assert collected.run.bench.resyncs == 1


def test_stop_request_at_the_gate_raises_instead_of_starting_the_episode(tmp_path):
    run = _collect(tmp_path, [Episode(gate=[])], run=False)
    run.environment.request_stop()
    with pytest.raises(KeyboardInterrupt):
        run.runtime.run()
    run.recorder.finalize()
    assert run.bench.applied == [[]]


def test_episode_interrupted_by_the_runtime_is_not_saved(tmp_path):
    run = _collect(tmp_path, [Episode()], run=False)
    bench_apply = run.bench.apply_action

    def apply_then_stop(action):
        bench_apply(action)
        if run.bench.step == 3:
            run.runtime.request_stop()  # what the SIGINT handler does

    run.bench.apply_action = apply_then_stop
    run.runtime.run()
    run.recorder.finalize()
    assert len(run.bench.applied[0]) == 3
    assert run.recorder._dataset.meta.total_episodes == 0


def test_recorded_columns_are_the_ones_the_training_side_merges_and_checks():
    # The robot host does not install openpi, so the definition exists twice; LeRobot refuses to merge
    # the rollouts into the marked demonstrations as soon as the two differ.
    collection_columns = pytest.importorskip("openpi.recap.collection_columns")
    assert _recorder.COLLECTION_FEATURES == collection_columns.COLLECTION_FEATURES


def test_videos_are_recorded_in_the_codec_of_the_demonstrations(collected):
    # LeRobot compares the codec when merging; the bi_flexiv demonstrations are h264.
    cameras = [f"observation.images.{camera}" for camera in _CAMERAS]
    assert collected.dataset.meta.video_keys == cameras
    assert {collected.dataset.meta.features[camera]["info"]["video.codec"] for camera in cameras} == {"h264"}
