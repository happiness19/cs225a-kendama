"""Replay a Flexiv RDK free-drive log through OpenSai.

The latest log was recorded from the robot's RDK state while the primitive was
FloatingCartesian. During that mode the OpenSai Cartesian state keys in the CSV
can be stale, so this script replays the changing `flange_pose`/`tcp_pose`
column instead of `flange_transform` or `cartesian_task_current_orientation`
when Cartesian mode is requested. Joint mode uses the recorded `q` trajectory,
which is the default because it is frame-invariant for this same robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import redis


DEFAULT_LOG = Path("logs/latest.csv")
DEFAULT_EXPECTED_CONFIG = "kendama.xml"

# This is the default pose from main.py, repeated here to avoid importing a
# script that parses command line arguments at import time.
DEFAULT_JOINTS = np.array(
    [
        math.radians(1.70),
        math.radians(-74.46),
        math.radians(-2.47),
        math.radians(75.42),
        math.radians(-0.90),
        math.radians(148.87),
        math.radians(-5.87),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PoseSample:
    t: float
    position: np.ndarray
    orientation: np.ndarray


@dataclass(frozen=True)
class JointSample:
    t: float
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


@dataclass(frozen=True)
class RedisKeys:
    joint_controller_name: str
    cartesian_controller_name: str
    active_controller: str
    config_file_name: str
    joint_goal_position: str
    joint_goal_velocity: str
    joint_goal_acceleration: str
    joint_current_controller: str
    joint_current_sensor: str
    cartesian_goal_position: str
    cartesian_goal_orientation: str
    cartesian_current_position: str
    cartesian_current_orientation: str


def make_keys(robot_name: str, joint_controller_name: str, cartesian_controller_name: str) -> RedisKeys:
    base = f"opensai::controllers::{robot_name}"
    joint_base = f"{base}::{joint_controller_name}::joint_task"
    cartesian_base = f"{base}::{cartesian_controller_name}::cartesian_task"
    sensors = f"opensai::sensors::{robot_name}"
    return RedisKeys(
        joint_controller_name=joint_controller_name,
        cartesian_controller_name=cartesian_controller_name,
        active_controller=f"{base}::active_controller_name",
        config_file_name="::sai-interfaces-webui::config_file_name",
        joint_goal_position=f"{joint_base}::goal_position",
        joint_goal_velocity=f"{joint_base}::goal_velocity",
        joint_goal_acceleration=f"{joint_base}::goal_acceleration",
        joint_current_controller=f"{joint_base}::current_position",
        joint_current_sensor=f"{sensors}::joint_positions",
        cartesian_goal_position=f"{cartesian_base}::goal_position",
        cartesian_goal_orientation=f"{cartesian_base}::goal_orientation",
        cartesian_current_position=f"{cartesian_base}::current_position",
        cartesian_current_orientation=f"{cartesian_base}::current_orientation",
    )


def parse_vec(text: str, expected: int | None = None) -> np.ndarray:
    values = np.array([float(x) for x in text.replace(",", " ").split()], dtype=float)
    if expected is not None and values.size != expected:
        raise ValueError(f"Expected {expected} values, got {values.size}")
    return values


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("Quaternion has near-zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=float,
        )
    else:
        i = int(np.argmax(np.diag(matrix)))
        if i == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ],
                dtype=float,
            )
        elif i == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ],
                dtype=float,
            )
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=float,
            )
    return quat / np.linalg.norm(quat)


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        quat = q0 + alpha * (q1 - q0)
        return quat / np.linalg.norm(quat)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    return math.cos(theta) * q0 + sin_theta / sin_theta_0 * (q1 - dot * q0)


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    rel = np.asarray(a).reshape(3, 3) @ np.asarray(b).reshape(3, 3).T
    cos_angle = (float(np.trace(rel)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos_angle)))


def parse_pose(text: str) -> tuple[np.ndarray, np.ndarray]:
    pose = parse_vec(text, expected=7)
    return pose[:3].copy(), quat_wxyz_to_matrix(pose[3:7])


def parse_json_matrix(text: str, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.array(json.loads(text), dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"Expected matrix shape {shape}, got {matrix.shape}")
    return matrix


def row_time(row: dict[str, str], time_source: str) -> float:
    if row.get("t"):
        return float(row["t"])
    if time_source == "robot" and row.get("robot_sec") and row.get("robot_nsec"):
        return float(row["robot_sec"]) + 1e-9 * float(row["robot_nsec"])
    if row.get("t_wall"):
        return float(row["t_wall"])
    raise ValueError("Cannot determine sample time from this log row")


def load_trajectory(
    log_path: Path,
    pose_column: str,
    time_source: str,
) -> tuple[list[PoseSample], dict[str, str]]:
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{log_path} has no CSV header")
        if pose_column not in reader.fieldnames:
            raise ValueError(f"{log_path} does not contain a {pose_column!r} column")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{log_path} has no samples")

    samples: list[PoseSample] = []
    raw_times: list[float] = []
    for row in rows:
        position, orientation = parse_pose(row[pose_column])
        raw_times.append(row_time(row, time_source))
        samples.append(PoseSample(0.0, position, orientation))

    times = np.asarray(raw_times, dtype=float)
    rel_times = times - times[0]
    if np.any(np.diff(rel_times) < -1e-9):
        raise ValueError(f"{time_source} timestamps are not monotonic")

    samples = [
        PoseSample(float(t), sample.position, sample.orientation)
        for t, sample in zip(rel_times, samples)
    ]

    return samples, rows[0]


def derive_joint_derivatives(samples: list[JointSample]) -> list[JointSample]:
    if len(samples) < 2:
        return samples
    times = np.asarray([sample.t for sample in samples], dtype=float)
    q = np.vstack([sample.q for sample in samples])
    dq = np.gradient(q, times, axis=0, edge_order=1)
    ddq = np.gradient(dq, times, axis=0, edge_order=1)
    return [
        JointSample(float(t), q_i.copy(), dq_i.copy(), ddq_i.copy())
        for t, q_i, dq_i, ddq_i in zip(times, q, dq, ddq)
    ]


def load_joint_trajectory(
    log_path: Path,
    time_source: str,
) -> tuple[list[JointSample], dict[str, str]]:
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{log_path} has no CSV header")
        if "q" not in reader.fieldnames:
            raise ValueError(f"{log_path} does not contain a 'q' column")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{log_path} has no samples")

    raw_times = np.asarray([row_time(row, time_source) for row in rows], dtype=float)
    rel_times = raw_times - raw_times[0]
    if np.any(np.diff(rel_times) < -1e-9):
        raise ValueError(f"{time_source} timestamps are not monotonic")

    if "dq" in rows[0]:
        dq = np.vstack([parse_vec(row["dq"], expected=7) for row in rows])
    else:
        q = np.vstack([parse_vec(row["q"], expected=7) for row in rows])
        dq = np.gradient(q, rel_times, axis=0, edge_order=1)
    if "ddq" in rows[0]:
        ddq = np.vstack([parse_vec(row["ddq"], expected=7) for row in rows])
    else:
        ddq = np.gradient(dq, rel_times, axis=0, edge_order=1)

    samples = [
        JointSample(float(t), parse_vec(row["q"], expected=7), dq_i.copy(), ddq_i.copy())
        for t, row, dq_i, ddq_i in zip(rel_times, rows, dq, ddq)
    ]
    return samples, rows[0]


def moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values.copy()
    if window_samples % 2 == 0:
        window_samples += 1
    pad = window_samples // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window_samples, dtype=float) / window_samples
    smoothed = np.empty_like(values, dtype=float)
    for col in range(values.shape[1]):
        smoothed[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return smoothed


def smooth_joint_samples(samples: list[JointSample], window_sec: float) -> list[JointSample]:
    if window_sec <= 0 or len(samples) < 3:
        return samples
    times = np.asarray([sample.t for sample in samples], dtype=float)
    median_dt = float(np.median(np.diff(times)))
    if median_dt <= 0:
        return samples
    window_samples = max(1, int(round(window_sec / median_dt)))
    q = moving_average(np.vstack([sample.q for sample in samples]), window_samples)
    q[0] = samples[0].q
    q[-1] = samples[-1].q
    smoothed = [
        JointSample(float(t), q_i.copy(), np.zeros(7), np.zeros(7))
        for t, q_i in zip(times, q)
    ]
    return derive_joint_derivatives(smoothed)


def downsample_joint_samples(samples: list[JointSample], output_hz: float | None) -> list[JointSample]:
    if output_hz is None:
        return samples
    if output_hz <= 0:
        raise ValueError("--downsample-hz must be positive")
    if len(samples) < 2:
        return samples
    times = np.asarray([sample.t for sample in samples], dtype=float)
    q = np.vstack([sample.q for sample in samples])
    period = 1.0 / output_hz
    new_times = np.arange(0.0, times[-1], period)
    if len(new_times) == 0 or not math.isclose(float(new_times[-1]), float(times[-1]), abs_tol=1e-9):
        new_times = np.append(new_times, times[-1])
    new_q = np.column_stack([np.interp(new_times, times, q[:, j]) for j in range(q.shape[1])])
    downsampled = [
        JointSample(float(t), q_i.copy(), np.zeros(7), np.zeros(7))
        for t, q_i in zip(new_times, new_q)
    ]
    return derive_joint_derivatives(downsampled)


def shape_joint_amplitude(
    samples: list[JointSample],
    amplitude: float,
    center: str,
    per_joint: np.ndarray | None,
) -> list[JointSample]:
    if amplitude == 1.0 and per_joint is None:
        return samples
    if amplitude <= 0:
        raise ValueError("--joint-amplitude must be positive")
    q_ref = samples[0].q if center == "start" else DEFAULT_JOINTS
    gains = amplitude * np.ones(7)
    if per_joint is not None:
        if per_joint.shape != (7,):
            raise ValueError("--joint-amplitude-vector must contain 7 values")
        gains *= per_joint
    shaped = [
        JointSample(
            sample.t,
            q_ref + gains * (sample.q - q_ref),
            gains * sample.dq,
            gains * sample.ddq,
        )
        for sample in samples
    ]
    return shaped


def shape_joint_feedforward(
    samples: list[JointSample],
    velocity_gain: float,
    acceleration_gain: float,
) -> list[JointSample]:
    if velocity_gain == 1.0 and acceleration_gain == 1.0:
        return samples
    if velocity_gain < 0 or acceleration_gain < 0:
        raise ValueError("--joint-velocity-gain and --joint-acceleration-gain must be non-negative")
    return [
        JointSample(
            sample.t,
            sample.q,
            velocity_gain * sample.dq,
            acceleration_gain * sample.ddq,
        )
        for sample in samples
    ]


def shape_cartesian_samples(
    samples: list[PoseSample],
    amplitude: float,
    lock_axis: str | None,
) -> list[PoseSample]:
    if amplitude == 1.0 and lock_axis is None:
        return samples
    if amplitude <= 0:
        raise ValueError("--cartesian-amplitude must be positive")
    axis_index = None
    if lock_axis is not None:
        axis_index = {"x": 0, "y": 1, "z": 2}[lock_axis]
    reference = samples[0].position
    shaped: list[PoseSample] = []
    for sample in samples:
        position = reference + amplitude * (sample.position - reference)
        if axis_index is not None:
            position[axis_index] = reference[axis_index]
        shaped.append(PoseSample(sample.t, position, sample.orientation))
    return shaped


def apply_orientation_mode(
    samples: list[PoseSample],
    first_row: dict[str, str],
    orientation_mode: str,
) -> tuple[list[PoseSample], str]:
    if orientation_mode == "rdk":
        return samples, "using RDK flange_pose orientation directly"

    if orientation_mode == "task-offset":
        raw_orientation0 = samples[0].orientation
        task_orientation0 = parse_json_matrix(first_row["cartesian_task_current_orientation"], (3, 3))
        offset = raw_orientation0.T @ task_orientation0
        corrected = [
            PoseSample(sample.t, sample.position, sample.orientation @ offset)
            for sample in samples
        ]
        offset_angle = math.degrees(rotation_angle(task_orientation0, raw_orientation0))
        return corrected, f"applied RDK-to-OpenSai task orientation offset ({offset_angle:.2f} deg)"

    if orientation_mode == "hold-start":
        if first_row.get("cartesian_task_current_orientation"):
            held_orientation = parse_json_matrix(first_row["cartesian_task_current_orientation"], (3, 3))
        else:
            held_orientation = samples[0].orientation
        held = [PoseSample(sample.t, sample.position, held_orientation) for sample in samples]
        return held, "holding the starting OpenSai task orientation"

    raise ValueError(f"Unknown orientation mode {orientation_mode!r}")


def get_json_array(client: redis.Redis, key: str) -> np.ndarray | None:
    raw = client.get(key)
    if raw is None:
        return None
    return np.array(json.loads(raw), dtype=float)


def wait_for_json_array(
    client: redis.Redis,
    key: str,
    timeout: float,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        value = get_json_array(client, key)
        if value is not None and (shape is None or value.shape == shape):
            return value
        time.sleep(0.01)
    shape_text = "" if shape is None else f" with shape {shape}"
    raise TimeoutError(f"Timed out waiting for Redis key {key}{shape_text}")


def set_json_array(client: redis.Redis, key: str, value: np.ndarray) -> None:
    client.set(key, json.dumps(np.asarray(value, dtype=float).tolist()))


def read_current_joints(client: redis.Redis, keys: RedisKeys) -> np.ndarray | None:
    for key in (keys.joint_current_sensor, keys.joint_current_controller):
        value = get_json_array(client, key)
        if value is not None and value.shape == (7,):
            return value
    return None


def set_joint_command(
    client: redis.Redis,
    keys: RedisKeys,
    position: np.ndarray,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
) -> None:
    zero = np.zeros_like(position)
    set_json_array(client, keys.joint_goal_position, position)
    set_json_array(client, keys.joint_goal_velocity, zero if velocity is None else velocity)
    set_json_array(client, keys.joint_goal_acceleration, zero if acceleration is None else acceleration)


def set_joint_goal(client: redis.Redis, keys: RedisKeys, position: np.ndarray) -> None:
    set_joint_command(client, keys, position)


def set_cartesian_goal(
    client: redis.Redis,
    keys: RedisKeys,
    position: np.ndarray,
    orientation: np.ndarray,
) -> None:
    set_json_array(client, keys.cartesian_goal_position, position)
    set_json_array(client, keys.cartesian_goal_orientation, orientation.reshape(3, 3))


def set_active_controller(
    client: redis.Redis,
    keys: RedisKeys,
    controller_name: str,
    timeout: float,
) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        client.set(keys.active_controller, controller_name)
        current = client.get(keys.active_controller)
        if current == controller_name:
            return
        time.sleep(0.01)
    raise TimeoutError(f"Timed out switching to {controller_name}")


def check_config(
    client: redis.Redis,
    keys: RedisKeys,
    expected_config: str,
    skip_check: bool,
) -> None:
    if skip_check:
        return
    current = client.get(keys.config_file_name)
    if current != expected_config:
        raise RuntimeError(
            f"Expected OpenSai config {expected_config!r}, but Redis has "
            f"{current!r}. Launch OpenSai with {expected_config} or pass "
            "--skip-config-check."
        )


def move_to_default_joints(
    client: redis.Redis,
    keys: RedisKeys,
    rate: float,
    max_step_deg: float,
    hold_time: float,
    wait_for_detection: bool,
    threshold: float,
    settle_count: int,
    timeout: float,
) -> None:
    set_active_controller(client, keys, keys.joint_controller_name, timeout=5.0)
    current = read_current_joints(client, keys)
    if current is None:
        deadline = time.perf_counter() + min(timeout, 1.0)
        while current is None and time.perf_counter() < deadline:
            current = read_current_joints(client, keys)
            time.sleep(0.01)

    if current is None:
        print("Warning: could not read current joints; commanding the default pose directly.")
        command = DEFAULT_JOINTS.copy()
    else:
        command = current.copy()

    max_step = math.radians(max_step_deg)
    period = 1.0 / rate
    deadline = time.perf_counter() + (timeout if wait_for_detection else hold_time)
    stable = 0

    while time.perf_counter() < deadline:
        delta = DEFAULT_JOINTS - command
        distance = float(np.linalg.norm(delta))
        if distance > max_step:
            delta *= max_step / distance
        command = command + delta
        set_joint_goal(client, keys, command)

        if wait_for_detection:
            current = read_current_joints(client, keys)
            if current is not None:
                error = float(np.linalg.norm(DEFAULT_JOINTS - current))
                if error < threshold:
                    stable += 1
                    if stable >= settle_count:
                        return
                else:
                    stable = 0

        time.sleep(period)

    set_joint_goal(client, keys, DEFAULT_JOINTS)
    if wait_for_detection:
        raise TimeoutError("Timed out parking at the default joint pose from main.py")


def read_current_cartesian(
    client: redis.Redis,
    keys: RedisKeys,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = wait_for_json_array(
        client,
        keys.cartesian_current_position,
        timeout=timeout,
        shape=(3,),
    )
    orientation = wait_for_json_array(
        client,
        keys.cartesian_current_orientation,
        timeout=timeout,
        shape=(3, 3),
    )
    return position, orientation


def prealign_to_first_sample(
    client: redis.Redis,
    keys: RedisKeys,
    target: PoseSample,
    rate: float,
    max_linear_speed: float,
    max_angular_speed: float,
    min_duration: float,
    timeout: float,
) -> None:
    set_active_controller(client, keys, keys.cartesian_controller_name, timeout=5.0)
    time.sleep(0.2)
    start_position, start_orientation = read_current_cartesian(client, keys, timeout=timeout)
    set_cartesian_goal(client, keys, start_position, start_orientation)

    distance = float(np.linalg.norm(target.position - start_position))
    angle = rotation_angle(target.orientation, start_orientation)
    duration = max(
        min_duration,
        distance / max_linear_speed if max_linear_speed > 0 else 0.0,
        angle / max_angular_speed if max_angular_speed > 0 else 0.0,
    )
    steps = max(1, int(math.ceil(duration * rate)))
    start_quat = matrix_to_quat_wxyz(start_orientation)
    target_quat = matrix_to_quat_wxyz(target.orientation)
    start_time = time.perf_counter()

    for step in range(steps + 1):
        alpha = step / steps
        position = (1.0 - alpha) * start_position + alpha * target.position
        orientation = quat_wxyz_to_matrix(slerp(start_quat, target_quat, alpha))
        set_cartesian_goal(client, keys, position, orientation)
        next_time = start_time + (step + 1) / rate
        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    set_cartesian_goal(client, keys, target.position, target.orientation)


def prealign_to_first_joint_sample(
    client: redis.Redis,
    keys: RedisKeys,
    target: JointSample,
    rate: float,
    max_step_deg: float,
    min_duration: float,
) -> None:
    current = read_current_joints(client, keys)
    if current is None:
        current = DEFAULT_JOINTS.copy()
    max_step = math.radians(max_step_deg)
    max_delta = float(np.max(np.abs(target.q - current)))
    steps = max(1, int(math.ceil(max_delta / max_step)), int(math.ceil(min_duration * rate)))
    start_time = time.perf_counter()

    for step in range(steps + 1):
        alpha = step / steps
        q = (1.0 - alpha) * current + alpha * target.q
        set_joint_command(client, keys, q)
        next_time = start_time + (step + 1) / rate
        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    set_joint_goal(client, keys, target.q)


def joint_feedforward(sample: JointSample, mode: str, max_acceleration: float | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mode == "none":
        return None, None
    if mode == "velocity":
        return sample.dq, None
    if mode == "velocity-acceleration":
        acceleration = sample.ddq
        if max_acceleration is not None:
            acceleration = np.clip(acceleration, -max_acceleration, max_acceleration)
        return sample.dq, acceleration
    raise ValueError(f"Unknown joint feedforward mode {mode!r}")


def replay_joint_samples(
    client: redis.Redis,
    keys: RedisKeys,
    samples: list[JointSample],
    speed: float,
    status_period: float,
    feedforward_mode: str,
    max_acceleration: float | None,
) -> None:
    start = time.perf_counter()
    last_status = start

    for index, sample in enumerate(samples):
        target_time = start + sample.t / speed
        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        velocity, acceleration = joint_feedforward(sample, feedforward_mode, max_acceleration)
        if velocity is not None:
            velocity = velocity * speed
        if acceleration is not None:
            acceleration = acceleration * speed * speed
        set_joint_command(client, keys, sample.q, velocity=velocity, acceleration=acceleration)

        now = time.perf_counter()
        if status_period > 0 and now - last_status >= status_period:
            print(
                f"Joint replay {index + 1}/{len(samples)} "
                f"t={sample.t:.3f}s q_deg={np.round(np.degrees(sample.q), 2).tolist()}"
            )
            last_status = now

    set_joint_goal(client, keys, samples[-1].q)


def replay_samples(
    client: redis.Redis,
    keys: RedisKeys,
    samples: list[PoseSample],
    speed: float,
    status_period: float,
) -> None:
    start = time.perf_counter()
    last_status = start

    for index, sample in enumerate(samples):
        target_time = start + sample.t / speed
        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        set_cartesian_goal(client, keys, sample.position, sample.orientation)

        now = time.perf_counter()
        if status_period > 0 and now - last_status >= status_period:
            print(
                f"Replay {index + 1}/{len(samples)} "
                f"t={sample.t:.3f}s pos={np.round(sample.position, 4).tolist()}"
            )
            last_status = now

    last = samples[-1]
    set_cartesian_goal(client, keys, last.position, last.orientation)


def summarize(samples: list[PoseSample], first_row: dict[str, str], pose_column: str) -> None:
    positions = np.vstack([sample.position for sample in samples])
    duration = samples[-1].t - samples[0].t
    dt = np.diff([sample.t for sample in samples])
    hz = (len(samples) - 1) / duration if duration > 0 else 0.0
    start_orientation = samples[0].orientation
    orientation_deltas = np.array(
        [rotation_angle(sample.orientation, start_orientation) for sample in samples]
    )

    print(f"Log pose column: {pose_column}")
    print(f"Robot in log: {first_row.get('robot_name', '')} {first_row.get('robot_sn', '')}")
    print(f"Primitive in log: {first_row.get('primitive', '')}")
    print(f"Samples: {len(samples)}")
    print(f"Duration: {duration:.3f} s")
    print(f"Median dt: {np.median(dt):.6f} s ({hz:.2f} Hz average)")
    print(f"Start position: {np.round(positions[0], 6).tolist()}")
    print(f"End position: {np.round(positions[-1], 6).tolist()}")
    print(f"Position span: {np.round(np.ptp(positions, axis=0), 6).tolist()} m")
    print(f"Max orientation change: {math.degrees(float(np.max(orientation_deltas))):.2f} deg")


def summarize_joint(samples: list[JointSample], first_row: dict[str, str]) -> None:
    q = np.vstack([sample.q for sample in samples])
    dq = np.vstack([sample.dq for sample in samples])
    ddq = np.vstack([sample.ddq for sample in samples])
    times = np.asarray([sample.t for sample in samples], dtype=float)
    duration = times[-1] - times[0]
    dt = np.diff(times)
    hz = (len(samples) - 1) / duration if duration > 0 else 0.0
    max_step_deg = np.degrees(np.max(np.abs(np.diff(q, axis=0)), axis=0)) if len(samples) > 1 else np.zeros(7)

    print("Replay mode: joint")
    print(f"Robot in log: {first_row.get('robot_name', '')} {first_row.get('robot_sn', '')}")
    print(f"Primitive in log: {first_row.get('primitive') or first_row.get('source_primitive', '')}")
    print(f"Samples: {len(samples)}")
    print(f"Duration: {duration:.3f} s")
    print(f"Median dt: {np.median(dt):.6f} s ({hz:.2f} Hz average)")
    print(f"Start q deg: {np.round(np.degrees(q[0]), 3).tolist()}")
    print(f"End q deg: {np.round(np.degrees(q[-1]), 3).tolist()}")
    print(f"Joint span deg: {np.round(np.degrees(np.ptp(q, axis=0)), 3).tolist()}")
    print(f"Max per-sample step deg: {np.round(max_step_deg, 4).tolist()}")
    print(f"Peak dq rad/s: {np.round(np.max(np.abs(dq), axis=0), 3).tolist()}")
    print(f"Peak ddq rad/s^2: {np.round(np.max(np.abs(ddq), axis=0), 2).tolist()}")


def write_joint_trajectory(path: Path, samples: list[JointSample], first_row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "robot_name",
                "robot_sn",
                "source_primitive",
                "q",
                "dq",
                "ddq",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "t": f"{sample.t:.9f}",
                    "robot_name": first_row.get("robot_name", ""),
                    "robot_sn": first_row.get("robot_sn", ""),
                    "source_primitive": first_row.get("primitive") or first_row.get("source_primitive", ""),
                    "q": " ".join(f"{x:.10g}" for x in sample.q),
                    "dq": " ".join(f"{x:.10g}" for x in sample.dq),
                    "ddq": " ".join(f"{x:.10g}" for x in sample.ddq),
                }
            )


def write_cartesian_trajectory(path: Path, samples: list[PoseSample], first_row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "robot_name",
                "robot_sn",
                "source_primitive",
                "position",
                "orientation",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "t": f"{sample.t:.9f}",
                    "robot_name": first_row.get("robot_name", ""),
                    "robot_sn": first_row.get("robot_sn", ""),
                    "source_primitive": first_row.get("primitive", ""),
                    "position": " ".join(f"{x:.10g}" for x in sample.position),
                    "orientation": json.dumps(sample.orientation.tolist()),
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay logs/latest.csv on the Rizon through OpenSai Redis goals."
    )
    parser.add_argument("logfile", nargs="?", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--robot-name", help="Defaults to the robot_name stored in the log.")
    parser.add_argument("--joint-controller", default="joint_controller")
    parser.add_argument("--cartesian-controller", default="cartesian_controller")
    parser.add_argument("--expected-config", default=DEFAULT_EXPECTED_CONFIG)
    parser.add_argument("--skip-config-check", action="store_true")
    parser.add_argument("--mode", default="joint", choices=("joint", "cartesian"))
    parser.add_argument("--pose-column", default="flange_pose", choices=("flange_pose", "tcp_pose"))
    parser.add_argument(
        "--orientation-mode",
        default="task-offset",
        choices=("task-offset", "rdk", "hold-start"),
        help="Cartesian-only orientation handling. task-offset converts RDK flange orientation into OpenSai task orientation.",
    )
    parser.add_argument("--time-source", default="robot", choices=("robot", "wall"))
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 replays the original timing.")
    parser.add_argument("--smooth-window", type=float, default=0.0, help="Joint-mode moving-average window in seconds.")
    parser.add_argument("--downsample-hz", type=float, help="Joint-mode output rate after interpolation.")
    parser.add_argument("--joint-amplitude", type=float, default=1.0, help="Scale joint motion away from the reference pose.")
    parser.add_argument(
        "--joint-amplitude-center",
        default="start",
        choices=("start", "default"),
        help="Reference pose for joint amplitude scaling.",
    )
    parser.add_argument(
        "--joint-amplitude-vector",
        help="Optional 7-value per-joint multiplier applied with --joint-amplitude.",
    )
    parser.add_argument("--joint-velocity-gain", type=float, default=1.0)
    parser.add_argument("--joint-acceleration-gain", type=float, default=1.0)
    parser.add_argument(
        "--joint-feedforward",
        default="velocity",
        choices=("none", "velocity", "velocity-acceleration"),
        help="Joint-mode feed-forward terms sent with each q sample.",
    )
    parser.add_argument(
        "--max-goal-acceleration",
        type=float,
        default=15.0,
        help="Clip joint feed-forward acceleration in rad/s^2. Use a negative value to disable clipping.",
    )
    parser.add_argument("--cartesian-amplitude", type=float, default=1.0)
    parser.add_argument(
        "--cartesian-lock-axis",
        choices=("x", "y", "z"),
        help="Cartesian-only: keep this world coordinate fixed at the first sample.",
    )
    parser.add_argument("--save-trajectory", type=Path, help="Write the generated replay trajectory to a new CSV.")
    parser.add_argument("--execute", action="store_true", help="Actually command Redis goals.")
    parser.add_argument("--yes", action="store_true", help="Skip the execute confirmation prompt.")
    parser.add_argument("--no-prealign", action="store_true", help="Stream the log immediately after switching.")
    parser.add_argument("--parking-rate", type=float, default=100.0)
    parser.add_argument("--parking-step-deg", type=float, default=0.5)
    parser.add_argument(
        "--default-hold-time",
        type=float,
        default=10.0,
        help="Seconds to command main.py's default joint pose before switching to Cartesian.",
    )
    parser.add_argument(
        "--wait-for-default-detection",
        action="store_true",
        help="Use measured joint error instead of fixed-time parking.",
    )
    parser.add_argument("--joint-threshold", type=float, default=0.05)
    parser.add_argument("--joint-settle-count", type=int, default=20)
    parser.add_argument("--parking-timeout", type=float, default=45.0)
    parser.add_argument("--align-rate", type=float, default=200.0)
    parser.add_argument("--align-max-linear-speed", type=float, default=0.03)
    parser.add_argument("--align-max-angular-speed", type=float, default=0.4)
    parser.add_argument("--align-min-duration", type=float, default=0.5)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--status-period", type=float, default=1.0)
    parser.add_argument(
        "--return-default-on-stop",
        action="store_true",
        help="Move back to main.py's default joint pose after normal completion or Ctrl+C.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    if args.parking_rate <= 0 or args.align_rate <= 0:
        raise ValueError("--parking-rate and --align-rate must be positive")
    if args.default_hold_time < 0:
        raise ValueError("--default-hold-time must be non-negative")
    if args.smooth_window < 0:
        raise ValueError("--smooth-window must be non-negative")
    if args.max_goal_acceleration is not None and args.max_goal_acceleration < 0:
        args.max_goal_acceleration = None
    joint_amplitude_vector = parse_vec(args.joint_amplitude_vector, expected=7) if args.joint_amplitude_vector else None

    if args.mode == "joint":
        samples, first_row = load_joint_trajectory(args.logfile, args.time_source)
        samples = smooth_joint_samples(samples, args.smooth_window)
        samples = downsample_joint_samples(samples, args.downsample_hz)
        samples = shape_joint_amplitude(
            samples,
            args.joint_amplitude,
            args.joint_amplitude_center,
            joint_amplitude_vector,
        )
        samples = shape_joint_feedforward(
            samples,
            args.joint_velocity_gain,
            args.joint_acceleration_gain,
        )
        summarize_joint(samples, first_row)
        if args.save_trajectory:
            write_joint_trajectory(args.save_trajectory, samples, first_row)
            print(f"Wrote generated joint trajectory to {args.save_trajectory}")
    else:
        samples, first_row = load_trajectory(args.logfile, args.pose_column, args.time_source)
        samples, orientation_note = apply_orientation_mode(samples, first_row, args.orientation_mode)
        samples = shape_cartesian_samples(samples, args.cartesian_amplitude, args.cartesian_lock_axis)
        summarize(samples, first_row, args.pose_column)
        print(f"Orientation handling: {orientation_note}")
        if args.cartesian_lock_axis:
            print(f"Cartesian projection: locked world {args.cartesian_lock_axis}")
        if args.cartesian_amplitude != 1.0:
            print(f"Cartesian amplitude: {args.cartesian_amplitude}")
        if args.save_trajectory:
            write_cartesian_trajectory(args.save_trajectory, samples, first_row)
            print(f"Wrote generated Cartesian trajectory to {args.save_trajectory}")

    robot_name = args.robot_name or first_row.get("robot_name") or "Titania"
    keys = make_keys(robot_name, args.joint_controller, args.cartesian_controller)

    print(f"Replay robot: {robot_name}")
    print(f"OpenSai config check: {args.expected_config!r}")
    print(f"Active controller key: {keys.active_controller}")
    if args.mode == "joint":
        print(f"Joint goal key: {keys.joint_goal_position}")
        if args.joint_amplitude != 1.0 or joint_amplitude_vector is not None:
            print(f"Joint amplitude: {args.joint_amplitude} around {args.joint_amplitude_center}")
            if joint_amplitude_vector is not None:
                print(f"Joint amplitude vector: {joint_amplitude_vector.tolist()}")
        if args.joint_velocity_gain != 1.0 or args.joint_acceleration_gain != 1.0:
            print(f"Joint feed-forward gains: velocity={args.joint_velocity_gain}, acceleration={args.joint_acceleration_gain}")
        print(f"Joint feed-forward: {args.joint_feedforward}")
        if args.joint_feedforward == "velocity-acceleration":
            print(f"Max goal acceleration: {args.max_goal_acceleration}")
    else:
        print(f"Cartesian goal keys: {keys.cartesian_goal_position}, {keys.cartesian_goal_orientation}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to command the robot.")
        return

    if not args.yes:
        response = input("Type 'yes' to move the robot with this trajectory: ")
        if response.strip().lower() != "yes":
            print("Cancelled.")
            return

    client = redis.Redis(host=args.host, port=args.port, decode_responses=True)
    check_config(client, keys, args.expected_config, args.skip_config_check)

    try:
        print("Parking at main.py default joint pose.")
        move_to_default_joints(
            client,
            keys,
            rate=args.parking_rate,
            max_step_deg=args.parking_step_deg,
            hold_time=args.default_hold_time,
            wait_for_detection=args.wait_for_default_detection,
            threshold=args.joint_threshold,
            settle_count=args.joint_settle_count,
            timeout=args.parking_timeout,
        )

        if args.mode == "joint":
            print("Replaying logged joint motion.")
            prealign_to_first_joint_sample(
                client,
                keys,
                samples[0],
                rate=args.parking_rate,
                max_step_deg=args.parking_step_deg,
                min_duration=0.5,
            )
            replay_joint_samples(
                client,
                keys,
                samples,
                speed=args.speed,
                status_period=args.status_period,
                feedforward_mode=args.joint_feedforward,
                max_acceleration=args.max_goal_acceleration,
            )
            print("Replay complete; holding final joint target.")
        else:
            print("Switching to Cartesian control.")
            set_active_controller(client, keys, keys.cartesian_controller_name, timeout=5.0)

            if args.no_prealign:
                first = samples[0]
                set_cartesian_goal(client, keys, first.position, first.orientation)
            else:
                print("Pre-aligning to the first logged flange pose.")
                prealign_to_first_sample(
                    client,
                    keys,
                    samples[0],
                    rate=args.align_rate,
                    max_linear_speed=args.align_max_linear_speed,
                    max_angular_speed=args.align_max_angular_speed,
                    min_duration=args.align_min_duration,
                    timeout=args.state_timeout,
                )

            print("Replaying logged Cartesian motion.")
            replay_samples(
                client,
                keys,
                samples,
                speed=args.speed,
                status_period=args.status_period,
            )
            print("Replay complete; holding final Cartesian target.")

    except KeyboardInterrupt:
        print("\nInterrupted; holding the last commanded target.")
    finally:
        if args.return_default_on_stop:
            print("Returning to main.py default joint pose.")
            move_to_default_joints(
                client,
                keys,
                rate=args.parking_rate,
                max_step_deg=args.parking_step_deg,
                hold_time=args.default_hold_time,
                wait_for_detection=args.wait_for_default_detection,
                threshold=args.joint_threshold,
                settle_count=args.joint_settle_count,
                timeout=args.parking_timeout,
            )


if __name__ == "__main__":
    main()
