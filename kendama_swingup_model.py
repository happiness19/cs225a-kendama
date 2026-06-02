"""Explicit flange-space kendama swing-up and catch model.

This script is the bridge between the simple joint oscillation in
``rising_dragon_oscillate.py`` and a physics model we can tune.

The model separates three things:

1. A fixed flange-to-kendama geometry:

       p_anchor = p_flange + R_flange @ r_flange_anchor
       p_cup    = p_flange + R_flange @ r_flange_cup

2. A taut-string pendulum swing-up phase driven by anchor acceleration:

       theta_ddot = ((g - a_anchor) dot e_theta) / L
       E_dot      = -L * theta_dot * (a_anchor dot e_theta)

3. A catch phase that predicts where the falling ball crosses the cup plane
   and commands the cup center under that crossing.

By default the script runs an offline dry-run simulation. It reads Redis only
when ``--redis`` or ``--command`` is passed, and it only writes Redis cartesian
goals when ``--command`` is passed.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable

import numpy as np
import redis


GRAVITY_MAG = 9.81
GRAVITY = np.array([0.0, 0.0, -GRAVITY_MAG])
WORLD_UP = np.array([0.0, 0.0, 1.0])
PHYSICAL_CUP_OFFSET_F = np.array([0.0, 0.0, 0.1143])
PARALLEL_URDF_CUP_OFFSET_F = np.array([0.0, -0.04, 0.20])


class Phase(Enum):
    SWING_UP = auto()
    TRACK_CATCH = auto()
    DAMP = auto()


@dataclass
class RedisKeys:
    robot_name: str

    @property
    def prefix(self) -> str:
        return f"opensai::controllers::{self.robot_name}"

    @property
    def active_controller(self) -> str:
        return f"{self.prefix}::active_controller_name"

    @property
    def cartesian_goal_position(self) -> str:
        return f"{self.prefix}::cartesian_controller::cartesian_task::goal_position"

    @property
    def cartesian_goal_orientation(self) -> str:
        return f"{self.prefix}::cartesian_controller::cartesian_task::goal_orientation"

    @property
    def flange_transform(self) -> str:
        return f"opensai::sensors::{self.robot_name}::flange_transform"


@dataclass
class Transform:
    position: np.ndarray
    rotation: np.ndarray


@dataclass
class KendamaGeometry:
    """Offsets are expressed in the flange frame."""

    cup_offset_f: np.ndarray = field(
        default_factory=lambda: PHYSICAL_CUP_OFFSET_F.copy()
    )
    anchor_offset_f: np.ndarray = field(
        default_factory=lambda: PHYSICAL_CUP_OFFSET_F.copy()
    )
    ball_center_above_cup: float = 0.039


@dataclass
class SwingParams:
    string_length: float = 0.40
    max_string_length_error: float = 0.15
    max_ball_speed: float = 8.0
    energy_fraction: float = 0.92
    pump_gain: float = 2.2
    max_anchor_accel: float = 5.0
    max_anchor_speed: float = 0.45
    max_anchor_offset: float = 0.12
    recenter_kp: float = 8.0
    recenter_kd: float = 3.0
    catch_xy_window: float = 0.045
    catch_height_window: float = 0.08
    catch_time_horizon: float = 0.90
    catch_target_timeout: float = 0.20
    damp_drop: float = 0.025
    damp_time: float = 0.25


@dataclass
class PendulumState:
    theta: float
    theta_dot: float


@dataclass
class CatchTarget:
    cup_center_w: np.ndarray
    crossing_time: float
    updated_at: float


@dataclass
class BallStateValidity:
    valid: bool
    reason: str
    string_length_error: float | None
    ball_speed: float | None


@dataclass
class ModelSnapshot:
    flange: Transform | None
    cup_w: np.ndarray | None
    anchor_w: np.ndarray | None
    ball_w: np.ndarray | None
    ball_velocity_w: np.ndarray | None
    ball_velocity_estimated: bool
    ball_anchor_distance: float | None
    string_length_error: float | None
    ball_state_validity: BallStateValidity | None
    pendulum: PendulumState | None
    energy: float | None
    target_energy: float | None
    catch: tuple[np.ndarray, float] | None
    missing_keys: list[str]


@dataclass
class BallPoseCandidate:
    key: str
    available: bool
    position_w: np.ndarray | None = None
    cup_distance: float | None = None
    anchor_distance: float | None = None
    string_length_error: float | None = None
    parse_error: str | None = None


class MemoryRedis:
    """Small Redis-compatible store for offline fixture checks."""

    def __init__(self) -> None:
        self.values = {}

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value) -> None:
        self.values[key] = value

    def scan_iter(self, match: str | None = None):
        for key in self.values:
            if match is None or fnmatch.fnmatch(key, match):
                yield key


def parse_vec(text: str, expected: int | None = None) -> np.ndarray:
    values = np.array([float(x.strip()) for x in text.split(",")], dtype=float)
    if expected is not None and len(values) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {len(values)}")
    return values


def split_key_candidates(keys_text: str) -> list[str]:
    return [key.strip() for key in keys_text.split(",") if key.strip()]


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def redis_key_to_text(key: bytes | str) -> str:
    return key.decode("utf-8") if isinstance(key, bytes) else key


def get_first_available(client, keys_text: str) -> tuple[object | None, str | None, list[str]]:
    candidates = split_key_candidates(keys_text)
    for key in candidates:
        raw = client.get(key)
        if raw is not None:
            return raw, key, candidates
    return None, None, candidates


def estimate_velocity_from_pose_samples(
    client,
    ball_pose_key: str,
    sample_count: int,
    sample_period: float,
    sleep_fn=time.sleep,
) -> np.ndarray | None:
    positions = []
    times = []
    sample_count = max(2, sample_count)
    for sample_index in range(sample_count):
        raw_ball, _, _ = get_first_available(client, ball_pose_key)
        if raw_ball is None:
            return None
        positions.append(parse_pose_position(raw_ball))
        times.append(sample_index * sample_period)
        if sample_index + 1 < sample_count:
            sleep_fn(sample_period)

    positions_arr = np.vstack(positions)
    times_arr = np.array(times)
    centered_t = times_arr - times_arr.mean()
    denom = float(np.dot(centered_t, centered_t))
    if denom < 1e-12:
        return None
    centered_p = positions_arr - positions_arr.mean(axis=0)
    return centered_t @ centered_p / denom


def normed(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return fallback.copy()
    return vec / norm


def clamp_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if limit <= 0.0 or norm <= limit or norm < 1e-12:
        return vec
    return vec * (limit / norm)


def quat_xyzw_to_matrix(quat: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("Quaternion has near-zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def parse_transform_value(raw: bytes | str) -> Transform:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    obj = json.loads(text)
    return parse_transform_object(obj)


def parse_transform_object(obj) -> Transform:
    if isinstance(obj, dict):
        for key in ("transform", "matrix", "T"):
            if key in obj:
                return parse_transform_object(obj[key])

        position = obj.get("position") or obj.get("translation") or obj.get("pos")
        rotation = obj.get("rotation") or obj.get("orientation") or obj.get("quaternion")
        if position is not None and rotation is not None:
            p = np.array(position, dtype=float)
            r = np.array(rotation, dtype=float)
            if r.shape == (3, 3):
                return Transform(p, r)
            if r.size == 9:
                return Transform(p, r.reshape(3, 3))
            if r.size == 4:
                return Transform(p, quat_xyzw_to_matrix(r))

    arr = np.array(obj, dtype=float)
    if arr.shape == (4, 4):
        return Transform(arr[:3, 3].copy(), arr[:3, :3].copy())
    if arr.shape == (3, 4):
        return Transform(arr[:3, 3].copy(), arr[:3, :3].copy())
    if arr.size == 16:
        mat = arr.reshape(4, 4)
        return Transform(mat[:3, 3].copy(), mat[:3, :3].copy())
    if arr.size == 12:
        mat = arr.reshape(3, 4)
        return Transform(mat[:3, 3].copy(), mat[:3, :3].copy())
    if arr.size == 7:
        flat = arr.reshape(-1)
        return Transform(flat[:3].copy(), quat_xyzw_to_matrix(flat[3:7]))

    raise ValueError(f"Cannot parse transform payload with shape {arr.shape}")


def parse_pose_position(raw: bytes | str) -> np.ndarray:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    obj = json.loads(text)
    if isinstance(obj, dict):
        if "position" in obj:
            return np.array(obj["position"], dtype=float)
        if "pose" in obj:
            return parse_pose_position(json.dumps(obj["pose"]))
    arr = np.array(obj, dtype=float)
    if arr.shape == (4, 4):
        return arr[:3, 3].copy()
    if arr.size == 16:
        return arr.reshape(4, 4)[:3, 3].copy()
    if arr.size >= 3:
        return arr.reshape(-1)[:3].copy()
    raise ValueError("Could not parse pose position")


def parse_velocity(raw: bytes | str) -> np.ndarray:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    arr = np.array(json.loads(text), dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError("Velocity payload has fewer than three elements")
    return arr[:3].copy()


def transform_payload(transform: Transform) -> str:
    matrix = np.eye(4)
    matrix[:3, :3] = transform.rotation
    matrix[:3, 3] = transform.position
    return json.dumps(matrix.tolist())


def pose_payload(position_w: np.ndarray) -> str:
    return json.dumps({"position": np.asarray(position_w).tolist(), "orientation": [0.0, 0.0, 0.0, 1.0]})


def cup_center(transform_wf: Transform, geometry: KendamaGeometry) -> np.ndarray:
    return transform_wf.position + transform_wf.rotation @ geometry.cup_offset_f


def anchor_position(transform_wf: Transform, geometry: KendamaGeometry) -> np.ndarray:
    return transform_wf.position + transform_wf.rotation @ geometry.anchor_offset_f


def flange_position_for_cup(
    cup_target_w: np.ndarray,
    flange_rotation_wf: np.ndarray,
    geometry: KendamaGeometry,
) -> np.ndarray:
    return cup_target_w - flange_rotation_wf @ geometry.cup_offset_f


def flange_position_for_anchor(
    anchor_target_w: np.ndarray,
    flange_rotation_wf: np.ndarray,
    geometry: KendamaGeometry,
) -> np.ndarray:
    return anchor_target_w - flange_rotation_wf @ geometry.anchor_offset_f


def estimate_cup_offset_from_ball(
    transform_wf: Transform,
    ball_position_w: np.ndarray,
    ball_center_above_cup: float,
    cup_normal_w: np.ndarray = WORLD_UP,
) -> np.ndarray:
    cup_normal = normed(cup_normal_w, WORLD_UP)
    cup_center_w = ball_position_w - ball_center_above_cup * cup_normal
    return transform_wf.rotation.T @ (cup_center_w - transform_wf.position)


def estimate_anchor_offset_from_hanging_ball(
    transform_wf: Transform,
    ball_position_w: np.ndarray,
    string_length: float,
    anchor_to_ball_direction_w: np.ndarray = -WORLD_UP,
) -> np.ndarray:
    anchor_to_ball = normed(anchor_to_ball_direction_w, -WORLD_UP)
    anchor_w = ball_position_w - string_length * anchor_to_ball
    return transform_wf.rotation.T @ (anchor_w - transform_wf.position)


def pendulum_frame(swing_axis_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_axis = normed(swing_axis_world - WORLD_UP * np.dot(swing_axis_world, WORLD_UP), np.array([1.0, 0.0, 0.0]))
    return x_axis, WORLD_UP.copy()


def pendulum_units(theta: float, x_axis: np.ndarray, z_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radial = math.sin(theta) * x_axis - math.cos(theta) * z_axis
    tangent = math.cos(theta) * x_axis + math.sin(theta) * z_axis
    return radial, tangent


def pendulum_energy(theta: float, theta_dot: float, string_length: float) -> float:
    return 0.5 * (string_length * theta_dot) ** 2 + GRAVITY_MAG * string_length * (1.0 - math.cos(theta))


def estimate_pendulum_state(
    anchor_w: np.ndarray,
    ball_w: np.ndarray,
    ball_velocity_w: np.ndarray,
    string_length: float,
    swing_axis_world: np.ndarray,
) -> PendulumState:
    x_axis, z_axis = pendulum_frame(swing_axis_world)
    rel = ball_w - anchor_w
    x = float(np.dot(rel, x_axis))
    z = float(np.dot(rel, z_axis))
    theta = math.atan2(x, -z)
    _, tangent = pendulum_units(theta, x_axis, z_axis)
    theta_dot = float(np.dot(ball_velocity_w, tangent) / max(string_length, 1e-6))
    return PendulumState(theta, theta_dot)


def swingup_anchor_accel(
    state: PendulumState,
    params: SwingParams,
    swing_axis_world: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    x_axis, z_axis = pendulum_frame(swing_axis_world)
    _, tangent = pendulum_units(state.theta, x_axis, z_axis)

    energy = pendulum_energy(state.theta, state.theta_dot, params.string_length)
    target_energy = params.energy_fraction * 2.0 * GRAVITY_MAG * params.string_length
    normalized_error = (target_energy - energy) / max(GRAVITY_MAG * params.string_length, 1e-9)

    # Positive energy error means "pump"; choose acceleration opposite theta_dot
    # along e_theta so E_dot = -L * theta_dot * (a dot e_theta) is positive.
    accel_tangent = -params.pump_gain * state.theta_dot * normalized_error
    accel_tangent = float(np.clip(accel_tangent, -params.max_anchor_accel, params.max_anchor_accel))
    return accel_tangent * tangent, energy, target_energy


def step_taut_pendulum(
    state: PendulumState,
    anchor_accel_w: np.ndarray,
    params: SwingParams,
    swing_axis_world: np.ndarray,
    dt: float,
) -> PendulumState:
    x_axis, z_axis = pendulum_frame(swing_axis_world)
    _, tangent = pendulum_units(state.theta, x_axis, z_axis)
    theta_ddot = float(np.dot(GRAVITY - anchor_accel_w, tangent) / params.string_length)
    theta_dot = state.theta_dot + theta_ddot * dt
    theta = wrap_angle(state.theta + theta_dot * dt)
    return PendulumState(theta, theta_dot)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ball_position_from_state(
    anchor_w: np.ndarray,
    state: PendulumState,
    params: SwingParams,
    swing_axis_world: np.ndarray,
) -> np.ndarray:
    x_axis, z_axis = pendulum_frame(swing_axis_world)
    radial, _ = pendulum_units(state.theta, x_axis, z_axis)
    return anchor_w + params.string_length * radial


def ball_anchor_distance(anchor_w: np.ndarray, ball_w: np.ndarray) -> float:
    return float(np.linalg.norm(ball_w - anchor_w))


def ball_anchor_string_error(anchor_w: np.ndarray, ball_w: np.ndarray, string_length: float) -> float:
    return ball_anchor_distance(anchor_w, ball_w) - string_length


def evaluate_ball_state_validity(
    anchor_w: np.ndarray,
    ball_w: np.ndarray,
    ball_velocity_w: np.ndarray | None,
    params: SwingParams,
) -> BallStateValidity:
    if not np.all(np.isfinite(anchor_w)) or not np.all(np.isfinite(ball_w)):
        return BallStateValidity(False, "nonfinite-position", None, None)

    string_error = ball_anchor_string_error(anchor_w, ball_w, params.string_length)
    if not math.isfinite(string_error):
        return BallStateValidity(False, "nonfinite-string-error", None, None)
    if abs(string_error) > params.max_string_length_error:
        return BallStateValidity(False, "string-mismatch", string_error, None)

    ball_speed = None
    if ball_velocity_w is not None:
        if not np.all(np.isfinite(ball_velocity_w)):
            return BallStateValidity(False, "nonfinite-velocity", string_error, None)
        ball_speed = float(np.linalg.norm(ball_velocity_w))
        if not math.isfinite(ball_speed):
            return BallStateValidity(False, "nonfinite-speed", string_error, None)
        if ball_speed > params.max_ball_speed:
            return BallStateValidity(False, "ball-speed-too-high", string_error, ball_speed)

    return BallStateValidity(True, "ok", string_error, ball_speed)


def invalid_ball_state_desired_flange(
    action: str,
    current_flange_goal: np.ndarray,
    base_flange_position: np.ndarray,
) -> np.ndarray:
    if action == "hold":
        return current_flange_goal.copy()
    if action == "recenter":
        return base_flange_position.copy()
    raise ValueError(f"Unknown invalid ball state action: {action}")


def is_pose_like_key_name(key: str) -> bool:
    lower_key = key.lower()
    excluded_tokens = ("ori", "orientation", "velocity", "vel")
    if any(token in lower_key for token in excluded_tokens):
        return False
    return any(token in lower_key for token in ("pos", "pose", "position", "object_pose"))


def discover_pose_keys(
    client,
    patterns_text: str,
    max_scan_keys: int,
) -> list[str]:
    discovered = []
    seen = set()
    scanned_count = 0
    for pattern in split_key_candidates(patterns_text):
        for raw_key in client.scan_iter(match=pattern):
            scanned_count += 1
            if scanned_count > max_scan_keys:
                return discovered
            key = redis_key_to_text(raw_key)
            if key in seen:
                continue
            seen.add(key)
            if not is_pose_like_key_name(key):
                continue
            raw_value = client.get(key)
            if raw_value is None:
                continue
            try:
                position = parse_pose_position(raw_value).reshape(-1)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if position.size == 3 and np.all(np.isfinite(position)):
                discovered.append(key)
    return discovered


def diagnose_ball_pose_candidates(
    client,
    keys_text: str,
    cup_w: np.ndarray | None,
    anchor_w: np.ndarray | None,
    string_length: float,
) -> list[BallPoseCandidate]:
    diagnostics = []
    for key in split_key_candidates(keys_text):
        raw = client.get(key)
        if raw is None:
            diagnostics.append(BallPoseCandidate(key=key, available=False))
            continue

        try:
            position = parse_pose_position(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            diagnostics.append(
                BallPoseCandidate(
                    key=key,
                    available=True,
                    parse_error=str(exc),
                )
            )
            continue

        cup_distance = None
        anchor_distance = None
        string_error = None
        if cup_w is not None:
            cup_distance = float(np.linalg.norm(position - cup_w))
        if anchor_w is not None:
            anchor_distance = ball_anchor_distance(anchor_w, position)
            string_error = anchor_distance - string_length
        diagnostics.append(
            BallPoseCandidate(
                key=key,
                available=True,
                position_w=position,
                cup_distance=cup_distance,
                anchor_distance=anchor_distance,
                string_length_error=string_error,
            )
        )
    return diagnostics


def ball_candidate_keys_text(
    ball_pose_key: str,
    ball_candidate_key: str | None,
    discovered_keys: Iterable[str] | None = None,
) -> str:
    if ball_candidate_key is not None:
        candidates = split_key_candidates(ball_candidate_key)
    else:
        candidates = split_key_candidates(ball_pose_key)
        candidates.extend(["KendamaBallBlue::pos"])
    if discovered_keys is not None:
        candidates.extend(discovered_keys)
    return ",".join(dedupe_preserving_order(candidates))


def format_ball_candidate_diagnostics(
    diagnostics: list[BallPoseCandidate],
    max_string_length_error: float | None = None,
) -> list[str]:
    lines = ["ball_pose_candidates:"]
    if not diagnostics:
        lines.append("  none configured")
        return lines

    for diagnostic in diagnostics:
        if not diagnostic.available:
            lines.append(f"  {diagnostic.key}: missing")
            continue
        if diagnostic.parse_error is not None:
            lines.append(f"  {diagnostic.key}: parse_error={diagnostic.parse_error}")
            continue

        fields = [
            f"position={np.round(diagnostic.position_w, 6).tolist()}",
        ]
        if diagnostic.cup_distance is not None:
            fields.append(f"cup_dist={diagnostic.cup_distance:.6f}")
        if diagnostic.anchor_distance is not None:
            fields.append(f"anchor_dist={diagnostic.anchor_distance:.6f}")
        if diagnostic.string_length_error is not None:
            fields.append(f"string_err={diagnostic.string_length_error:.6f}")
            if max_string_length_error is not None:
                status = (
                    "plausible"
                    if abs(diagnostic.string_length_error) <= max_string_length_error
                    else "string-mismatch"
                )
                fields.append(f"status={status}")
        lines.append(f"  {diagnostic.key}: " + " ".join(fields))
    return lines


def print_ball_candidate_diagnostics(
    diagnostics: list[BallPoseCandidate],
    max_string_length_error: float | None = None,
) -> None:
    for line in format_ball_candidate_diagnostics(diagnostics, max_string_length_error):
        print(line)


def diagnostic_ball_candidate_keys(client, args: argparse.Namespace) -> str:
    discovered_keys = []
    if args.scan_ball_candidates:
        discovered_keys = discover_pose_keys(
            client,
            args.ball_scan_pattern,
            args.max_ball_scan_keys,
        )
    return ball_candidate_keys_text(args.ball_pose_key, args.ball_candidate_key, discovered_keys)


def ball_velocity_from_state(
    state: PendulumState,
    params: SwingParams,
    swing_axis_world: np.ndarray,
) -> np.ndarray:
    x_axis, z_axis = pendulum_frame(swing_axis_world)
    _, tangent = pendulum_units(state.theta, x_axis, z_axis)
    return params.string_length * state.theta_dot * tangent


def predict_descent_time_to_height(
    ball_position_w: np.ndarray,
    ball_velocity_w: np.ndarray,
    target_height: float,
) -> float | None:
    # z(t) = z0 + vz*t - 0.5*g*t^2
    z0 = float(ball_position_w[2])
    vz = float(ball_velocity_w[2])
    a = 0.5 * GRAVITY_MAG
    b = -vz
    c = target_height - z0
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    descending = [root for root in roots if root > 0.0 and vz - GRAVITY_MAG * root < 0.0]
    if not descending:
        return None
    return min(descending)


def catch_cup_target(
    ball_position_w: np.ndarray,
    ball_velocity_w: np.ndarray,
    cup_height: float,
    geometry: KendamaGeometry,
    horizon: float,
) -> tuple[np.ndarray, float] | None:
    target_ball_height = cup_height + geometry.ball_center_above_cup
    crossing_time = predict_descent_time_to_height(ball_position_w, ball_velocity_w, target_ball_height)
    if crossing_time is None or crossing_time > horizon:
        return None
    target = np.array(
        [
            ball_position_w[0] + ball_velocity_w[0] * crossing_time,
            ball_position_w[1] + ball_velocity_w[1] * crossing_time,
            cup_height,
        ],
        dtype=float,
    )
    return target, crossing_time


def update_catch_latch(
    prediction: tuple[np.ndarray, float] | None,
    previous: CatchTarget | None,
    now: float,
    timeout: float,
) -> CatchTarget | None:
    if prediction is not None:
        cup_center, crossing_time = prediction
        return CatchTarget(cup_center.copy(), crossing_time, now)
    if previous is not None and now - previous.updated_at <= timeout:
        return previous
    return None


def bounded_flange_goal(
    current_goal: np.ndarray,
    desired_goal: np.ndarray,
    base_flange_position: np.ndarray,
    max_step: float,
    max_displacement: float,
) -> np.ndarray:
    bounded_desired = base_flange_position + clamp_norm(
        desired_goal - base_flange_position,
        max_displacement,
    )
    return current_goal + clamp_norm(bounded_desired - current_goal, max_step)


def set_active_controller(client: redis.Redis, key: str, controller_name: str) -> None:
    while True:
        client.set(key, controller_name)
        current = client.get(key)
        if current is not None and current.decode("utf-8") == controller_name:
            return
        time.sleep(0.005)


def set_cartesian_goal(
    client: redis.Redis,
    keys: RedisKeys,
    position: np.ndarray,
    orientation: np.ndarray,
) -> None:
    client.set(keys.cartesian_goal_position, json.dumps(np.asarray(position).tolist()))
    client.set(keys.cartesian_goal_orientation, json.dumps(np.asarray(orientation).tolist()))


def publish_ball_in_cup_fixture(
    client,
    keys: RedisKeys,
    transform_wf: Transform,
    geometry: KendamaGeometry,
    ball_pose_key: str,
    cup_normal_w: np.ndarray = WORLD_UP,
) -> np.ndarray:
    cup_normal = normed(cup_normal_w, WORLD_UP)
    cup_w = cup_center(transform_wf, geometry)
    ball_w = cup_w + geometry.ball_center_above_cup * cup_normal
    client.set(keys.flange_transform, transform_payload(transform_wf))
    client.set(ball_pose_key, pose_payload(ball_w))
    return ball_w


def collect_cup_offset_samples(
    client,
    keys: RedisKeys,
    ball_pose_key: str,
    sample_count: int,
    sample_period: float,
    ball_center_above_cup: float,
    cup_normal_w: np.ndarray,
    sleep_fn=time.sleep,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = []

    for _ in range(sample_count):
        raw_flange = client.get(keys.flange_transform)
        raw_ball, _, ball_pose_candidates = get_first_available(client, ball_pose_key)
        if raw_flange is None:
            raise RuntimeError(f"Missing flange transform Redis key: {keys.flange_transform}")
        if raw_ball is None:
            raise RuntimeError(f"Missing ball pose Redis key candidates: {', '.join(ball_pose_candidates)}")

        flange = parse_transform_value(raw_flange)
        ball = parse_pose_position(raw_ball)
        offsets.append(
            estimate_cup_offset_from_ball(
                flange,
                ball,
                ball_center_above_cup,
                cup_normal_w,
            )
        )
        sleep_fn(sample_period)

    samples = np.vstack(offsets)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return samples, mean, std


def collect_anchor_offset_samples(
    client,
    keys: RedisKeys,
    ball_pose_key: str,
    sample_count: int,
    sample_period: float,
    string_length: float,
    anchor_to_ball_direction_w: np.ndarray,
    sleep_fn=time.sleep,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = []

    for _ in range(sample_count):
        raw_flange = client.get(keys.flange_transform)
        raw_ball, _, ball_pose_candidates = get_first_available(client, ball_pose_key)
        if raw_flange is None:
            raise RuntimeError(f"Missing flange transform Redis key: {keys.flange_transform}")
        if raw_ball is None:
            raise RuntimeError(f"Missing ball pose Redis key candidates: {', '.join(ball_pose_candidates)}")

        flange = parse_transform_value(raw_flange)
        ball = parse_pose_position(raw_ball)
        offsets.append(
            estimate_anchor_offset_from_hanging_ball(
                flange,
                ball,
                string_length,
                anchor_to_ball_direction_w,
            )
        )
        sleep_fn(sample_period)

    samples = np.vstack(offsets)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return samples, mean, std


def build_model_snapshot(
    client,
    keys: RedisKeys,
    geometry: KendamaGeometry,
    params: SwingParams,
    ball_pose_key: str,
    ball_velocity_key: str,
    swing_axis_world: np.ndarray,
    ball_velocity_override: np.ndarray | None = None,
) -> ModelSnapshot:
    missing_keys = []
    raw_flange = client.get(keys.flange_transform)
    raw_ball, _, ball_pose_candidates = get_first_available(client, ball_pose_key)
    raw_ball_velocity, _, ball_velocity_candidates = get_first_available(client, ball_velocity_key)

    if raw_flange is None:
        missing_keys.append(keys.flange_transform)
    if raw_ball is None:
        missing_keys.append(" | ".join(ball_pose_candidates))
    if raw_ball_velocity is None and ball_velocity_override is None:
        missing_keys.append(" | ".join(ball_velocity_candidates))

    flange = parse_transform_value(raw_flange) if raw_flange is not None else None
    cup_w = cup_center(flange, geometry) if flange is not None else None
    anchor_w = anchor_position(flange, geometry) if flange is not None else None
    ball_w = parse_pose_position(raw_ball) if raw_ball is not None else None
    ball_velocity_estimated = raw_ball_velocity is None and ball_velocity_override is not None
    ball_velocity_w = parse_velocity(raw_ball_velocity) if raw_ball_velocity is not None else ball_velocity_override
    observed_string_length = None
    observed_string_error = None
    validity = None
    if anchor_w is not None and ball_w is not None:
        observed_string_length = ball_anchor_distance(anchor_w, ball_w)
        observed_string_error = observed_string_length - params.string_length
        validity = evaluate_ball_state_validity(anchor_w, ball_w, ball_velocity_w, params)

    pendulum = None
    energy = None
    target_energy = None
    catch = None
    if (
        anchor_w is not None
        and ball_w is not None
        and ball_velocity_w is not None
        and validity is not None
        and validity.valid
    ):
        pendulum = estimate_pendulum_state(
            anchor_w,
            ball_w,
            ball_velocity_w,
            params.string_length,
            swing_axis_world,
        )
        energy = pendulum_energy(pendulum.theta, pendulum.theta_dot, params.string_length)
        target_energy = params.energy_fraction * 2.0 * GRAVITY_MAG * params.string_length
        if cup_w is not None:
            catch = catch_cup_target(
                ball_w,
                ball_velocity_w,
                cup_w[2],
                geometry,
                params.catch_time_horizon,
            )

    return ModelSnapshot(
        flange=flange,
        cup_w=cup_w,
        anchor_w=anchor_w,
        ball_w=ball_w,
        ball_velocity_w=ball_velocity_w,
        ball_velocity_estimated=ball_velocity_estimated,
        ball_anchor_distance=observed_string_length,
        string_length_error=observed_string_error,
        ball_state_validity=validity,
        pendulum=pendulum,
        energy=energy,
        target_energy=target_energy,
        catch=catch,
        missing_keys=missing_keys,
    )


def print_model_snapshot(snapshot: ModelSnapshot) -> None:
    print("kendama model preflight")
    if snapshot.missing_keys:
        print("missing_keys:")
        for key in snapshot.missing_keys:
            print(f"  {key}")
    else:
        print("missing_keys: none")

    if snapshot.flange is not None:
        print(f"flange_position={np.round(snapshot.flange.position, 6).tolist()}")
    if snapshot.cup_w is not None:
        print(f"cup_center={np.round(snapshot.cup_w, 6).tolist()}")
    if snapshot.anchor_w is not None:
        print(f"anchor={np.round(snapshot.anchor_w, 6).tolist()}")
    if snapshot.ball_w is not None:
        print(f"ball_position={np.round(snapshot.ball_w, 6).tolist()}")
    if snapshot.ball_anchor_distance is not None and snapshot.string_length_error is not None:
        print(f"ball_anchor_distance={snapshot.ball_anchor_distance:.6f}")
        print(f"string_length_error={snapshot.string_length_error:.6f}")
    if snapshot.ball_velocity_w is not None:
        print(f"ball_velocity={np.round(snapshot.ball_velocity_w, 6).tolist()}")
        print(f"ball_velocity_estimated={snapshot.ball_velocity_estimated}")
    if snapshot.ball_state_validity is not None:
        validity = snapshot.ball_state_validity
        fields = [f"valid={validity.valid}", f"reason={validity.reason}"]
        if validity.string_length_error is not None:
            fields.append(f"string_err={validity.string_length_error:.6f}")
        if validity.ball_speed is not None:
            fields.append(f"speed={validity.ball_speed:.6f}")
        print("ball_state=" + " ".join(fields))
    if snapshot.pendulum is not None:
        print(
            "pendulum="
            f"theta_deg={math.degrees(snapshot.pendulum.theta):.3f} "
            f"theta_dot_deg_s={math.degrees(snapshot.pendulum.theta_dot):.3f}"
        )
    if snapshot.energy is not None and snapshot.target_energy is not None:
        print(f"energy_ratio={snapshot.energy / max(snapshot.target_energy, 1e-9):.6f}")
    if snapshot.catch is not None:
        cup_target, crossing_time = snapshot.catch
        print(f"catch_time={crossing_time:.6f}")
        print(f"catch_cup_target={np.round(cup_target, 6).tolist()}")


def preflight(args: argparse.Namespace, geometry: KendamaGeometry, params: SwingParams) -> None:
    keys = RedisKeys(args.robot_name)
    client = redis.Redis(host=args.redis_host, port=args.redis_port)
    velocity_override = None
    if args.estimate_ball_velocity:
        velocity_override = estimate_velocity_from_pose_samples(
            client,
            args.ball_pose_key,
            args.velocity_estimate_samples,
            args.velocity_estimate_period,
        )
    snapshot = build_model_snapshot(
        client,
        keys,
        geometry,
        params,
        args.ball_pose_key,
        args.ball_velocity_key,
        parse_vec(args.swing_axis_world, 3),
        ball_velocity_override=velocity_override,
    )
    print_model_snapshot(snapshot)
    if args.diagnose_ball_candidates:
        diagnostics = diagnose_ball_pose_candidates(
            client,
            diagnostic_ball_candidate_keys(client, args),
            snapshot.cup_w,
            snapshot.anchor_w,
            params.string_length,
        )
        print_ball_candidate_diagnostics(diagnostics, args.max_string_length_error)


def print_cup_offset_estimate(samples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    print("estimated flange-frame cup offset from ball-in-cup samples")
    print(f"samples={len(samples)}")
    print(f"mean={np.round(mean, 6).tolist()}")
    print(f"std ={np.round(std, 6).tolist()}")
    print("--cup-offset-f " + ",".join(f"{value:.6f}" for value in mean))


def print_anchor_offset_estimate(samples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    print("estimated flange-frame anchor offset from hanging-ball samples")
    print(f"samples={len(samples)}")
    print(f"mean={np.round(mean, 6).tolist()}")
    print(f"std ={np.round(std, 6).tolist()}")
    print("--anchor-offset-f " + ",".join(f"{value:.6f}" for value in mean))


def calibrate_cup_offset_from_ball(args: argparse.Namespace) -> None:
    keys = RedisKeys(args.robot_name)
    client = redis.Redis(host=args.redis_host, port=args.redis_port)
    sample_count = max(1, args.calibration_samples)
    sample_period = 1.0 / args.calibration_rate
    cup_normal_w = parse_vec(args.calibration_cup_normal_world, 3)
    samples, mean, std = collect_cup_offset_samples(
        client,
        keys,
        args.ball_pose_key,
        sample_count,
        sample_period,
        args.ball_center_above_cup,
        cup_normal_w,
    )
    print_cup_offset_estimate(samples, mean, std)


def calibrate_anchor_offset_from_hanging_ball(args: argparse.Namespace) -> None:
    keys = RedisKeys(args.robot_name)
    client = redis.Redis(host=args.redis_host, port=args.redis_port)
    sample_count = max(1, args.calibration_samples)
    sample_period = 1.0 / args.calibration_rate
    anchor_to_ball_direction_w = parse_vec(args.hanging_anchor_to_ball_direction_world, 3)
    samples, mean, std = collect_anchor_offset_samples(
        client,
        keys,
        args.ball_pose_key,
        sample_count,
        sample_period,
        args.string_length,
        anchor_to_ball_direction_w,
    )
    print_anchor_offset_estimate(samples, mean, std)


def dry_run_calibration(args: argparse.Namespace, geometry: KendamaGeometry) -> None:
    keys = RedisKeys(args.robot_name)
    client = MemoryRedis()
    angle = math.radians(25.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = Transform(np.array([0.42, -0.10, 0.28]), rotation)
    cup_normal_w = parse_vec(args.calibration_cup_normal_world, 3)

    publish_ball_in_cup_fixture(
        client,
        keys,
        transform,
        geometry,
        args.ball_pose_key,
        cup_normal_w,
    )
    samples, mean, std = collect_cup_offset_samples(
        client,
        keys,
        args.ball_pose_key,
        max(1, args.calibration_samples),
        0.0,
        geometry.ball_center_above_cup,
        cup_normal_w,
        sleep_fn=lambda _: None,
    )

    print("dry-run calibration fixture")
    print(f"true_offset={np.round(geometry.cup_offset_f, 6).tolist()}")
    print_cup_offset_estimate(samples, mean, std)


def dry_run(args: argparse.Namespace, geometry: KendamaGeometry, params: SwingParams) -> None:
    dt = 1.0 / args.rate
    swing_axis = parse_vec(args.swing_axis_world, 3)
    base_anchor = np.zeros(3)
    anchor = base_anchor.copy()
    anchor_velocity = np.zeros(3)
    state = PendulumState(math.radians(args.initial_theta_deg), math.radians(args.initial_theta_dot_deg))

    print("dry-run swing-up model")
    print(f"string_length={params.string_length:.3f} m")
    print(f"target_energy={params.energy_fraction:.2f} * upright energy")
    print(f"cup_offset_f={np.round(geometry.cup_offset_f, 4).tolist()}")
    print(f"anchor_offset_f={np.round(geometry.anchor_offset_f, 4).tolist()}")

    print_period = max(1, int(args.rate * args.print_period))
    total_steps = int(args.duration * args.rate)
    max_abs_theta = abs(state.theta)

    for step in range(total_steps + 1):
        accel, energy, target_energy = swingup_anchor_accel(state, params, swing_axis)
        recenter = -params.recenter_kp * (anchor - base_anchor) - params.recenter_kd * anchor_velocity
        anchor_accel = clamp_norm(accel + recenter, params.max_anchor_accel)
        anchor_velocity = clamp_norm(anchor_velocity + anchor_accel * dt, params.max_anchor_speed)
        anchor = anchor + anchor_velocity * dt

        offset = anchor - base_anchor
        if np.linalg.norm(offset) > params.max_anchor_offset:
            offset = clamp_norm(offset, params.max_anchor_offset)
            anchor = base_anchor + offset
            anchor_velocity = anchor_velocity - offset * max(0.0, np.dot(anchor_velocity, offset)) / max(
                np.dot(offset, offset), 1e-9
            )

        state = step_taut_pendulum(state, anchor_accel, params, swing_axis, dt)
        max_abs_theta = max(max_abs_theta, abs(state.theta))

        if step % print_period == 0 or step == total_steps:
            ball = ball_position_from_state(anchor, state, params, swing_axis)
            ball_vel = ball_velocity_from_state(state, params, swing_axis)
            catch = catch_cup_target(ball, ball_vel, 0.0, geometry, params.catch_time_horizon)
            catch_text = ""
            if catch is not None:
                _, crossing_time = catch
                catch_text = f" catch_t={crossing_time:.3f}s"
            print(
                f"t={step * dt:5.2f}s "
                f"theta={math.degrees(state.theta):7.2f}deg "
                f"theta_dot={math.degrees(state.theta_dot):8.2f}deg/s "
                f"E/Eup={energy / (2.0 * GRAVITY_MAG * params.string_length):5.2f} "
                f"anchor={np.round(anchor, 3).tolist()}"
                f"{catch_text}"
            )

    print(f"max_abs_theta={math.degrees(max_abs_theta):.1f} deg")


def dry_run_catch(args: argparse.Namespace, geometry: KendamaGeometry, params: SwingParams) -> None:
    dt = 1.0 / args.rate
    base_cup = np.array([0.0, 0.0, 0.0])
    current_cup = base_cup.copy()
    ball = np.array([-0.18, 0.0, 0.70])
    velocity = np.array([0.42, 0.0, -0.20])
    phase = Phase.SWING_UP
    catch_latch = None
    dropped_prediction_steps = 0
    print("dry-run catch state machine")

    total_steps = int(args.duration * args.rate)
    print_period = max(1, int(args.rate * args.print_period))
    for step in range(total_steps + 1):
        now = step * dt
        ball = ball + velocity * dt
        velocity = velocity + GRAVITY * dt

        prediction = catch_cup_target(
            ball,
            velocity,
            base_cup[2],
            geometry,
            params.catch_time_horizon,
        )
        if catch_latch is not None and prediction is not None and dropped_prediction_steps < 3:
            # Exercise the latch with a short artificial prediction dropout.
            dropped_prediction_steps += 1
            prediction = None

        catch_latch = update_catch_latch(
            prediction,
            catch_latch,
            now,
            params.catch_target_timeout,
        )
        near_height = abs(ball[2] - (base_cup[2] + geometry.ball_center_above_cup)) < params.catch_height_window
        aligned = np.linalg.norm(ball[:2] - current_cup[:2]) < params.catch_xy_window

        if phase == Phase.SWING_UP and catch_latch is not None and velocity[2] < -0.05:
            phase = Phase.TRACK_CATCH
        elif phase == Phase.TRACK_CATCH and catch_latch is None:
            phase = Phase.SWING_UP
        elif phase == Phase.TRACK_CATCH and near_height and aligned and velocity[2] < -0.05:
            phase = Phase.DAMP

        if phase == Phase.TRACK_CATCH and catch_latch is not None:
            current_cup[:2] = catch_latch.cup_center_w[:2]

        if step % print_period == 0 or step == total_steps:
            latch_text = "none"
            if catch_latch is not None:
                latch_text = (
                    f"{np.round(catch_latch.cup_center_w, 3).tolist()} "
                    f"age={now - catch_latch.updated_at:.3f}s"
                )
            print(
                f"t={now:5.2f}s phase={phase.name:11s} "
                f"ball={np.round(ball, 3).tolist()} "
                f"vel={np.round(velocity, 3).tolist()} "
                f"cup={np.round(current_cup, 3).tolist()} "
                f"latch={latch_text}"
            )

    print(f"final_phase={phase.name}")


def redis_loop(args: argparse.Namespace, geometry: KendamaGeometry, params: SwingParams) -> None:
    keys = RedisKeys(args.robot_name)
    client = redis.Redis(host=args.redis_host, port=args.redis_port)
    dt = 1.0 / args.rate
    swing_axis = parse_vec(args.swing_axis_world, 3)
    ball_pose_key = args.ball_pose_key
    ball_velocity_key = args.ball_velocity_key

    raw_flange = client.get(keys.flange_transform)
    if raw_flange is None:
        raise RuntimeError(f"Missing flange transform Redis key: {keys.flange_transform}")

    flange = parse_transform_value(raw_flange)
    hold_rotation = flange.rotation.copy()
    base_anchor = anchor_position(flange, geometry)
    base_cup = cup_center(flange, geometry)
    anchor_goal = base_anchor.copy()
    anchor_velocity = np.zeros(3)
    flange_goal = flange.position.copy()
    base_flange_position = flange.position.copy()
    phase = Phase.SWING_UP
    damp_start = None
    damp_base_flange = None
    catch_latch = None
    previous_ball = None
    previous_ball_time = None

    if args.command:
        raw_ball_start, _, ball_pose_candidates = get_first_available(client, ball_pose_key)
        if raw_ball_start is None:
            raise RuntimeError(
                "Cannot arm command mode without an initial ball pose. "
                f"Checked candidates: {', '.join(ball_pose_candidates)}"
            )
        ball_start = parse_pose_position(raw_ball_start)
        initial_string_error = ball_anchor_string_error(base_anchor, ball_start, params.string_length)
        if abs(initial_string_error) > args.max_string_length_error:
            initial_distance = ball_anchor_distance(base_anchor, ball_start)
            diagnostics = diagnose_ball_pose_candidates(
                client,
                diagnostic_ball_candidate_keys(client, args),
                base_cup,
                base_anchor,
                params.string_length,
            )
            diagnostic_text = "\n".join(
                format_ball_candidate_diagnostics(
                    diagnostics,
                    args.max_string_length_error,
                )
            )
            raise RuntimeError(
                "Cannot arm command mode: initial ball-anchor distance is inconsistent "
                f"with taut-string model. distance={initial_distance:.4f} m, "
                f"string_length={params.string_length:.4f} m, "
                f"error={initial_string_error:.4f} m, "
                f"tolerance={args.max_string_length_error:.4f} m\n"
                f"{diagnostic_text}"
            )
        set_active_controller(client, keys.active_controller, "cartesian_controller")
        set_cartesian_goal(client, keys, flange_goal, hold_rotation)

    print("flange-space kendama swing-up model")
    print(f"commanding Redis: {args.command}")
    print(f"flange key: {keys.flange_transform}")
    print(f"base cup: {np.round(base_cup, 4).tolist()}")
    print(f"base anchor: {np.round(base_anchor, 4).tolist()}")
    print(f"cup offset F: {np.round(geometry.cup_offset_f, 4).tolist()}")
    print(f"anchor offset F: {np.round(geometry.anchor_offset_f, 4).tolist()}")

    last_print = time.perf_counter()
    next_tick = time.perf_counter()
    start_time = time.perf_counter()

    while True:
        now = time.perf_counter()
        if args.duration > 0.0 and now - start_time >= args.duration:
            print(f"Run duration reached ({args.duration:.3f}s).")
            break

        sleep_time = next_tick - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        next_tick += dt
        now = time.perf_counter()

        raw_flange = client.get(keys.flange_transform)
        if raw_flange is None:
            continue
        flange = parse_transform_value(raw_flange)
        current_anchor = anchor_position(flange, geometry)
        current_cup = cup_center(flange, geometry)

        raw_ball, _, _ = get_first_available(client, ball_pose_key)
        raw_ball_velocity, _, _ = get_first_available(client, ball_velocity_key)
        if raw_ball is None or (raw_ball_velocity is None and not args.estimate_ball_velocity):
            if now - last_print > args.print_period:
                print("waiting for ball pose and velocity")
                last_print = now
            continue

        ball = parse_pose_position(raw_ball)
        if raw_ball_velocity is not None:
            ball_velocity = parse_velocity(raw_ball_velocity)
        elif previous_ball is not None and previous_ball_time is not None and now > previous_ball_time:
            ball_velocity = (ball - previous_ball) / (now - previous_ball_time)
        else:
            previous_ball = ball.copy()
            previous_ball_time = now
            if now - last_print > args.print_period:
                print("waiting for a second ball pose sample to estimate velocity")
                last_print = now
            continue
        previous_ball = ball.copy()
        previous_ball_time = now
        current_string_distance = ball_anchor_distance(current_anchor, ball)
        current_string_error = current_string_distance - params.string_length
        validity = evaluate_ball_state_validity(current_anchor, ball, ball_velocity, params)
        if not validity.valid:
            phase = Phase.SWING_UP
            catch_latch = None
            damp_start = None
            damp_base_flange = None
            anchor_goal = base_anchor.copy()
            anchor_velocity[:] = 0.0
            desired_flange = invalid_ball_state_desired_flange(
                args.invalid_state_action,
                flange_goal,
                base_flange_position,
            )
            flange_goal = bounded_flange_goal(
                flange_goal,
                desired_flange,
                base_flange_position,
                args.max_step,
                args.max_flange_displacement,
            )
            if args.command:
                set_cartesian_goal(client, keys, flange_goal, hold_rotation)
            if now - last_print > args.print_period:
                speed_text = ""
                if validity.ball_speed is not None:
                    speed_text = f" speed={validity.ball_speed:.3f}m/s"
                print(
                    f"INVALID_BALL reason={validity.reason} "
                    f"string_err={current_string_error: .3f}m{speed_text} "
                    f"cup={np.round(current_cup, 3).tolist()} "
                    f"flange_goal={np.round(flange_goal, 3).tolist()} "
                    f"action={args.invalid_state_action}"
                )
                last_print = now
            continue

        state = estimate_pendulum_state(
            current_anchor,
            ball,
            ball_velocity,
            params.string_length,
            swing_axis,
        )

        catch = catch_cup_target(
            ball,
            ball_velocity,
            base_cup[2],
            geometry,
            params.catch_time_horizon,
        )
        catch_latch = update_catch_latch(
            catch,
            catch_latch,
            now,
            params.catch_target_timeout,
        )
        ball_near_catch_height = abs(ball[2] - (base_cup[2] + geometry.ball_center_above_cup)) < params.catch_height_window
        ball_aligned_with_cup = np.linalg.norm(ball[:2] - current_cup[:2]) < params.catch_xy_window
        if phase == Phase.SWING_UP and catch_latch is not None and ball_velocity[2] < -0.05:
            phase = Phase.TRACK_CATCH
        elif phase == Phase.TRACK_CATCH and catch_latch is None:
            phase = Phase.SWING_UP
        elif phase == Phase.TRACK_CATCH and ball_near_catch_height and ball_aligned_with_cup and ball_velocity[2] < -0.05:
            phase = Phase.DAMP
            damp_start = now
            damp_base_flange = flange_goal.copy()

        energy = pendulum_energy(state.theta, state.theta_dot, params.string_length)
        target_energy = params.energy_fraction * 2.0 * GRAVITY_MAG * params.string_length

        if phase == Phase.SWING_UP:
            accel, energy, target_energy = swingup_anchor_accel(state, params, swing_axis)
            recenter = -params.recenter_kp * (anchor_goal - base_anchor) - params.recenter_kd * anchor_velocity
            anchor_accel = clamp_norm(accel + recenter, params.max_anchor_accel)
            anchor_velocity = clamp_norm(anchor_velocity + anchor_accel * dt, params.max_anchor_speed)
            anchor_goal = base_anchor + clamp_norm(anchor_goal + anchor_velocity * dt - base_anchor, params.max_anchor_offset)
            desired_flange = flange_position_for_anchor(anchor_goal, hold_rotation, geometry)
        elif phase == Phase.TRACK_CATCH and catch_latch is not None:
            desired_flange = flange_position_for_cup(catch_latch.cup_center_w, hold_rotation, geometry)
        elif phase == Phase.DAMP:
            elapsed = now - float(damp_start)
            alpha = min(1.0, elapsed / max(params.damp_time, 1e-6))
            drop = params.damp_drop * math.sin(math.pi * alpha)
            desired_flange = np.asarray(damp_base_flange).copy()
            desired_flange[2] -= drop
            if alpha >= 1.0:
                phase = Phase.SWING_UP
                anchor_goal = base_anchor.copy()
                anchor_velocity[:] = 0.0
                catch_latch = None
        else:
            desired_flange = flange_goal.copy()

        flange_goal = bounded_flange_goal(
            flange_goal,
            desired_flange,
            base_flange_position,
            args.max_step,
            args.max_flange_displacement,
        )

        if args.command:
            set_cartesian_goal(client, keys, flange_goal, hold_rotation)

        if now - last_print > args.print_period:
            catch_text = ""
            if catch_latch is not None:
                age = now - catch_latch.updated_at
                catch_text = (
                    f" catch_t={catch_latch.crossing_time:.3f}s "
                    f"catch_age={age:.3f}s "
                    f"cup_target={np.round(catch_latch.cup_center_w, 3).tolist()}"
                )
            print(
                f"{phase.name:11s} "
                f"theta={math.degrees(state.theta):7.2f}deg "
                f"theta_dot={math.degrees(state.theta_dot):8.2f}deg/s "
                f"E/Edes={energy / max(target_energy, 1e-9):5.2f} "
                f"string_err={current_string_error: .3f}m "
                f"cup={np.round(current_cup, 3).tolist()} "
                f"flange_goal={np.round(flange_goal, 3).tolist()}"
                f"{catch_text}"
            )
            last_print = now


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit flange-space kendama swing-up/catch model."
    )
    parser.add_argument("--dry-run", action="store_true", help="Run offline pendulum simulation and do not use Redis.")
    parser.add_argument("--dry-run-catch", action="store_true", help="Run offline catch state-machine simulation.")
    parser.add_argument("--dry-run-calibration", action="store_true", help="Run offline Redis-payload calibration fixture.")
    parser.add_argument("--preflight", action="store_true", help="Read live Redis keys once and print parsed model quantities.")
    parser.add_argument("--redis", action="store_true", help="Read Redis and compute live flange/cup targets.")
    parser.add_argument("--command", action="store_true", help="Actually write cartesian flange goals to Redis.")
    parser.add_argument(
        "--calibrate-cup-from-ball",
        action="store_true",
        help="Estimate --cup-offset-f from flange_transform and a ball sitting in the cup.",
    )
    parser.add_argument(
        "--calibrate-anchor-from-hanging-ball",
        action="store_true",
        help="Estimate --anchor-offset-f from flange_transform and a still ball hanging on a taut string.",
    )
    parser.add_argument("--robot-name", default="Titania")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--rate", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=8.0, help="Run duration in seconds; use 0 for infinite Redis mode.")
    parser.add_argument("--print-period", type=float, default=0.25)
    parser.add_argument("--max-step", type=float, default=0.004, help="Max flange goal step per cycle in meters.")
    parser.add_argument(
        "--max-flange-displacement",
        type=float,
        default=0.18,
        help="Maximum flange goal distance from the starting flange position in meters.",
    )
    parser.add_argument("--ball-pose-key", default="opensai::sensors::KendamaBall::object_pose,KendamaBall::pos")
    parser.add_argument("--ball-velocity-key", default="opensai::sensors::KendamaBall::object_velocity")
    parser.add_argument(
        "--diagnose-ball-candidates",
        action="store_true",
        help="In preflight, report each configured ball pose candidate against cup/anchor geometry.",
    )
    parser.add_argument(
        "--ball-candidate-key",
        default=None,
        help=(
            "Comma-separated ball pose keys to score with --diagnose-ball-candidates. "
            "Defaults to --ball-pose-key plus KendamaBallBlue::pos."
        ),
    )
    parser.add_argument(
        "--scan-ball-candidates",
        action="store_true",
        help="Expand ball diagnostics by scanning Redis for parseable ball pose keys.",
    )
    parser.add_argument(
        "--ball-scan-pattern",
        default="*KendamaBall*pos*,*KendamaBall*pose*,*Ball*pos*,*Ball*pose*",
        help="Comma-separated Redis SCAN patterns used with --scan-ball-candidates.",
    )
    parser.add_argument(
        "--max-ball-scan-keys",
        type=int,
        default=100,
        help="Maximum Redis keys to inspect while scanning ball candidates.",
    )
    parser.add_argument(
        "--estimate-ball-velocity",
        action="store_true",
        help="Estimate ball velocity from consecutive pose samples when velocity key data is unavailable.",
    )
    parser.add_argument("--velocity-estimate-samples", type=int, default=5)
    parser.add_argument("--velocity-estimate-period", type=float, default=0.02)
    parser.add_argument("--swing-axis-world", default="1,0,0", help="Horizontal swing-plane axis in world frame.")
    parser.add_argument(
        "--cup-offset-f",
        default="0,0,0.1143",
        help="Flange-frame vector from flange to cup center. Default is 4.5 in along flange +Z.",
    )
    parser.add_argument(
        "--urdf-cup-offset",
        action="store_true",
        help="Use the simulated parallel-URDF cup offset [0, -0.04, 0.20] instead of the physical 4.5 in default.",
    )
    parser.add_argument(
        "--anchor-offset-f",
        default=None,
        help="Flange-frame vector from flange to string anchor. Defaults to --cup-offset-f.",
    )
    parser.add_argument("--ball-center-above-cup", type=float, default=0.039)
    parser.add_argument("--string-length", type=float, default=0.40)
    parser.add_argument(
        "--max-string-length-error",
        type=float,
        default=0.15,
        help="Maximum allowed initial |ball-anchor distance - string length| before command mode refuses to arm.",
    )
    parser.add_argument(
        "--max-ball-speed",
        type=float,
        default=8.0,
        help="Maximum plausible ball speed before live model/catch updates are gated off.",
    )
    parser.add_argument(
        "--invalid-state-action",
        choices=("recenter", "hold"),
        default="recenter",
        help="Flange behavior when live ball sensing violates model assumptions.",
    )
    parser.add_argument("--energy-fraction", type=float, default=0.92)
    parser.add_argument("--pump-gain", type=float, default=2.2)
    parser.add_argument("--max-anchor-accel", type=float, default=5.0)
    parser.add_argument("--max-anchor-speed", type=float, default=0.45)
    parser.add_argument("--max-anchor-offset", type=float, default=0.12)
    parser.add_argument("--catch-xy-window", type=float, default=0.045)
    parser.add_argument("--catch-height-window", type=float, default=0.08)
    parser.add_argument("--catch-time-horizon", type=float, default=0.90)
    parser.add_argument("--catch-target-timeout", type=float, default=0.20)
    parser.add_argument("--initial-theta-deg", type=float, default=20.0)
    parser.add_argument("--initial-theta-dot-deg", type=float, default=0.0)
    parser.add_argument("--calibration-samples", type=int, default=100)
    parser.add_argument("--calibration-rate", type=float, default=100.0)
    parser.add_argument(
        "--calibration-cup-normal-world",
        default="0,0,1",
        help="World-frame cup normal while ball is sitting in the cup during calibration.",
    )
    parser.add_argument(
        "--hanging-anchor-to-ball-direction-world",
        default="0,0,-1",
        help="World-frame direction from string anchor to ball while hanging during anchor calibration.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.rate <= 0.0:
        raise ValueError("--rate must be positive")
    if args.duration < 0.0:
        raise ValueError("--duration must be non-negative")
    if args.max_step <= 0.0:
        raise ValueError("--max-step must be positive")
    if args.max_flange_displacement <= 0.0:
        raise ValueError("--max-flange-displacement must be positive")
    if args.catch_xy_window < 0.0:
        raise ValueError("--catch-xy-window must be non-negative")
    if args.catch_height_window < 0.0:
        raise ValueError("--catch-height-window must be non-negative")
    if args.catch_time_horizon <= 0.0:
        raise ValueError("--catch-time-horizon must be positive")
    if args.catch_target_timeout < 0.0:
        raise ValueError("--catch-target-timeout must be non-negative")
    if args.velocity_estimate_samples < 2:
        raise ValueError("--velocity-estimate-samples must be at least 2")
    if args.velocity_estimate_period <= 0.0:
        raise ValueError("--velocity-estimate-period must be positive")
    if args.string_length <= 0.0:
        raise ValueError("--string-length must be positive")
    if args.max_string_length_error < 0.0:
        raise ValueError("--max-string-length-error must be non-negative")
    if args.max_ball_speed <= 0.0:
        raise ValueError("--max-ball-speed must be positive")
    if args.max_ball_scan_keys <= 0:
        raise ValueError("--max-ball-scan-keys must be positive")
    if args.calibration_rate <= 0.0:
        raise ValueError("--calibration-rate must be positive")
    if args.calibration_samples <= 0:
        raise ValueError("--calibration-samples must be positive")

    if args.calibrate_cup_from_ball:
        calibrate_cup_offset_from_ball(args)
        return
    if args.calibrate_anchor_from_hanging_ball:
        calibrate_anchor_offset_from_hanging_ball(args)
        return

    cup_offset = PARALLEL_URDF_CUP_OFFSET_F.copy() if args.urdf_cup_offset else parse_vec(args.cup_offset_f, 3)
    anchor_offset = parse_vec(args.anchor_offset_f, 3) if args.anchor_offset_f else cup_offset.copy()
    geometry = KendamaGeometry(
        cup_offset_f=cup_offset,
        anchor_offset_f=anchor_offset,
        ball_center_above_cup=args.ball_center_above_cup,
    )
    params = SwingParams(
        string_length=args.string_length,
        max_string_length_error=args.max_string_length_error,
        max_ball_speed=args.max_ball_speed,
        energy_fraction=args.energy_fraction,
        pump_gain=args.pump_gain,
        max_anchor_accel=args.max_anchor_accel,
        max_anchor_speed=args.max_anchor_speed,
        max_anchor_offset=args.max_anchor_offset,
        catch_xy_window=args.catch_xy_window,
        catch_height_window=args.catch_height_window,
        catch_time_horizon=args.catch_time_horizon,
        catch_target_timeout=args.catch_target_timeout,
    )

    if args.preflight:
        preflight(args, geometry, params)
    elif args.dry_run_calibration:
        dry_run_calibration(args, geometry)
    elif args.dry_run_catch:
        dry_run_catch(args, geometry, params)
    elif args.command or args.redis:
        redis_loop(args, geometry, params)
    else:
        dry_run(args, geometry, params)


if __name__ == "__main__":
    main()
