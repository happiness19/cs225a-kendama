"""Rising dragon loop: spiral up, hold at the top, spiral down, repeat.

The robot parks in a low pose, switches to Cartesian control, captures the
low-pose orientation, then loops forever until Ctrl+C:

    spiral up -> hold top -> spiral down -> spiral up -> hold top -> ...

The captured orientation is held throughout the loop so the last link remains
parallel to the ground while carrying the ball.
"""

# Useful tunables:

# python3 rising_dragon_rising.py --radius 0.08 --rise-height 0.22 --turns 1.5 --duration 3.0
# For real robot:

# python3 rising_dragon_rising.py --real

# Hold at the top for 2 seconds:
# python3 rising_dragon_rising.py --hold-top 2

import argparse
import json
import math
import time
from enum import Enum, auto

import numpy as np
import redis


LOWERED_START_JOINTS = np.array([
    0.0,
    -1.1,
    0.0,
    1.57079632679,
    0.0,
    2.8,
    0.0,
])

JOINT_ARRIVAL_TOL = 0.2
DEFAULT_RATE = 200.0


class State(Enum):
    RESETTING_JOINTS = auto()
    SWITCHING_TO_CARTESIAN = auto()
    RISING_UP = auto()
    HOLDING_TOP = auto()
    SPIRALING_DOWN = auto()


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def spiral_position(
    base_pos: np.ndarray,
    radius: float,
    rise_height: float,
    turns: float,
    start_angle: float,
    alpha: float,
    upward: bool,
) -> np.ndarray:
    phase = smoothstep(alpha)
    theta = start_angle + 2.0 * math.pi * turns * phase
    radius_envelope = radius * math.sin(math.pi * phase) ** 2
    xy_offset = np.array([
        radius_envelope * math.cos(theta),
        radius_envelope * math.sin(theta),
        0.0,
    ])
    z_height = rise_height * phase if upward else rise_height * (1.0 - phase)
    z_offset = np.array([0.0, 0.0, z_height])
    return base_pos + xy_offset + z_offset


def limited_step(current: np.ndarray, desired: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0.0:
        return desired
    delta = desired - current
    dist = np.linalg.norm(delta)
    if dist <= max_step:
        return desired
    return current + delta * (max_step / dist)


def load_json_vec(client: redis.Redis, key: str, label: str) -> np.ndarray:
    raw = client.get(key)
    if raw is None:
        raise RuntimeError(f"Missing Redis key for {label}: {key}")
    return np.array(json.loads(raw.decode("utf-8")))


def set_vec(client: redis.Redis, key: str, value: np.ndarray) -> None:
    client.set(key, json.dumps(np.asarray(value).tolist()))


def set_active_controller(client: redis.Redis, key: str, name: str) -> None:
    while True:
        cur = client.get(key)
        if cur is not None and cur.decode("utf-8") == name:
            return
        client.set(key, name)
        time.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loop a spiral rise followed by a top hold and spiral descent."
    )
    parser.add_argument("--real", action="store_true", help="Use the real robot (Titania) instead of the simulator.")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Command rate in Hz.")
    parser.add_argument("--duration", type=float, default=9.0, help="Seconds spent on each spiral up/down phase.")
    parser.add_argument("--radius", type=float, default=1.5, help="Maximum spiral radius in meters.")
    parser.add_argument("--rise-height", type=float, default=0.2, help="Total upward travel in meters.")
    parser.add_argument("--turns", type=float, default=3.5, help="Number of spiral revolutions during each up/down phase.")
    parser.add_argument("--start-angle-deg", type=float, default=0.0, help="Initial phase of the spiral.")
    parser.add_argument("--hold-top", type=float, default=1.0, help="Seconds to hold the top pose before spiraling down.")
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.0005,
        help="Maximum Cartesian goal movement per command cycle in meters; set <=0 to disable.",
    )
    parser.add_argument(
        "--arrival-tol",
        type=float,
        default=0.003,
        help="Cartesian goal arrival tolerance in meters for top/bottom phase changes.",
    )
    parser.add_argument(
        "--no-return",
        action="store_true",
        help="Do not park back at the lowered joint pose after Ctrl+C.",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.hold_top < 0.0:
        raise ValueError("--hold-top must be non-negative for the repeating loop")
    if args.arrival_tol < 0.0:
        raise ValueError("--arrival-tol must be non-negative")

    robot_name = "Titania" if args.real else "Rizon4r"
    prefix = f"opensai::controllers::{robot_name}"
    key_joint_goal = f"{prefix}::joint_controller::joint_task::goal_position"
    key_joint_current = f"{prefix}::joint_controller::joint_task::current_position"
    key_goal_pos = f"{prefix}::cartesian_controller::cartesian_task::goal_position"
    key_goal_ori = f"{prefix}::cartesian_controller::cartesian_task::goal_orientation"
    key_current_pos = f"{prefix}::cartesian_controller::cartesian_task::current_position"
    key_current_ori = f"{prefix}::cartesian_controller::cartesian_task::current_orientation"
    key_active = f"{prefix}::active_controller_name"
    key_config = "::sai-interfaces-webui::config_file_name"

    client = redis.Redis()
    cfg = client.get(key_config)
    if cfg:
        print(f"Connected to config {cfg.decode('utf-8')}")
    else:
        print("Warning: config key missing; make sure the simulator or robot is launched.")

    dt = 1.0 / args.rate if args.rate > 0 else 0.005
    state = State.RESETTING_JOINTS
    phase_start_time = None
    hold_start_time = None
    center_pos = None
    level_ori = None
    goal_pos = None
    command_pos = None
    cycle_count = 0
    start_angle = math.radians(args.start_angle_deg)

    print("=" * 60)
    print(f"RISING DRAGON LOOP - {robot_name} ({'real' if args.real else 'sim'})")
    print(
        f"radius={args.radius:.3f} m, rise={args.rise_height:.3f} m, "
        f"turns={args.turns:.2f}, duration={args.duration:.2f} s, "
        f"hold_top={args.hold_top:.2f} s, "
        f"max_step={args.max_step:.4f} m, arrival_tol={args.arrival_tol:.4f} m"
    )
    print("The low-pose orientation is held fixed so the last link stays level.")
    print("=" * 60)

    set_active_controller(client, key_active, "joint_controller")
    set_vec(client, key_joint_goal, LOWERED_START_JOINTS)
    print("Parking in the lowered start pose...")

    try:
        while True:
            time.sleep(dt)

            if state == State.RESETTING_JOINTS:
                print("Waiting for the robot to reach the lowered start pose...")
                q = load_json_vec(client, key_joint_current, "current joint position")
                joint_error = np.linalg.norm(LOWERED_START_JOINTS - q)
                if joint_error < JOINT_ARRIVAL_TOL:
                    print("Lowered pose reached. Switching to Cartesian controller.")
                    state = State.SWITCHING_TO_CARTESIAN

            elif state == State.SWITCHING_TO_CARTESIAN:
                set_active_controller(client, key_active, "cartesian_controller")
                time.sleep(0.2)

                center_pos = load_json_vec(client, key_current_pos, "current Cartesian position")
                level_ori = load_json_vec(client, key_current_ori, "current Cartesian orientation")
                goal_pos = center_pos.copy()
                command_pos = center_pos.copy()

                set_vec(client, key_goal_pos, center_pos)
                set_vec(client, key_goal_ori, level_ori)

                print(f"Loop base center: {np.round(center_pos, 4).tolist()}")
                print("Starting spiral-up -> hold-top -> spiral-down loop. Ctrl+C to stop.")
                phase_start_time = time.perf_counter()
                state = State.RISING_UP

            elif state == State.RISING_UP:
                elapsed = time.perf_counter() - phase_start_time
                alpha = elapsed / args.duration
                desired_pos = spiral_position(
                    center_pos,
                    args.radius,
                    args.rise_height,
                    args.turns,
                    start_angle + cycle_count * 4.0 * math.pi * args.turns,
                    alpha,
                    upward=True,
                )
                command_pos = limited_step(command_pos, desired_pos, args.max_step)
                goal_pos = command_pos

                set_vec(client, key_goal_pos, goal_pos)
                set_vec(client, key_goal_ori, level_ori)

                top_pos = center_pos + np.array([0.0, 0.0, args.rise_height])
                if alpha >= 1.0 and np.linalg.norm(command_pos - top_pos) <= args.arrival_tol:
                    hold_start_time = time.perf_counter()
                    state = State.HOLDING_TOP
                    print(f"Cycle {cycle_count + 1}: reached top {np.round(goal_pos, 4).tolist()}")

            elif state == State.HOLDING_TOP:
                top_pos = center_pos + np.array([0.0, 0.0, args.rise_height])
                command_pos = limited_step(command_pos, top_pos, args.max_step)
                goal_pos = command_pos
                set_vec(client, key_goal_pos, goal_pos)
                set_vec(client, key_goal_ori, level_ori)
                if time.perf_counter() - hold_start_time >= args.hold_top:
                    print(f"Cycle {cycle_count + 1}: top hold complete, spiraling down.")
                    phase_start_time = time.perf_counter()
                    state = State.SPIRALING_DOWN

            elif state == State.SPIRALING_DOWN:
                elapsed = time.perf_counter() - phase_start_time
                alpha = elapsed / args.duration
                desired_pos = spiral_position(
                    center_pos,
                    args.radius,
                    args.rise_height,
                    args.turns,
                    start_angle + (cycle_count * 4.0 + 2.0) * math.pi * args.turns,
                    alpha,
                    upward=False,
                )
                command_pos = limited_step(command_pos, desired_pos, args.max_step)
                goal_pos = command_pos

                set_vec(client, key_goal_pos, goal_pos)
                set_vec(client, key_goal_ori, level_ori)

                if alpha >= 1.0 and np.linalg.norm(command_pos - center_pos) <= args.arrival_tol:
                    cycle_count += 1
                    print(f"Cycle {cycle_count}: spiral down complete, rising again.")
                    phase_start_time = time.perf_counter()
                    state = State.RISING_UP

    except KeyboardInterrupt:
        print("\nStopping by request.")
        if not args.no_return:
            print("Returning to lowered joint pose.")
            set_active_controller(client, key_active, "joint_controller")
            set_vec(client, key_joint_goal, LOWERED_START_JOINTS)


if __name__ == "__main__":
    main()
