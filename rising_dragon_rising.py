"""Rising dragon: start low, then move the cup/head upward on a spiral path.

The old version directly oscillated joints, which produced a left-right swing
but could not guarantee a Cartesian spiral or a level final link. This version
uses the Cartesian controller after parking in a low pose: the end-effector
position traces a helix while the captured low-pose orientation is held fixed.
"""

# Useful tunables:

# python3 rising_dragon_oscillate.py --radius 0.08 --rise-height 0.22 --turns 1.5 --duration 7.0
# For real robot:

# python3 rising_dragon_oscillate.py --real
# To return automatically after holding the top for 2 seconds:

# python3 rising_dragon_oscillate.py --hold-top 2

import argparse
import json
import math
import time
from enum import Enum, auto

import numpy as np
import redis


LOWERED_START_JOINTS = np.array([
    0.0,
    -0.6981317007977318,
    0.0,
    1.57079632679,
    0.0,
    2.2689280275926285,
    0.0,
])

JOINT_ARRIVAL_TOL = 3e-2
DEFAULT_RATE = 200.0


class State(Enum):
    RESETTING_JOINTS = auto()
    SWITCHING_TO_CARTESIAN = auto()
    SPIRALING_UP = auto()
    HOLDING_TOP = auto()


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


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
        description="Move the robot from a low pose through a rising Cartesian spiral."
    )
    parser.add_argument("--real", action="store_true", help="Use the real robot (Titania) instead of the simulator.")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Command rate in Hz.")
    parser.add_argument("--duration", type=float, default=7.0, help="Seconds spent spiraling upward.")
    parser.add_argument("--radius", type=float, default=0.08, help="Spiral radius in meters.")
    parser.add_argument("--rise-height", type=float, default=0.22, help="Total upward travel in meters.")
    parser.add_argument("--turns", type=float, default=1.5, help="Number of spiral revolutions during the rise.")
    parser.add_argument("--start-angle-deg", type=float, default=0.0, help="Initial phase of the spiral.")
    parser.add_argument("--hold-top", type=float, default=-1.0, help="Seconds to hold the final top pose; negative holds forever.")
    parser.add_argument(
        "--no-return",
        action="store_true",
        help="Do not park back at the lowered joint pose after the top hold.",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")

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
    spiral_start_time = None
    hold_start_time = None
    center_pos = None
    level_ori = None
    last_goal_pos = None

    print("=" * 60)
    print(f"RISING DRAGON SPIRAL - {robot_name} ({'real' if args.real else 'sim'})")
    print(
        f"radius={args.radius:.3f} m, rise={args.rise_height:.3f} m, "
        f"turns={args.turns:.2f}, duration={args.duration:.2f} s"
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
                last_goal_pos = center_pos.copy()

                set_vec(client, key_goal_pos, center_pos)
                set_vec(client, key_goal_ori, level_ori)

                print(f"Spiral base center: {np.round(center_pos, 4).tolist()}")
                print("Starting upward spiral. Ctrl+C to stop.")
                spiral_start_time = time.perf_counter()
                state = State.SPIRALING_UP

            elif state == State.SPIRALING_UP:
                elapsed = time.perf_counter() - spiral_start_time
                alpha = smoothstep(elapsed / args.duration)
                theta = math.radians(args.start_angle_deg) + 2.0 * math.pi * args.turns * alpha

                radial_ramp = smoothstep(min(alpha * 3.0, 1.0))
                radius = args.radius * radial_ramp
                xy_offset = np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0])
                z_offset = np.array([0.0, 0.0, args.rise_height * alpha])
                last_goal_pos = center_pos + xy_offset + z_offset

                set_vec(client, key_goal_pos, last_goal_pos)
                set_vec(client, key_goal_ori, level_ori)

                if alpha >= 1.0:
                    hold_start_time = time.perf_counter()
                    state = State.HOLDING_TOP
                    print(f"Reached spiral top: {np.round(last_goal_pos, 4).tolist()}")

            elif state == State.HOLDING_TOP:
                set_vec(client, key_goal_pos, last_goal_pos)
                set_vec(client, key_goal_ori, level_ori)
                if args.hold_top >= 0.0 and time.perf_counter() - hold_start_time >= args.hold_top:
                    print("Top hold complete.")
                    if args.no_return:
                        break
                    print("Returning to lowered joint pose.")
                    set_active_controller(client, key_active, "joint_controller")
                    set_vec(client, key_joint_goal, LOWERED_START_JOINTS)
                    break

    except KeyboardInterrupt:
        print("\nStopping by request.")
        if not args.no_return:
            print("Returning to lowered joint pose.")
            set_active_controller(client, key_active, "joint_controller")
            set_vec(client, key_joint_goal, LOWERED_START_JOINTS)


if __name__ == "__main__":
    main()
