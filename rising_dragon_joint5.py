"""Joint-only rising dragon: sweep a single joint back and forth."""

import argparse
import json
import math
import time

import numpy as np
import redis


parser = argparse.ArgumentParser(description="Oscillate one joint while holding the rest fixed.")
parser.add_argument("--real", action="store_true", help="Use the real robot (Titania) instead of the simulator.")
parser.add_argument("--joint-index", type=int, default=5, help="Joint index to sweep (0-based).")
parser.add_argument("--amplitude", type=float, default=90.0, help="Max swing in degrees (± around starting position).")
parser.add_argument("--period", type=float, default=10.0, help="Seconds for a full back-and-forth cycle.")
parser.add_argument("--rate", type=float, default=200.0, help="Command rate in Hz.")
args = parser.parse_args()

robot_name = "Titania" if args.real else "Rizon4r"
prefix = f"opensai::controllers::{robot_name}"
KEY_JOINT_GOAL = f"{prefix}::joint_controller::joint_task::goal_position"
KEY_JOINT_CURRENT = f"{prefix}::joint_controller::joint_task::current_position"
KEY_ACTIVE = f"{prefix}::active_controller_name"
KEY_CONFIG = "::sai-interfaces-webui::config_file_name"

DEFAULT_JOINTS = np.array([
    0.0,
    -0.6981317007977318,
    0.0,
    1.57079632679,
    0.0,
    2.2689280275926285,
    0.0,
])

redis_client = redis.Redis()


def set_joint_goal(position: np.ndarray) -> None:
    redis_client.set(KEY_JOINT_GOAL, json.dumps(position.tolist()))


def set_active_controller(name: str) -> None:
    while True:
        cur = redis_client.get(KEY_ACTIVE)
        if cur is not None and cur.decode("utf-8") == name:
            return
        redis_client.set(KEY_ACTIVE, name)
        time.sleep(0.001)


def main() -> None:
    print("Running rising_dragon_joint5: joint-only motion")
    cfg = redis_client.get(KEY_CONFIG)
    if cfg:
        print(f"Connected to config {cfg.decode('utf-8')}")
    else:
        print("Warning: config key missing, make sure the simulator or robot is launched.")

    set_active_controller("joint_controller")
    print("Switching to joint controller and parking in the lowered pose...")
    set_joint_goal(DEFAULT_JOINTS)
    time.sleep(0.2)

    print(
        f"Sweeping joint {args.joint_index} ±{args.amplitude}° with {args.period}s period at {args.rate}Hz."
    )

    zero_pose = DEFAULT_JOINTS.copy()
    sweep_rad = math.radians(args.amplitude)
    omega = 2.0 * math.pi / args.period
    dt = 1.0 / args.rate if args.rate > 0 else 0.01
    start_time = time.perf_counter()

    try:
        while True:
            t = time.perf_counter() - start_time
            angle = sweep_rad * math.sin(omega * t)
            zero_pose[args.joint_index] = angle
            set_joint_goal(zero_pose)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("Stopping; parking the robot back to the default joint pose.")
        set_joint_goal(DEFAULT_JOINTS)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
