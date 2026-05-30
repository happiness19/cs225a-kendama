import argparse
import json
import math
import signal
import time

import numpy as np
import redis

DEFAULT_HOME_JOINTS = np.array([
    math.radians(0.0),
    math.radians(-40.0),
    math.radians(0.0),
    math.radians(90.0),
    math.radians(0.0),
    math.radians(40.0),
    math.radians(0.0)
])

ROBOT_NAME = "Titania"

MAX_Z = 0.319234  # Z ceiling — robot will not move above this height\
MIN_Y = -0.35
MAX_Y = 0.35
MIN_X = 0.4
MAX_X = 0.8



def make_keys(robot_name):
    base = f"opensai::controllers::{robot_name}"
    sensors = f"opensai::sensors::{robot_name}"
    return {
        "active":       f"{base}::active_controller_name",
        "joint_goal":   f"{base}::joint_controller::joint_task::goal_position",
        "joint_cur":    f"{sensors}::joint_positions",
        "pos_goal":     f"{base}::cartesian_controller::cartesian_task::goal_position",
        "ori_goal":     f"{base}::cartesian_controller::cartesian_task::goal_orientation",
        "pos_cur":      f"{base}::cartesian_controller::cartesian_task::current_position",
        "ori_cur":      f"{base}::cartesian_controller::cartesian_task::current_orientation",
    }


def get_vec(client, key):
    raw = client.get(key)
    if raw is None:
        return None
    return np.array(json.loads(raw.decode("utf-8")))


def set_vec(client, key, vec):
    client.set(key, json.dumps(np.asarray(vec).tolist()))


def set_active_controller(client, key, name):
    """Write controller name and confirm it was accepted."""
    while True:
        client.set(key, name)
        cur = client.get(key)
        print(f"Current controller: {cur.decode('utf-8') if cur is not None else 'None'}")
        if cur is not None and cur.decode("utf-8") == name:
            return
        time.sleep(0.01)


def go_home(client, keys, home_joints, max_step_deg=0.5, threshold=0.2, settle_count=20):
    """Drive robot slowly to home joints by interpolating in small steps."""
    print("Setting joint controller active...")
    set_active_controller(client, keys["active"], "joint_controller")

    q = None
    while q is None:
        q = get_vec(client, keys["joint_cur"])
        time.sleep(0.01)

    target = q.copy()
    max_step = math.radians(max_step_deg)

    print("Moving slowly to home from current position...")

    stable = 0
    while True:
        delta = home_joints - target
        dist = np.linalg.norm(delta)
        if dist > max_step:
            delta *= max_step / dist
        target = target + delta

        set_vec(client, keys["joint_goal"], target)

        q = get_vec(client, keys["joint_cur"])
        if q is not None:
            err = np.linalg.norm(q - home_joints)
            print(f"Robot position: {np.degrees(q).round(2)}, err={err:.4f} rad",
                  f"Target position: {np.degrees(target).round(2)}")
            if err < threshold:
                stable += 1
                if stable >= settle_count:
                    print(f"Home reached (err={err:.4f} rad).")
                    return
            else:
                stable = 0

        time.sleep(0.01)


def track_ball(client, keys, ball_pos_key, max_step, dt):
    """Switch to Cartesian control and track the ball indefinitely."""
    print("Switching to Cartesian controller...")
    set_active_controller(client, keys["active"], "cartesian_controller")
    time.sleep(0.2)

    ee_pos = get_vec(client, keys["pos_cur"])
    ee_ori = get_vec(client, keys["ori_cur"])
    ball_pos = get_vec(client, ball_pos_key)

    if ee_pos is None or ee_ori is None or ball_pos is None:
        raise RuntimeError("Could not read EE pose or ball position from Redis.")

    offset = ee_pos - ball_pos
    fixed_ori = ee_ori.copy()
    target = ee_pos.copy()

    print(f"EE position  : {ee_pos}")
    print(f"Ball position: {ball_pos}")
    print(f"Offset       : {offset}")
    print(f"Z ceiling    : {MAX_Z}")
    print("Tracking ball — press Ctrl+C to stop.")

    while True:
        ball_pos = get_vec(client, ball_pos_key)
        if ball_pos is not None:
            desired = ball_pos + offset
            desired[0] = min(desired[0], MAX_X)  # Clamp X to workspace
            desired[0] = max(desired[0], MIN_X)  # Clamp X to workspace

            desired[1] = min(desired[1], MAX_Y)  # Clamp Y to workspace
            desired[1] = max(desired[1], MIN_Y)  # Clamp Y to workspace

            desired[2] = min(desired[2], MAX_Z)  # Clamp Z to ceiling
            desired[2] = max(desired[2], ee_pos[2])  # Clamp Z to ee_pos to avoid going below current height

            delta = desired - target
            dist = np.linalg.norm(delta)
            if dist > max_step:
                delta *= max_step / dist

            target = target + delta

            set_vec(client, keys["pos_goal"], target)
            set_vec(client, keys["ori_goal"], fixed_ori)

        print(f"Waiting for {dt}s, Robot Position: {ee_pos}, Target position: {target}, Ball position: {ball_pos}")
        time.sleep(dt)


def main():
    parser = argparse.ArgumentParser(description="Track KendamaBall using Cartesian control.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--ball-pos", default="KendamaBall::pos")
    parser.add_argument("--max-step", type=float, default=0.003,
                        help="Maximum Cartesian motion per cycle (meters).")
    args = parser.parse_args()

    client = redis.Redis(host=args.host, port=args.port)
    keys = make_keys(ROBOT_NAME)

    def _shutdown(sig, frame):
        print("\nShutting down.")
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _shutdown)

    go_home(client, keys, DEFAULT_HOME_JOINTS)
    time.sleep(1.0)
    track_ball(client, keys, args.ball_pos, args.max_step, 1.0 / args.rate)


if __name__ == "__main__":
    main()