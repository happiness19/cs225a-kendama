"""Move joint 1 toward KendamaBall XY while staying inside the limit."""

import argparse
import json
import math
import time

import redis


DEFAULT_HOME_JOINTS = [
    math.radians(50.23),
    math.radians(-50.02),
    math.radians(-29.81),
    math.radians(75.98),
    math.radians(-46.65),
    math.radians(-13.02),
    math.radians(-45.80),
]


def set_joint_goal(client, key, vec):
    client.set(key, json.dumps(vec))


def set_active_controller(client, key, name):
    while True:
        cur = client.get(key)
        if cur is not None and cur.decode("utf-8") == name:
            return
        client.set(key, name)
        time.sleep(0.005)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate joint 1 to track KendamaBall XY within a safety window.")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--ball-pos", default="KendamaBall::pos", help="Redis key for ball XY")
    parser.add_argument("--joint-index", type=int, default=1, help="Joint index to move")
    parser.add_argument("--base-x", type=float, default=0.0, help="World X that maps to zero joint angle")
    parser.add_argument("--base-y", type=float, default=0.0, help="World Y that maps to zero joint angle")
    parser.add_argument("--limit", type=float, default=60.0, help="Max angle (degrees) from zero, full stop outside this window")
    parser.add_argument("--rate", type=float, default=25.0, help="Control loop rate in Hz")
    args = parser.parse_args()

    joint_controller_prefix = f"opensai::controllers::Rizon4r::joint_controller::joint_task"
    KEY_JOINT_GOAL = f"{joint_controller_prefix}::goal_position"
    KEY_ACTIVE = "opensai::controllers::Rizon4r::active_controller_name"

    client = redis.Redis(host=args.host, port=args.port)
    dt = 1.0 / args.rate if args.rate > 0 else 0.05
    limit_rad = math.radians(abs(args.limit))

    print("Switching to the joint controller and parking the joints...")
    set_active_controller(client, KEY_ACTIVE, "joint_controller")
    pose = DEFAULT_HOME_JOINTS.copy()
    set_joint_goal(client, KEY_JOINT_GOAL, pose)

    print("Starting joint follower loop")
    waiting = False
    last_angle = 0.0
    try:
        while True:
            raw = client.get(args.ball_pos)
            if raw is None:
                if not waiting:
                    print("Waiting for ball data...")
                    waiting = True
                time.sleep(dt)
                continue

            waiting = False
            try:
                ball = json.loads(raw.decode("utf-8"))
            except Exception:
                time.sleep(dt)
                continue

            dx = ball[0] - args.base_x
            dy = ball[1] - args.base_y
            target_angle = math.atan2(dy, dx)
            if abs(target_angle) > limit_rad:
                print(f"Ball out of range (angle {math.degrees(target_angle):.1f}°). Holding until it returns.")
                time.sleep(dt)
                continue

            last_angle = target_angle
            pose = DEFAULT_HOME_JOINTS.copy()
            pose[args.joint_index] = target_angle
            set_joint_goal(client, KEY_JOINT_GOAL, pose)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("Stopping and returning to home pose.")
        set_joint_goal(client, KEY_JOINT_GOAL, DEFAULT_HOME_JOINTS)


if __name__ == "__main__":
    main()
