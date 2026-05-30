"""Use the robot end-effector pose to help joint 1 follow the ball without leaving safety bounds."""

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
    parser = argparse.ArgumentParser(
        description="Move joint 1 toward KendamaBall XY while respecting limits and using the robot pose as reference."
    )
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--ball-pos", default="KendamaBall::pos", help="Redis key for the ball position")
    parser.add_argument("--bot-pos", default="KendamaBot::pos", help="Redis key for the robot end-effector position")
    parser.add_argument("--limit", type=float, default=60.0, help="Maximum allowed deviation for joint 1 (degrees)")
    parser.add_argument("--rate", type=float, default=25.0, help="Control loop rate in Hz")
    args = parser.parse_args()

    state_prefix = "opensai::controllers::Rizon4r"
    KEY_JOINT_GOAL = f"{state_prefix}::joint_controller::joint_task::goal_position"
    KEY_ACTIVE = f"{state_prefix}::active_controller_name"

    client = redis.Redis(host=args.host, port=args.port)
    dt = 1.0 / args.rate if args.rate > 0 else 0.05
    limit_rad = math.radians(abs(args.limit))

    print("Switching to joint controller and parking joints")
    set_active_controller(client, KEY_ACTIVE, "joint_controller")
    set_joint_goal(client, KEY_JOINT_GOAL, DEFAULT_HOME_JOINTS)

    waiting = False
    try:
        while True:
            raw_ball = client.get(args.ball_pos)
            raw_bot = client.get(args.bot_pos)
            if raw_ball is None or raw_bot is None:
                if not waiting:
                    print("Waiting for KendamaBall and KendamaBot data...")
                    waiting = True
                time.sleep(dt)
                continue

            waiting = False
            try:
                ball = json.loads(raw_ball.decode("utf-8"))
                bot = json.loads(raw_bot.decode("utf-8"))
            except Exception:
                time.sleep(dt)
                continue

            dx = ball[0] - bot[0]
            dy = ball[1] - bot[1]
            target_angle = math.atan2(dy, dx)
            if abs(target_angle) > limit_rad:
                print(f"Ball outside ±{args.limit}° (angle {math.degrees(target_angle):.1f}). Holding until it returns.")
                time.sleep(dt)
                continue

            pose = DEFAULT_HOME_JOINTS.copy()
            pose[1] = target_angle
            set_joint_goal(client, KEY_JOINT_GOAL, pose)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("Stopping; returning to rest pose.")
        set_joint_goal(client, KEY_JOINT_GOAL, DEFAULT_HOME_JOINTS)


if __name__ == "__main__":
    main()
