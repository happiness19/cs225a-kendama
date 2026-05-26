"""
Move the robot to follow the Kendama ball's X and Y while keeping the robot's Z and orientation unchanged.

Usage:
  python3 move_with_ball.py --ball-key opensai::sensors::KendamaBall::object_pose --rate 100
  python3 move_with_ball.py --pos-key tidybot01::pos --ori-key tidybot01::ori --rate 50

The script uses Redis keys similar to `kendama_throw_and_catch.py` for cartesian control.
"""
import argparse
import json
import math
import time
import redis
import numpy as np
from dataclasses import dataclass


def parse_pose_raw(raw_bytes):
    if raw_bytes is None:
        return None, None
    try:
        s = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
        obj = json.loads(s)
    except Exception:
        return None, None

    if isinstance(obj, dict):
        if "position" in obj and "orientation" in obj:
            return obj["position"], obj["orientation"]
        if "pose" in obj and isinstance(obj["pose"], dict):
            p = obj["pose"].get("position") or obj["pose"].get("pos")
            o = obj["pose"].get("orientation") or obj["pose"].get("ori")
            return p, o
        for k in ("pos", "position", "p"):
            if k in obj:
                p = obj[k]
                o = obj.get("orientation") or obj.get("ori")
                return p, o
        return None, None

    if isinstance(obj, list):
        if len(obj) == 2 and (isinstance(obj[0], list) or isinstance(obj[0], tuple)):
            return list(obj[0]), list(obj[1])
        if len(obj) >= 3 and all(isinstance(x, (int, float)) for x in obj[:3]):
            p = [float(x) for x in obj[:3]]
            o = [float(x) for x in obj[3:]] if len(obj) > 3 else None
            return p, o

    return None, None


@dataclass
class RedisKeys:
    def __init__(self, robot_name: str):
        self.cartesian_task_goal_position = f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::goal_position"
        self.cartesian_task_goal_orientation = f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::goal_orientation"
        self.cartesian_task_current_position = f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::current_position"
        self.cartesian_task_current_orientation = f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::current_orientation"
        self.joint_task_goal_position = f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_position"
        self.joint_task_goal_velocity = f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_velocity"
        self.joint_task_goal_acceleration = f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_acceleration"
        self.active_controller = f"opensai::controllers::{robot_name}::active_controller_name"


def set_cartesian_goal(rdb, keys: RedisKeys, position, orientation):
    rdb.set(keys.cartesian_task_goal_position, json.dumps(np.asarray(position).tolist()))
    rdb.set(keys.cartesian_task_goal_orientation, json.dumps(np.asarray(orientation).tolist()))


def set_active_controller(rdb, keys: RedisKeys, controller_name):
    # Try to set until the controller reports as active
    try:
        while True:
            cur = rdb.get(keys.active_controller)
            if cur is not None and cur.decode("utf-8") == controller_name:
                break
            rdb.set(keys.active_controller, controller_name)
            time.sleep(0.01)
    except Exception:
        # if active controller key is missing, still attempt to set once
        rdb.set(keys.active_controller, controller_name)


DEFAULT_LOWERED_JOINTS = np.array([
    0.0,
    0.7853981633974483,
    0.0,
    -1.57079632679,
    0.0,
    0.7853981633974483,
    1.57079632679,
])


def set_joint_goal(rdb, keys: RedisKeys, positions):
    positions = np.asarray(positions).tolist()
    rdb.set(keys.joint_task_goal_position, json.dumps(positions))
    zero_vel = [0.0] * len(positions)
    rdb.set(keys.joint_task_goal_velocity, json.dumps(zero_vel))
    rdb.set(keys.joint_task_goal_acceleration, json.dumps(zero_vel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Run against real robot (Titania) instead of simulation")
    parser.add_argument("--ball-key", default="opensai::sensors::KendamaBall::object_pose", help="Single redis key holding pose")
    parser.add_argument("--pos-key", default=None, help="Redis key holding position array (overrides --ball-key)")
    parser.add_argument("--ori-key", default=None, help="Redis key holding orientation array (overrides --ball-key)")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--rate", type=float, default=100.0, help="Control loop rate (Hz)")
    parser.add_argument(
        "--desired-orientation",
        default=None,
        help=(
            "Fixed orientation quaternion to use (JSON or comma-separated list) instead"
            " of reading the current cartesian orientation."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=("cartesian", "joint"),
        default="cartesian",
        help="Whether to send cartesian goals (default) or update just joint 0 based on the ball",
    )
    parser.add_argument("--base-x", type=float, default=0.0, help="X location that corresponds to joint 0 zero")
    parser.add_argument("--base-y", type=float, default=0.0, help="Y location that corresponds to joint 0 zero")
    args = parser.parse_args()

    robot_name = "Titania" if args.real else "Rizon4r"
    keys = RedisKeys(robot_name)
    rdb = redis.Redis(host=args.host, port=args.port)

    cartesian_controller = "cartesian_controller"

    # Activate cartesian controller and capture robot Z and orientation
    print(f"Activating cartesian controller for {robot_name}...")
    set_active_controller(rdb, keys, cartesian_controller)
    time.sleep(0.05)

    # read current robot pose to preserve Z and orientation
    cur_pos_raw = rdb.get(keys.cartesian_task_current_position)
    cur_ori_raw = rdb.get(keys.cartesian_task_current_orientation)
    if cur_pos_raw is None:
        print("Warning: could not read current robot position; will wait until available.")
    # Try to parse
    robot_z = None
    robot_ori = None
    try:
        if cur_pos_raw is not None:
            robot_pos = json.loads(cur_pos_raw.decode('utf-8'))
            robot_z = float(robot_pos[2])
    except Exception:
        robot_z = None

    try:
        if cur_ori_raw is not None:
            robot_ori = json.loads(cur_ori_raw.decode('utf-8'))
    except Exception:
        robot_ori = None

    if args.desired_orientation:
        try:
            if args.desired_orientation.strip().startswith("["):
                robot_ori = json.loads(args.desired_orientation)
            else:
                robot_ori = [float(x) for x in args.desired_orientation.split(",")]
        except Exception as exc:
            print(f"Failed to parse desired orientation: {exc}")
            return

    # If we couldn't read Z or orientation yet, keep trying briefly
    start = time.time()
    while (robot_z is None or robot_ori is None) and (time.time() - start) < 5.0:
        cur_pos_raw = rdb.get(keys.cartesian_task_current_position)
        cur_ori_raw = rdb.get(keys.cartesian_task_current_orientation)
        try:
            if cur_pos_raw is not None:
                robot_pos = json.loads(cur_pos_raw.decode('utf-8'))
                robot_z = float(robot_pos[2])
        except Exception:
            robot_z = robot_z
        try:
            if cur_ori_raw is not None:
                robot_ori = json.loads(cur_ori_raw.decode('utf-8'))
        except Exception:
            robot_ori = robot_ori
        time.sleep(0.05)

    if robot_z is None or robot_ori is None:
        print("Error: unable to determine robot Z/orientation from Redis. Exiting.")
        return

    print(f"Using robot Z={robot_z} and orientation={robot_ori}")

    dt = 1.0 / float(args.rate) if args.rate > 0 else 0.01

    if args.control_mode == "joint":
        joint_controller = "joint_controller"
        print("Switching to joint controller so only joint 0 moves.")
        set_active_controller(rdb, keys, joint_controller)
        set_joint_goal(rdb, keys, DEFAULT_LOWERED_JOINTS)

    try:
        print(f"Following ball key(s): {'pos='+args.pos_key if args.pos_key else args.ball_key}{', ori='+args.ori_key if args.ori_key else ''}")
        while True:
            # read ball pose
            pos = None
            ori = None
            if args.pos_key or args.ori_key:
                if args.pos_key:
                    raw_p = rdb.get(args.pos_key)
                    try:
                        pos = json.loads(raw_p.decode('utf-8')) if raw_p is not None else None
                    except Exception:
                        pos = None
                if args.ori_key:
                    raw_o = rdb.get(args.ori_key)
                    try:
                        ori = json.loads(raw_o.decode('utf-8')) if raw_o is not None else None
                    except Exception:
                        ori = None
            else:
                raw = rdb.get(args.ball_key)
                pos, ori = parse_pose_raw(raw)

            if pos is None:
                # nothing to do this loop
                time.sleep(dt)
                continue

            if args.control_mode == "joint":
                target_angle = math.atan2(pos[1] - args.base_y, pos[0] - args.base_x)
                target_joints = DEFAULT_LOWERED_JOINTS.copy()
                target_joints[0] = float(target_angle)
                set_joint_goal(rdb, keys, target_joints)
                time.sleep(dt)
                continue

            # construct target: match ball x,y, preserve robot_z
            target = [float(pos[0]), float(pos[1]), float(robot_z)]
            # keep orientation as robot_ori
            set_cartesian_goal(rdb, keys, target, robot_ori)
            time.sleep(dt)

    except KeyboardInterrupt:
        print("Interrupted by user. Stopping.")


if __name__ == "__main__":
    main()
