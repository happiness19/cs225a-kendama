import numpy as np
import time
import json
import redis
import math
import argparse
from enum import Enum, auto
from dataclasses import dataclass

DEG_TO_RAD = math.pi / 180.0

class State(Enum):
    RESETTING_JOINTS = auto()
    IDLE = auto()
    JOLT_UP = auto()
    JOLT_DOWN = auto()


real_robot_name = "Titania"
simulation_robot_name = "Rizon4r"

real_robot_config_file = "single_rizon_real.xml"
simulation_config_file = "single_rizon_vis.xml"
parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true", help="Run against the real robot instead of simulation.")
args = parser.parse_args()

ENV = "real" if args.real else "simulation"
if ENV == "real":
    robot_name = "Titania"
    config_file_for_this_example = real_robot_config_file
else:
    robot_name = "Rizon4r"
    config_file_for_this_example = simulation_config_file

@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::goal_position"
    cartesian_task_goal_orientation: str = f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::goal_orientation"
    cartesian_task_current_position: str = f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::current_position"
    cartesian_task_current_orientation: str = f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::current_orientation"
    joint_task_goal_position: str = f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_position"
    joint_task_goal_velocity: str = f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_velocity"
    joint_task_goal_acceleration: str = f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_acceleration"
    joint_task_current_position: str = f"opensai::controllers::{robot_name}::joint_controller::joint_task::current_position"
    ball_pose: str = "opensai::sensors::KendamaBall::object_pose"
    ball_velocity: str = "opensai::sensors::KendamaBall::object_velocity"
    active_controller: str = f"opensai::controllers::{robot_name}::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"

redis_keys = RedisKeys()

joint_controller = "joint_controller"
cartesian_controller = "cartesian_controller"

# Joint position with end-effector pointing downward
# Joint 2 and 4 at -90 degrees to point end-effector straight down
LOWERED_START_JOINT_POS_PERP_Y = np.array([
    0.0,
    0.0,
    0.0,
    1.57079632679,
    0.0,
    0.0,
    0.0,
])

LOWERED_START_JOINT_POS_PARALLEL_Y = np.array([
    0.0,
    0.7853981633974483,
    0.0,
    -1.57079632679,
    0.0,
    0.7853981633974483,
    1.57079632679,
])

default_joint_pos = LOWERED_START_JOINT_POS_PARALLEL_Y

# Motion + timing parameters
joint_arrival_threshold = 3e-2
idle_hold_duration = 1.0
jolt_height = 0.05         # meters cup rises during JOLT_UP
jolt_up_duration = 0.15     # seconds spent commanding the upward target
landing_dip = 0.02          # meters cup dips below rest to damp landing
jolt_down_duration = 0.35   # seconds to hold the dip
settle_duration = 0.5       # seconds to blend back to rest height
auto_repeat = False         # set True to loop continuously

# Pose bookkeeping
rest_cup_pos = None
rest_cup_ori = None

# State tracking
state = State.RESETTING_JOINTS
state_entry_time = 0.0

cycle_requested = True  # run one jolt sequence after reset by default

# redis client
redis_client = redis.Redis()

# check that the config file is correct
config_file_name = redis_client.get(redis_keys.config_file_name).decode("utf-8")
if config_file_name != config_file_for_this_example:
    print("This example is meant to be used with the config file: ", config_file_for_this_example)
    print("But instead you have: ", config_file_name)
    exit(0)

if default_joint_pos is None:
    default_joint_pos = LOWERED_START_JOINT_POS.copy()
    print("Default joint pose not set; using lowered start pose.")
else:
    print("Using lowered start joint pose:", default_joint_pos)

def get_pose_from_redis(position_key, orientation_key):
    position = np.array(json.loads(redis_client.get(position_key)))
    orientation = np.array(json.loads(redis_client.get(orientation_key)))
    return position, orientation


def set_cartesian_goal(position, orientation):
    redis_client.set(
        redis_keys.cartesian_task_goal_position,
        json.dumps(np.asarray(position).tolist()),
    )
    redis_client.set(
        redis_keys.cartesian_task_goal_orientation,
        json.dumps(np.asarray(orientation).tolist()),
    )


def set_joint_goal(position):
    redis_client.set(redis_keys.joint_task_goal_position, json.dumps(position.tolist()))
    redis_client.set(
        redis_keys.joint_task_goal_velocity,
        json.dumps(np.zeros_like(position).tolist()),
    )
    redis_client.set(
        redis_keys.joint_task_goal_acceleration,
        json.dumps(np.zeros_like(position).tolist()),
    )


def set_active_controller(controller_name):
    while redis_client.get(redis_keys.active_controller).decode("utf-8") != controller_name:
        redis_client.set(redis_keys.active_controller, controller_name)
        time.sleep(0.001)


# loop at 200 Hz
loop_time = 0.0
dt = 0.005

time.sleep(0.01)
init_time = time.perf_counter_ns() * 1e-9

print("=" * 60)
print("KENDAMA JOLT CONTROLLER")
print("=" * 60)

# Start in joint control mode to reset to known position
set_active_controller(joint_controller)
set_joint_goal(default_joint_pos)

print("Resetting to default joint position...")


try:
    while True:
        loop_time += dt
        time.sleep(max(0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))

        if state == State.RESETTING_JOINTS:
            current_joint_position = np.array(
                json.loads(redis_client.get(redis_keys.joint_task_current_position))
            )
            joint_error = np.linalg.norm(default_joint_pos - current_joint_position)
            if joint_error < joint_arrival_threshold:
                print("Default joint position reached. Capturing cup pose and going idle.")
                set_active_controller(cartesian_controller)
                time.sleep(0.1)
                rest_cup_pos = np.array(
                    json.loads(redis_client.get(redis_keys.cartesian_task_current_position))
                )
                rest_cup_ori = np.array(
                    json.loads(redis_client.get(redis_keys.cartesian_task_current_orientation))
                )
                set_cartesian_goal(rest_cup_pos, rest_cup_ori)
                print(f"kendama_big_cup rest position: {rest_cup_pos.tolist()}")
                print(f"kendama_big_cup rest orientation:\n{rest_cup_ori}")
                state = State.IDLE
                state_entry_time = loop_time

        elif state == State.IDLE:
            if rest_cup_pos is None:
                continue
            set_cartesian_goal(rest_cup_pos, rest_cup_ori)
            if cycle_requested and (loop_time - state_entry_time) > idle_hold_duration:
                cycle_requested = auto_repeat  # prevent auto-repeat unless requested
                state = State.JOLT_UP
                state_entry_time = loop_time
                print("Starting JOLT_UP.")

        elif state == State.JOLT_UP:
            # Pure vertical motion: keep XY and orientation locked to the rest pose.
            if rest_cup_pos is None:
                continue
            up_goal = np.array([rest_cup_pos[0], rest_cup_pos[1], rest_cup_pos[2] + jolt_height])
            set_cartesian_goal(up_goal, rest_cup_ori)
            if (loop_time - state_entry_time) > jolt_up_duration:
                state = State.JOLT_DOWN
                state_entry_time = loop_time
                print("Switching to JOLT_DOWN for landing dampening.")

        elif state == State.JOLT_DOWN:
            # Stay on the same XY column; only move along world Z.
            if rest_cup_pos is None:
                continue
            elapsed = loop_time - state_entry_time
            dip_z = rest_cup_pos[2] - landing_dip
            if elapsed < jolt_down_duration:
                target_z = dip_z
            elif elapsed < jolt_down_duration + settle_duration:
                alpha = (elapsed - jolt_down_duration) / settle_duration
                target_z = dip_z * (1 - alpha) + rest_cup_pos[2] * alpha
            else:
                print("Landing dampened. Returning to IDLE.")
                state = State.IDLE
                state_entry_time = loop_time
                continue
            target = np.array([rest_cup_pos[0], rest_cup_pos[1], target_z])
            set_cartesian_goal(target, rest_cup_ori)

except KeyboardInterrupt:
    print("\nKeyboard interrupt - stopping controller")
    pass
except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()
    pass
