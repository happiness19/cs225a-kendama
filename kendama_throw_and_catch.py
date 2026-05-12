import numpy as np
import time
import json
import redis
import math
import argparse
from enum import Enum, auto
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

DEG_TO_RAD = math.pi / 180.0

class State(Enum):
    RESETTING_JOINTS = auto()
    IDLE = auto()
    THROW = auto()


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
LOWERED_START_JOINT_POS = np.array([0.0, 0.7, 0.0, 1.3, 0.0, 1.5, 1.5])
default_joint_pos = LOWERED_START_JOINT_POS

# Motion parameters
motion_duration = 2.0  # seconds for smooth motion
joint_arrival_threshold = 3e-2

# Define absolute world positions for up and down motion
up_position = np.array([0.248, 0.125, 0.808])  # Starting high position
down_position = np.array([0.248, 0.125, 0.308])  # Ending low position (0.5m down)

# State tracking
state = State.RESETTING_JOINTS
motion_start_time = None
motion_start_pos = None
motion_target_pos = None
motion_orientation = None

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
    redis_client.set(redis_keys.joint_task_goal_velocity, json.dumps(np.zeros_like(position).tolist()))
    redis_client.set(redis_keys.joint_task_goal_acceleration, json.dumps(np.zeros_like(position).tolist()))

def set_active_controller(controller_name):
    while redis_client.get(redis_keys.active_controller).decode("utf-8") != controller_name:
        redis_client.set(redis_keys.active_controller, controller_name)
        time.sleep(0.001)

def get_ball_position():
    try:
        ball_pose_matrix = np.array(json.loads(redis_client.get(redis_keys.ball_pose)))
        return ball_pose_matrix[0:3, 3]
    except:
        return None

def get_ball_velocity():
    try:
        ball_velocity = np.array(json.loads(redis_client.get(redis_keys.ball_velocity)))
        return ball_velocity[0:3]
    except:
        return None

def predict_ball_position(*args, **kwargs):
    raise NotImplementedError("Ball prediction is disabled in idle mode.")

# loop at 200 Hz
loop_time = 0.0
dt = 0.005

time.sleep(0.01)
init_time = time.perf_counter_ns() * 1e-9

print("=" * 60)
print("KENDAMA THROW CATCH CONTROLLER")
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
            # Wait for joints to reach default position, then stay idle
            current_joint_position = np.array(
                json.loads(redis_client.get(redis_keys.joint_task_current_position))
            )
            joint_error = np.linalg.norm(default_joint_pos - current_joint_position)
            if joint_error < joint_arrival_threshold:
                print("Default joint position reached. Holding pose (idle).")
                # Briefly activate cartesian controller so kendama_big_cup
                # current_position gets published to redis.
                try:
                    set_active_controller(cartesian_controller)
                    time.sleep(0.2)
                    cup_pos = np.array(json.loads(
                        redis_client.get(redis_keys.cartesian_task_current_position)
                    ))
                    cup_ori = np.array(json.loads(
                        redis_client.get(redis_keys.cartesian_task_current_orientation)
                    ))
                    print(f"kendama_big_cup world position: {cup_pos.tolist()}")
                    print(f"kendama_big_cup world orientation:\n{cup_ori}")
                    set_active_controller(joint_controller)
                except Exception as e:
                    print(f"Could not read big cup position: {e}")
                state = State.IDLE

        elif state == State.IDLE:
            # Hold current position - joint controller maintains last goal
            pass

except KeyboardInterrupt:
    print("\n\nKeyboard interrupt - stopping simulation")
    pass
except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()
    pass

