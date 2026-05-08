import numpy as np
import time
import json
import redis
import math
from enum import Enum, auto
from dataclasses import dataclass

DEG_TO_RAD = math.pi / 180.0

class State(Enum):
  RESETTING_JOINTS = auto()
  GOING_UP = auto()

real_robot_name = "Titania"
simulation_robot_name = "Rizon4r"

real_robot_config_file = "single_rizon_real.xml"
simulation_config_file = "single_rizon_vis.xml"
ENV = "real" # "real" or "simulation"
if ENV == "real":
  robot_name = "Titania"
  config_file_for_this_example = real_robot_config_file
else:
  robot_name = "Rizon4r"
  config_file_for_this_example = simulation_config_file

# Step 1:
# sh scripts/launch.sh config_folder/xml_config_files/single_rizon_real.xml
# Step 2: run this script 
# python3 python_examples/rizon_up.py

# To run in real:
# Step 0: 
# Make sure the robot is powered on and the estop is released. Also make sure to have the safety key with you.
# Start redis server: src1@src1-GTi:~/OpenSai/drivers/FlexivRizonRedisDriver/redis_driver$ sh launch_titania-4s_gripper_driver.sh 
# and then do the other thing

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
  joint_task_integration_gain: str = f"opensai::controllers::{robot_name}::joint_controller::joint_task::integration_gain"

  active_controller: str = f"opensai::controllers::{robot_name}::active_controller_name"
  config_file_name: str = "::sai-interfaces-webui::config_file_name"
redis_keys = RedisKeys()



joint_controller = "joint_controller"
cartesian_controller = "cartesian_controller"

default_joint_pos = np.array([0.0, -0.6, 0.0, 1.6, 0.0, 1.0, 0.0])
up_goal_height = 0.8

joint_integration_threshold = 0.12
joint_arrival_threshold = 8e-2
height_arrival_threshold = 2e-2

# redis client
redis_client = redis.Redis()

# check that the config file is correct
config_file_name = redis_client.get(redis_keys.config_file_name).decode("utf-8")
if config_file_name != config_file_for_this_example:
    print("This example is meant to be used with the config file: ", config_file_for_this_example)
    print("But instead you have: ", config_file_name)
    exit(0)

def set_cartesian_goal(position, orientation):
  redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(position.tolist()))
  redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(orientation.tolist()))

def set_joint_goal(position):
  redis_client.set(redis_keys.joint_task_goal_position, json.dumps(position.tolist()))
  redis_client.set(redis_keys.joint_task_goal_velocity, json.dumps(np.zeros_like(position).tolist()))
  redis_client.set(redis_keys.joint_task_goal_acceleration, json.dumps(np.zeros_like(position).tolist()))

def set_active_controller(controller_name):
  while redis_client.get(redis_keys.active_controller).decode("utf-8") != controller_name:
    redis_client.set(redis_keys.active_controller, controller_name)
    time.sleep(0.001)



# loop at 200 Hz
loop_time = 0.0
dt = 0.005
state = State.RESETTING_JOINTS
up_goal_pos = None
up_goal_ori = None

time.sleep(0.01)
init_time = time.perf_counter_ns() * 1e-9

try:
  while True:
    loop_time += dt
    time.sleep(max(0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))
    
    # state machine
    if state == State.RESETTING_JOINTS:
      print("Resetting joints...")
      set_active_controller(joint_controller)
      set_joint_goal(default_joint_pos)
      # monitor error
      current_joint_position = np.array(json.loads(redis_client.get(redis_keys.joint_task_current_position)))
      joint_error = np.linalg.norm(default_joint_pos - current_joint_position)
      print("joint error is: ", joint_error, ", threshold is: ", joint_arrival_threshold)

      # if joint_error < joint_integration_threshold:
        # redis_client.set(redis_keys.joint_task_integration_gain, json.dumps([10.]))

      if joint_error < joint_arrival_threshold:
        print("inside if statement...")
        current_position = np.array(json.loads(redis_client.get(redis_keys.cartesian_task_current_position)))
        current_orientation = np.array(json.loads(redis_client.get(redis_keys.cartesian_task_current_orientation)))
        up_goal_pos = current_position.copy()
        up_goal_pos[2] = up_goal_height
        up_goal_ori = current_orientation

        set_cartesian_goal(current_position, current_orientation)
        set_active_controller(cartesian_controller)
        set_cartesian_goal(up_goal_pos, up_goal_ori)
        # redis_client.set(redis_keys.joint_task_integration_gain, json.dumps([0.]))
        state = State.GOING_UP
        print("Default joint position reached. Going Up")

    elif state == State.GOING_UP:
      # monitor error
      current_position = np.array(json.loads(redis_client.get(redis_keys.cartesian_task_current_position)))
      height_error = abs(up_goal_pos[2] - current_position[2])
      set_cartesian_goal(up_goal_pos, up_goal_ori)
      if height_error < height_arrival_threshold:
        # set_cartesian_goal(up_goal_pos, up_goal_ori)
        print("Reached 0.8 m height. Stopping.")
        break

except KeyboardInterrupt:
  print("Keyboard interrupt")
  pass
except Exception as e:
  print(e)
  pass
