# Step 1:
# sh scripts/launch.sh config_folder/xml_config_files/single_rizon_real.xml
# Step 2: run this script 
# python3 python_examples/rizon_up.py
# optional argument: --real, that runs the real robot
# usage: python3 python_examples/rizon_up.py --real, otherwise defaults to simulation

# To run in real:
# Step 0: 
# Make sure the robot is powered on and the estop is released. Also make sure to have the safety key with you.
# Start redis server: src1@src1-GTi:~/OpenSai/drivers/FlexivRizonRedisDriver/redis_driver$ sh launch_titania-4s_gripper_driver.sh 
# and then do the other thing

import argparse
import json
import math
import time
from enum import Enum, auto

import numpy as np
import redis


# TUNABLE TRICK PARAMETERS
MAX_SWING_DEG          = 0.0   # +/- degrees about world z (observed safe limit)
SWING_PERIOD           = 10.0    # seconds per back-and-forth cycle
VERTICAL_OSC_AMPLITUDE = 0.1    # meters; set to 0.0 to disable the bounce
VERTICAL_OSC_PERIOD    = 5.0    # seconds per up-down cycle


# REAL HARDWARE
parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true",
                    help="Run against the real robot (Titania).")
args = parser.parse_args()

if args.real:
    robot_name   = "Titania"
    expected_cfg = "single_rizon_real.xml"
else:
    robot_name   = "Rizon4r"
    expected_cfg = "single_rizon_vis.xml"


# ROBOT CONSTANTS
LOWERED_START_JOINT_POS = np.array([0.0,
    -0.6981317007977318, # =40 degrees
    0.0,
    1.57079632679, # =90 degrees
    0.0,
    2.2689280275926285, # =130 degrees
    0.0])
JOINT_ARRIVAL_TOL       = 3e-2   # L2 norm across the 7 joints (radians)


# REDIS KEYS
_PREFIX        = f"opensai::controllers::{robot_name}"
KEY_GOAL_POS   = f"{_PREFIX}::cartesian_controller::cartesian_task::goal_position"
KEY_GOAL_ORI   = f"{_PREFIX}::cartesian_controller::cartesian_task::goal_orientation"
KEY_CUR_POS    = f"{_PREFIX}::cartesian_controller::cartesian_task::current_position"
KEY_CUR_ORI    = f"{_PREFIX}::cartesian_controller::cartesian_task::current_orientation"
KEY_JOINT_GOAL = f"{_PREFIX}::joint_controller::joint_task::goal_position"
KEY_JOINT_CUR  = f"{_PREFIX}::joint_controller::joint_task::current_position"
KEY_ACTIVE     = f"{_PREFIX}::active_controller_name"
KEY_CONFIG     = "::sai-interfaces-webui::config_file_name"


# REDIS HELPERS
r = redis.Redis()

def set_cartesian_goal(position, orientation):
    """Write a Cartesian-space target (pos in meters, ori as a 3x3 matrix)."""
    r.set(KEY_GOAL_POS, json.dumps(np.asarray(position).tolist()))
    r.set(KEY_GOAL_ORI, json.dumps(np.asarray(orientation).tolist()))

def set_joint_goal(position):
    """Write a 7-vector joint-space target (radians)."""
    r.set(KEY_JOINT_GOAL, json.dumps(position.tolist()))

def set_active_controller(name):
    """Switch active controller, retrying until OpenSai confirms the switch."""
    while r.get(KEY_ACTIVE).decode() != name:
        r.set(KEY_ACTIVE, name)
        time.sleep(0.001)


# STATE MACHINE
# Three phases, each entered exactly once:
#   RESETTING_JOINTS       - wait until the joint controller parks at start
#   SWITCHING_TO_CARTESIAN - switch controllers and latch the cup's pose
#   RISING_DRAGON          - run the swing + bounce loop until Ctrl+C
class State(Enum):
    RESETTING_JOINTS       = auto()
    SWITCHING_TO_CARTESIAN = auto()
    RISING_DRAGON          = auto()


# VERIFY CONFIG
cfg = r.get(KEY_CONFIG)
if cfg is None or cfg.decode() != expected_cfg:
    print(f"Expected config:  {expected_cfg}")
    print(f"Running config:   {cfg.decode() if cfg else 'None'}")
    print(f"Launch the simulator first:  sh scripts/launch.sh {expected_cfg}")
    raise SystemExit(1)


# DERIVED CONSTANTS
omega_swing   = 2.0 * math.pi / SWING_PERIOD         # rad/s, orientation swing
omega_osc     = 2.0 * math.pi / VERTICAL_OSC_PERIOD  # rad/s, vertical bounce
max_swing_rad = math.radians(MAX_SWING_DEG)
dt            = 0.005                                # 200 Hz update rate


# MAIN LOOP
print("=" * 60)
print(f"RISING DRAGON  —  {robot_name} ({'real' if args.real else 'sim'})")
print(f"  Swing amplitude: +/-{MAX_SWING_DEG:.0f} deg, period {SWING_PERIOD:.1f} s")
print(f"  Vertical osc:    +/-{VERTICAL_OSC_AMPLITUDE*100:.1f} cm, period {VERTICAL_OSC_PERIOD:.1f} s")
print("=" * 60)

set_active_controller("joint_controller")
set_joint_goal(LOWERED_START_JOINT_POS)
print("Resetting to start joint pose...")

state            = State.RESETTING_JOINTS
cup_init_pos     = None
cup_init_ori     = None
trick_start_time = None

try:
    while True:
        time.sleep(dt)

        # ---- Phase 1: wait for the arm to reach the start pose -------------
        if state == State.RESETTING_JOINTS:
            q = np.array(json.loads(r.get(KEY_JOINT_CUR)))
            if np.linalg.norm(LOWERED_START_JOINT_POS - q) < JOINT_ARRIVAL_TOL:
                print("Start pose reached. Switching to Cartesian controller.")
                state = State.SWITCHING_TO_CARTESIAN

        # ---- Phase 2: switch controllers and latch the cup pose ------------
        elif state == State.SWITCHING_TO_CARTESIAN:
            set_active_controller("cartesian_controller")
            time.sleep(0.2)  # let current_position/orientation publish

            cup_init_pos = np.array(json.loads(r.get(KEY_CUR_POS)))
            cup_init_ori = np.array(json.loads(r.get(KEY_CUR_ORI)))

            # Pin the goal to the current pose so the controller doesn't lunge.
            set_cartesian_goal(cup_init_pos, cup_init_ori)

            print(f"  Cup start pos: {np.round(cup_init_pos, 4).tolist()}")
            print("Starting Rising Dragon. Ctrl+C to stop.")
            trick_start_time = time.perf_counter()
            state = State.RISING_DRAGON

        # ---- Phase 3: swing the cup orientation about world z --------------
        # theta(t) = A * sin(omega * t) bounds the joint travel to +/-A,
        # so no joint can wind past its limit the way continuous rotation
        # would. Left-multiplying R_z (world-frame rotation) keeps the
        # cup's opening pointing world +z because R_z @ [0,0,1] = [0,0,1].
        elif state == State.RISING_DRAGON:
            t     = time.perf_counter() - trick_start_time
            theta = max_swing_rad * math.sin(omega_swing * t)
            c, s  = math.cos(theta), math.sin(theta)
            R_z   = np.array([[c, -s, 0],
                              [s,  c, 0],
                              [0,  0, 1]])

            dz      = VERTICAL_OSC_AMPLITUDE * math.sin(omega_osc * t)
            cup_pos = cup_init_pos + np.array([0.0, 0.0, dz])
            cup_ori = R_z @ cup_init_ori

            set_cartesian_goal(cup_pos, cup_ori)

except KeyboardInterrupt:
    print("\nStopping. Returning to home pose.")
    set_active_controller("joint_controller")
    set_joint_goal(LOWERED_START_JOINT_POS)
