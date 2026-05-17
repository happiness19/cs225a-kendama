import argparse
import json
import math
import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import redis


class State(Enum):
    RESETTING_JOINTS = auto()
    SETTLING_START_POSE = auto()
    THROWING_UP = auto()
    RELEASING_BALL = auto()
    FLIPPING_TO_SMALL_CUP = auto()
    TRACKING_DESCENT = auto()
    HOLDING_CATCH = auto()


real_robot_config_file = "single_rizon_real.xml"
simulation_config_file = "single_rizon_vis_y.xml"

parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true", help="Run against the real robot instead of simulation.")
args = parser.parse_args()

robot_name = "Titania" if args.real else "Rizon4r"
config_file_for_this_example = real_robot_config_file if args.real else simulation_config_file


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::goal_orientation"
    )
    cartesian_task_current_position: str = (
        f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::current_position"
    )
    cartesian_task_current_orientation: str = (
        f"opensai::controllers::{robot_name}::cartesian_controller::cartesian_task::current_orientation"
    )
    joint_task_goal_position: str = (
        f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_position"
    )
    joint_task_goal_velocity: str = (
        f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_velocity"
    )
    joint_task_goal_acceleration: str = (
        f"opensai::controllers::{robot_name}::joint_controller::joint_task::goal_acceleration"
    )
    joint_task_current_position: str = (
        f"opensai::controllers::{robot_name}::joint_controller::joint_task::current_position"
    )
    ball_pose: str = "opensai::sensors::KendamaBall::object_pose"
    ball_velocity: str = "opensai::sensors::KendamaBall::object_velocity"
    active_controller: str = f"opensai::controllers::{robot_name}::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"


redis_keys = RedisKeys()
redis_client = redis.Redis()

joint_controller = "joint_controller"
cartesian_controller = "cartesian_controller"

# Matches the simulator spawn pose noted in world_single_rizon.urdf.
START_JOINT_POS = np.array([0.0, 0.7, 0.0, 1.3, 0.0, 1.5, 1.5])

JOINT_ARRIVAL_THRESHOLD = 3e-2
POSITION_ARRIVAL_THRESHOLD = 2.5e-2

SMALL_CUP_OFFSET_IN_BIG_CUP_FRAME = np.array([0.0, 0.08, 0.0])
BALL_CENTER_ABOVE_CUP = 0.039

SETTLE_DURATION = 0.35
THROW_DURATION = 0.22
RELEASE_DURATION = 0.08
FLIP_DURATION = 0.22
TRACK_TIMEOUT = 1.60
HOLD_DURATION = 1.00

THROW_HEIGHT = 0.42
RELEASE_DROP = 0.08
CATCH_CENTER_Z_BIAS = 0.080
CATCH_XY_LEAD_TIME = 0.18
MAX_TRACK_XY_STEP = 0.015
GRAVITY = 9.81
CATCH_TARGET_XY_BIAS = np.array([-0.070, -0.060])
ACTUAL_CATCH_HEIGHT_WINDOW = 0.040


def require_value(raw, name):
    if raw is None:
        raise RuntimeError(f"Missing redis value for {name}.")
    return raw


def load_json_key(key, name):
    return np.array(json.loads(require_value(redis_client.get(key), name)))


def ball_snapshot():
    ball_pose = load_json_key(redis_keys.ball_pose, redis_keys.ball_pose)
    ball_velocity = load_json_key(redis_keys.ball_velocity, redis_keys.ball_velocity)
    return ball_pose[0:3, 3], ball_velocity[0:3]


def print_ball_snapshot(label):
    ball_position, ball_velocity = ball_snapshot()
    print(
        f"{label}: ball_pos={ball_position.round(4).tolist()} "
        f"ball_vel={ball_velocity.round(4).tolist()}"
    )


def print_tracking_snapshot(label):
    ball_position, ball_velocity = ball_snapshot()
    current_big_cup_position = load_json_key(
        redis_keys.cartesian_task_current_position,
        redis_keys.cartesian_task_current_position,
    )
    current_big_cup_orientation = load_json_key(
        redis_keys.cartesian_task_current_orientation,
        redis_keys.cartesian_task_current_orientation,
    )
    current_small_cup_center = small_cup_center(
        current_big_cup_position,
        current_big_cup_orientation,
    )
    desired_ball_center = current_small_cup_center + np.array([0.0, 0.0, BALL_CENTER_ABOVE_CUP])
    error = np.linalg.norm(ball_position - desired_ball_center)
    print(
        f"{label}: small_cup_center={current_small_cup_center.round(4).tolist()} "
        f"ball_pos={ball_position.round(4).tolist()} "
        f"ball_vel={ball_velocity.round(4).tolist()} "
        f"alignment_error={error:.4f} m"
    )


def predict_descent_time_to_height(ball_position, ball_velocity, target_height):
    height_delta = target_height - ball_position[2]
    discriminant = ball_velocity[2] ** 2 - 2.0 * GRAVITY * height_delta
    if discriminant < 0.0:
        return 0.0
    sqrt_discriminant = math.sqrt(discriminant)
    roots = [
        (ball_velocity[2] - sqrt_discriminant) / GRAVITY,
        (ball_velocity[2] + sqrt_discriminant) / GRAVITY,
    ]
    positive_roots = [root for root in roots if root > 0.0]
    return max(positive_roots) if positive_roots else 0.0


def set_active_controller(controller_name):
    while require_value(redis_client.get(redis_keys.active_controller), redis_keys.active_controller).decode(
        "utf-8"
    ) != controller_name:
        redis_client.set(redis_keys.active_controller, controller_name)
        time.sleep(0.001)


def set_joint_goal(position):
    position = np.asarray(position)
    redis_client.set(redis_keys.joint_task_goal_position, json.dumps(position.tolist()))
    redis_client.set(redis_keys.joint_task_goal_velocity, json.dumps(np.zeros_like(position).tolist()))
    redis_client.set(redis_keys.joint_task_goal_acceleration, json.dumps(np.zeros_like(position).tolist()))


def set_cartesian_goal(position, orientation):
    redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(np.asarray(position).tolist()))
    redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(np.asarray(orientation).tolist()))


def smoothstep(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def interpolate_position(start, goal, alpha):
    blend = smoothstep(alpha)
    return (1.0 - blend) * start + blend * goal


def rotation_about_local_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def interpolate_flip(start_orientation, alpha):
    return start_orientation @ rotation_about_local_x(math.pi * smoothstep(alpha))


def small_cup_center(big_cup_position, big_cup_orientation):
    return big_cup_position + big_cup_orientation @ SMALL_CUP_OFFSET_IN_BIG_CUP_FRAME


def big_cup_position_for_small_cup_center(small_cup_target, big_cup_orientation):
    return small_cup_target - big_cup_orientation @ SMALL_CUP_OFFSET_IN_BIG_CUP_FRAME


def limited_xy_step(current_target, desired_target, max_step):
    next_target = current_target.copy()
    xy_delta = desired_target[:2] - current_target[:2]
    xy_norm = np.linalg.norm(xy_delta)
    if xy_norm > max_step and xy_norm > 1e-9:
        xy_delta = xy_delta * (max_step / xy_norm)
    next_target[:2] += xy_delta
    next_target[2] = desired_target[2]
    return next_target


config_file_name = require_value(redis_client.get(redis_keys.config_file_name), redis_keys.config_file_name).decode(
    "utf-8"
)
if config_file_name != config_file_for_this_example:
    print("This example is meant to be used with the config file:", config_file_for_this_example)
    print("But instead you have:", config_file_name)
    raise SystemExit(0)


state = State.RESETTING_JOINTS
loop_time = 0.0
dt = 0.005
state_start_time = None

start_big_cup_position = None
start_big_cup_orientation = None
throw_goal_position = None
release_goal_position = None
flipped_orientation = None
catch_small_cup_center = None
catch_big_cup_position = None
tracking_small_cup_target = None

time.sleep(0.01)
init_time = time.perf_counter_ns() * 1e-9

set_active_controller(joint_controller)
set_joint_goal(START_JOINT_POS)

print("=" * 60)
print("KENDAMA CUP FLIP CONTROLLER")
print("=" * 60)
print("Resetting to launch posture...")

try:
    while True:
        loop_time += dt
        time.sleep(max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))
        now = time.perf_counter_ns() * 1e-9

        if state == State.RESETTING_JOINTS:
            current_joint_position = load_json_key(
                redis_keys.joint_task_current_position,
                redis_keys.joint_task_current_position,
            )
            joint_error = np.linalg.norm(START_JOINT_POS - current_joint_position)
            if joint_error < JOINT_ARRIVAL_THRESHOLD:
                # The Cartesian task publishes its live link pose only after activation.
                set_active_controller(cartesian_controller)
                time.sleep(0.20)
                start_big_cup_position = load_json_key(
                    redis_keys.cartesian_task_current_position,
                    redis_keys.cartesian_task_current_position,
                )
                start_big_cup_orientation = load_json_key(
                    redis_keys.cartesian_task_current_orientation,
                    redis_keys.cartesian_task_current_orientation,
                )

                throw_goal_position = start_big_cup_position.copy()
                throw_goal_position[2] += THROW_HEIGHT
                release_goal_position = throw_goal_position.copy()
                release_goal_position[2] -= RELEASE_DROP

                flipped_orientation = start_big_cup_orientation @ rotation_about_local_x(math.pi)
                catch_small_cup_center = start_big_cup_position.copy()
                catch_small_cup_center[2] += CATCH_CENTER_Z_BIAS
                catch_big_cup_position = big_cup_position_for_small_cup_center(
                    catch_small_cup_center,
                    flipped_orientation,
                )
                tracking_small_cup_target = catch_small_cup_center.copy()

                set_cartesian_goal(start_big_cup_position, start_big_cup_orientation)
                state_start_time = now
                state = State.SETTLING_START_POSE
                print("Launch posture reached. Settling before the throw.")
                print_ball_snapshot("Start")

        elif state == State.SETTLING_START_POSE:
            set_cartesian_goal(start_big_cup_position, start_big_cup_orientation)
            if now - state_start_time >= SETTLE_DURATION:
                state_start_time = now
                state = State.THROWING_UP
                print("Throwing the ball upward.")

        elif state == State.THROWING_UP:
            alpha = (now - state_start_time) / THROW_DURATION
            throw_position = interpolate_position(start_big_cup_position, throw_goal_position, alpha)
            set_cartesian_goal(throw_position, start_big_cup_orientation)
            if alpha >= 1.0:
                ball_position, ball_velocity = ball_snapshot()
                target_ball_height = catch_small_cup_center[2] + BALL_CENTER_ABOVE_CUP
                predicted_catch_time = predict_descent_time_to_height(
                    ball_position,
                    ball_velocity,
                    target_ball_height,
                )
                predicted_catch_time = float(np.clip(predicted_catch_time, 0.0, 0.60))
                catch_small_cup_center[:2] = (
                    ball_position[:2] + predicted_catch_time * ball_velocity[:2]
                )
                catch_small_cup_center[:2] += CATCH_TARGET_XY_BIAS
                catch_big_cup_position = big_cup_position_for_small_cup_center(
                    catch_small_cup_center,
                    flipped_orientation,
                )
                tracking_small_cup_target = catch_small_cup_center.copy()
                state_start_time = now
                state = State.RELEASING_BALL
                print_ball_snapshot("After throw")
                print(
                    "Predicted small-cup catch center:",
                    catch_small_cup_center.round(4).tolist(),
                    f"in {predicted_catch_time:.3f} s",
                )
                print("Dropping the throwing cup away from the ball.")

        elif state == State.RELEASING_BALL:
            alpha = (now - state_start_time) / RELEASE_DURATION
            release_position = interpolate_position(throw_goal_position, release_goal_position, alpha)
            set_cartesian_goal(release_position, start_big_cup_orientation)
            if alpha >= 1.0:
                state_start_time = now
                state = State.FLIPPING_TO_SMALL_CUP
                print_ball_snapshot("Before flip")
                print("Rolling the kendama to present the small cup.")

        elif state == State.FLIPPING_TO_SMALL_CUP:
            alpha = (now - state_start_time) / FLIP_DURATION
            flip_orientation = interpolate_flip(start_big_cup_orientation, alpha)
            set_cartesian_goal(catch_big_cup_position, flip_orientation)
            if alpha >= 1.0:
                state_start_time = now
                state = State.TRACKING_DESCENT
                print_ball_snapshot("After flip")
                print_tracking_snapshot("Flip exit")
                print("Tracking the falling ball for the small-cup catch.")

        elif state == State.TRACKING_DESCENT:
            ball_pose = load_json_key(redis_keys.ball_pose, redis_keys.ball_pose)
            ball_velocity = load_json_key(redis_keys.ball_velocity, redis_keys.ball_velocity)
            ball_position = ball_pose[0:3, 3]
            linear_velocity = ball_velocity[0:3]

            desired_small_cup_center = tracking_small_cup_target.copy()
            desired_small_cup_center[:2] = ball_position[:2] + CATCH_XY_LEAD_TIME * linear_velocity[:2]
            desired_small_cup_center[2] = catch_small_cup_center[2]
            tracking_small_cup_target = limited_xy_step(
                tracking_small_cup_target,
                desired_small_cup_center,
                MAX_TRACK_XY_STEP,
            )
            tracking_big_cup_position = big_cup_position_for_small_cup_center(
                tracking_small_cup_target,
                flipped_orientation,
            )
            set_cartesian_goal(tracking_big_cup_position, flipped_orientation)

            current_big_cup_position = load_json_key(
                redis_keys.cartesian_task_current_position,
                redis_keys.cartesian_task_current_position,
            )
            current_big_cup_orientation = load_json_key(
                redis_keys.cartesian_task_current_orientation,
                redis_keys.cartesian_task_current_orientation,
            )
            actual_small_cup_center = small_cup_center(
                current_big_cup_position,
                current_big_cup_orientation,
            )

            falling = linear_velocity[2] < -0.05
            ball_near_small_cup = (
                abs(ball_position[2] - (actual_small_cup_center[2] + BALL_CENTER_ABOVE_CUP))
                < ACTUAL_CATCH_HEIGHT_WINDOW
            )
            timed_out = now - state_start_time >= TRACK_TIMEOUT
            if (falling and ball_near_small_cup) or timed_out:
                state_start_time = now
                state = State.HOLDING_CATCH
                print_tracking_snapshot("Catch window")
                print("Holding the small cup under the landing path.")

        elif state == State.HOLDING_CATCH:
            hold_big_cup_position = big_cup_position_for_small_cup_center(
                tracking_small_cup_target,
                flipped_orientation,
            )
            set_cartesian_goal(hold_big_cup_position, flipped_orientation)
            if now - state_start_time >= HOLD_DURATION:
                current_big_cup_position = load_json_key(
                    redis_keys.cartesian_task_current_position,
                    redis_keys.cartesian_task_current_position,
                )
                current_big_cup_orientation = load_json_key(
                    redis_keys.cartesian_task_current_orientation,
                    redis_keys.cartesian_task_current_orientation,
                )
                current_small_cup_center = small_cup_center(
                    current_big_cup_position,
                    current_big_cup_orientation,
                )
                ball_pose = load_json_key(redis_keys.ball_pose, redis_keys.ball_pose)
                ball_position = ball_pose[0:3, 3]
                catch_error = np.linalg.norm(
                    ball_position - (current_small_cup_center + np.array([0.0, 0.0, BALL_CENTER_ABOVE_CUP]))
                )
                print(f"Catch hold complete. Ball-to-small-cup center error: {catch_error:.4f} m")
                break

except KeyboardInterrupt:
    print("Keyboard interrupt")
except Exception as exc:
    print(f"Controller error: {exc}")
    raise
