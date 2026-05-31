import argparse
import json
import math
import signal
import time

import numpy as np
import redis

DEFAULT_HOME_JOINTS = np.array([
    math.radians(49.22),
    math.radians(-98.22),
    math.radians(-97.69),
    math.radians(83.55),
    math.radians(98.78),
    math.radians(-6.29),
    math.radians(-33.43)
])

ROBOT_NAME = "Titania"

MAX_Z = 0.319234  # Z ceiling — robot will not move above this height
MIN_Y = -0.35
MAX_Y = 0.35
MIN_X = 0.4
MAX_X = 0.8
CUP_Z = 0.2  # ⚠️ SET THIS to the actual Z height of your cup before running


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


class ThrowDetector:
    """
    Detects when a ball transitions from held to thrown using
    velocity and acceleration thresholds.
    """

    VELOCITY_THRESHOLD = 0.5   # m/s  — min speed to consider a throw
    ACCEL_THRESHOLD    = 3.0   # m/s² — min acceleration spike at release
    MIN_FLIGHT_SAMPLES = 5     # samples before confirming throw (debounce)

    def __init__(self, dt: float):
        self.dt = dt
        self.history: list = []
        self.velocities: list = []
        self.state = "held"
        self.flight_count = 0

    def update(self, pos: np.ndarray) -> str:
        self.history.append(pos.copy())
        if len(self.history) > 20:
            self.history.pop(0)

        if len(self.history) < 3:
            return self.state

        p0, p1, p2 = self.history[-3], self.history[-2], self.history[-1]
        v1 = (p1 - p0) / self.dt
        v2 = (p2 - p1) / self.dt
        accel = (v2 - v1) / self.dt

        speed     = np.linalg.norm(v2)
        accel_mag = np.linalg.norm(accel)

        self.velocities.append(v2.copy())
        if len(self.velocities) > 20:
            self.velocities.pop(0)

        if self.state == "held":
            if speed > self.VELOCITY_THRESHOLD and accel_mag > self.ACCEL_THRESHOLD:
                self.state = "throwing"
                self.flight_count = 1
                print(f"[ThrowDetector] Possible throw — speed={speed:.2f} accel={accel_mag:.2f}")

        elif self.state == "throwing":
            if speed > self.VELOCITY_THRESHOLD:
                self.flight_count += 1
                if self.flight_count >= self.MIN_FLIGHT_SAMPLES:
                    self.state = "in_flight"
                    print(f"[ThrowDetector] Throw CONFIRMED after {self.flight_count} samples.")
            else:
                self.state = "held"
                self.flight_count = 0
                print("[ThrowDetector] False alarm — back to held.")

        elif self.state == "in_flight":
            if speed < 0.1:
                self.state = "held"
                self.flight_count = 0
                print("[ThrowDetector] Ball stopped — resetting to held.")

        return self.state

    def reset(self):
        self.state = "held"
        self.flight_count = 0
        self.history.clear()
        self.velocities.clear()


def predict_landing(ball_positions: list, target_z: float, dt: float):
    """
    Predict where the ball will land at a fixed Z plane using projectile motion.

    Args:
        ball_positions: Recent list of (x, y, z) np.ndarray positions, newest last.
        target_z:       The Z height at which to predict landing (e.g. cup Z).
        dt:             Time between samples in seconds (must match actual loop dt).

    Returns:
        np.ndarray [x, y, target_z] predicted landing position, or None if
        not enough data or ball is not heading toward target_z.
    """
    if len(ball_positions) < 3:
        return None

    G = 9.81

    samples = np.array(ball_positions[-6:])
    n = len(samples)
    t = np.arange(n) * dt
    ones = np.ones(n)

    A_lin = np.column_stack([ones, t])
    vx = np.linalg.lstsq(A_lin, samples[:, 0], rcond=None)[0][1]
    vy = np.linalg.lstsq(A_lin, samples[:, 1], rcond=None)[0][1]

    A_quad = np.column_stack([ones, t, -0.5 * t**2])
    z_coeffs = np.linalg.lstsq(A_quad, samples[:, 2], rcond=None)[0]
    z0, vz, _ = z_coeffs

    x_cur = samples[-1, 0]
    y_cur = samples[-1, 1]
    z_cur = samples[-1, 2]

    a_coef =  0.5 * G
    b_coef = -vz
    c_coef =  target_z - z_cur

    discriminant = b_coef**2 - 4 * a_coef * c_coef
    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b_coef - sqrt_disc) / (2 * a_coef)
    t2 = (-b_coef + sqrt_disc) / (2 * a_coef)

    candidates = [s for s in [t1, t2] if s > 0]
    if not candidates:
        return None

    s = min(candidates)

    x_land = x_cur + vx * s
    y_land = y_cur + vy * s

    return np.array([x_land, y_land, target_z])


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

    detector = ThrowDetector(dt=dt)
    ball_history = []
    landing_target = None
    throw_state = "held"

    print(f"EE position  : {ee_pos}")
    print(f"Ball position: {ball_pos}")
    print(f"Offset       : {offset}")
    print(f"Z ceiling    : {MAX_Z}")
    print(f"Cup Z        : {CUP_Z}")
    print("Tracking ball — press Ctrl+C to stop.")

    while True:
        # Refresh current EE position each cycle for accurate Z floor
        ee_pos = get_vec(client, keys["pos_cur"])
        ball_pos = get_vec(client, ball_pos_key)

        if ee_pos is not None and ball_pos is not None:
            ball_history.append(ball_pos.copy())
            if len(ball_history) > 20:
                ball_history.pop(0)

            throw_state = detector.update(ball_pos)

            if throw_state == "held":
                landing_target = None
                desired = ball_pos + offset

            elif throw_state == "throwing":
                if landing_target is None:
                    landing_target = predict_landing(ball_history, target_z=CUP_Z, dt=dt)
                    if landing_target is not None:
                        print(f"[Prediction] Ball will land at: {landing_target}")
                # Copy to avoid mutating landing_target via in-place clamp below
                desired = landing_target.copy() if landing_target is not None else ball_pos + offset

            elif throw_state == "in_flight":
                refined = predict_landing(ball_history, target_z=CUP_Z, dt=dt)
                if refined is not None:
                    landing_target = refined
                desired = landing_target.copy() if landing_target is not None else ball_pos + offset

            else:
                desired = ball_pos + offset  # Fallback — should never hit this

            # Apply workspace clamps
            desired[0] = np.clip(desired[0], MIN_X, MAX_X)
            desired[1] = np.clip(desired[1], MIN_Y, MAX_Y)
            desired[2] = np.clip(desired[2], ee_pos[2], MAX_Z)

            delta = desired - target
            dist = np.linalg.norm(delta)
            if dist > max_step:
                delta *= max_step / dist

            target = target + delta

            set_vec(client, keys["pos_goal"], target)
            set_vec(client, keys["ori_goal"], fixed_ori)

        print(f"[{throw_state:10s}] Waiting for {dt}s, Robot Position: {ee_pos}, Target position: {target}, Ball position: {ball_pos}")
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