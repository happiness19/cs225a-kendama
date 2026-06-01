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
MIN_Z = 0.05      # Z floor  — robot will not move below this height
MIN_Y = -0.35
MAX_Y = 0.35
MIN_X = 0.4
MAX_X = 0.8

# How often (in cycles) to print status. At 100 Hz, 10 → print every 100 ms.
PRINT_EVERY_N_CYCLES = 10

# Max in-flight samples before auto-resetting (2 s @ 100 Hz = 200).
MAX_FLIGHT_FRAMES = 200

# Consecutive "held" frames required before recomputing the EE→ball offset.
OFFSET_REGEN_FRAMES = 30


def make_keys(robot_name):
    base = f"opensai::controllers::{robot_name}"
    sensors = f"opensai::sensors::{robot_name}"
    return {
        "active":     f"{base}::active_controller_name",
        "joint_goal": f"{base}::joint_controller::joint_task::goal_position",
        "joint_cur":  f"{sensors}::joint_positions",
        "pos_goal":   f"{base}::cartesian_controller::cartesian_task::goal_position",
        "ori_goal":   f"{base}::cartesian_controller::cartesian_task::goal_orientation",
        "pos_cur":    f"{base}::cartesian_controller::cartesian_task::current_position",
        "ori_cur":    f"{base}::cartesian_controller::cartesian_task::current_orientation",
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
            print(
                f"Robot position: {np.degrees(q).round(2)}, err={err:.4f} rad "
                f"Target position: {np.degrees(target).round(2)}"
            )
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
    Detects when a ball transitions from held → throwing → in_flight using
    velocity and acceleration thresholds.

    States
    ------
    held       : ball is stationary or moving slowly with the robot
    throwing   : velocity/accel spike detected, waiting for debounce
    in_flight  : throw confirmed; ball is ballistic
    """

    VELOCITY_THRESHOLD = 0.5   # m/s  — min speed to consider a throw
    ACCEL_THRESHOLD    = 3.0   # m/s² — min acceleration spike at release
    MIN_FLIGHT_SAMPLES = 5     # consecutive fast samples needed to confirm

    def __init__(self, dt: float):
        self.dt = dt
        self.history: list = []
        self.velocities: list = []
        self.state = "held"
        self.flight_count = 0
        self.flight_frames = 0  # total frames in in_flight (for timeout)

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
                    self.flight_frames = 0
                    print(f"[ThrowDetector] Throw CONFIRMED after {self.flight_count} samples.")
            else:
                self.state = "held"
                self.flight_count = 0
                print("[ThrowDetector] False alarm — back to held.")

        elif self.state == "in_flight":
            self.flight_frames += 1
            # FIX: timeout — prevents staying in_flight if ball disappears
            if self.flight_frames > MAX_FLIGHT_FRAMES:
                self.reset()
                print("[ThrowDetector] Flight timeout — resetting to held.")
            elif speed < 0.1:
                self.reset()
                print("[ThrowDetector] Ball stopped — resetting to held.")

        return self.state

    def reset(self):
        self.state = "held"
        self.flight_count = 0
        self.flight_frames = 0
        self.history.clear()
        self.velocities.clear()


def predict_landing(ball_positions: list, target_z: float, dt: float,
                    elapsed_cycles: int = 0):
    """
    Predict where the ball will cross a fixed Z catch plane while descending,
    using least-squares projectile fitting over the recent position window.

    Parameters
    ----------
    ball_positions : list of np.ndarray
        Recent ball positions (newest last).  Only the last 6 are used.
    target_z : float
        The Z height at which to predict the intercept (cup Z).
    dt : float
        Time between samples (seconds).
    elapsed_cycles : int
        Control cycles that have elapsed since the newest sample was recorded.
        Shifts the time axis so that t=0 corresponds to *now*, correcting for
        loop latency.

    Returns
    -------
    np.ndarray [x, y, target_z] or None. Returns None until the fitted
    trajectory has a future descending crossing of the catch plane.
    """
    if len(ball_positions) < 3:
        return None

    G = 9.81
    samples = np.array(ball_positions[-6:])
    n = len(samples)

    # Build time axis: t[-1] = -elapsed_cycles*dt  (samples live in the past;
    # t=0 is the current control cycle).
    t_raw = np.arange(n) * dt                     # 0 … (n-1)*dt
    t     = t_raw - t_raw[-1] - elapsed_cycles * dt  # shift newest to -latency

    ones  = np.ones(n)

    # XY: linear (no horizontal acceleration)
    A_lin  = np.column_stack([ones, t])
    cx     = np.linalg.lstsq(A_lin, samples[:, 0], rcond=None)[0]
    cy     = np.linalg.lstsq(A_lin, samples[:, 1], rcond=None)[0]
    x_now, vx = cx[0], cx[1]
    y_now, vy = cy[0], cy[1]

    # Z: quadratic with fixed gravity
    A_quad   = np.column_stack([ones, t, -0.5 * t**2])
    z_coeffs = np.linalg.lstsq(A_quad, samples[:, 2], rcond=None)[0]
    z_now, vz, _ = z_coeffs   # z_now and vz evaluated at t=0 (now)

    # Solve: z_now + vz*s - 0.5*G*s^2 == target_z,  s > 0
    a_coef =  0.5 * G
    b_coef = -vz
    c_coef =  target_z - z_now

    discriminant = b_coef**2 - 4 * a_coef * c_coef
    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b_coef - sqrt_disc) / (2 * a_coef)
    t2 = (-b_coef + sqrt_disc) / (2 * a_coef)

    candidates = [s for s in [t1, t2] if s > 0]
    descending = [s for s in candidates if vz - G * s < 0]
    if not descending:
        return None

    s = min(descending)
    return np.array([x_now + vx * s, y_now + vy * s, target_z])


def track_ball(client, keys, ball_pos_key, max_step, dt, cup_z):
    """Switch to Cartesian control and track/catch the ball."""
    print("Switching to Cartesian controller...")
    set_active_controller(client, keys["active"], "cartesian_controller")
    time.sleep(0.2)

    ee_pos   = get_vec(client, keys["pos_cur"])
    ee_ori   = get_vec(client, keys["ori_cur"])
    ball_pos = get_vec(client, ball_pos_key)

    if ee_pos is None or ee_ori is None or ball_pos is None:
        raise RuntimeError("Could not read EE pose or ball position from Redis.")

    # Offset = vector from ball to EE when "held".
    # Recomputed automatically after each catch/reset.
    offset    = ee_pos - ball_pos
    fixed_ori = ee_ori.copy()
    target    = ee_pos.copy()

    detector         = ThrowDetector(dt=dt)
    ball_history: list = []
    landing_target   = None
    prev_state       = "held"
    held_stable_count  = 0   # frames consecutively in held (offset regen)
    flight_cycle_count = 0   # frames since throw confirmed (latency fix)
    cycle              = 0   # for rate-limiting prints
    throw_state        = "held"

    print(f"EE position  : {ee_pos}")
    print(f"Ball position: {ball_pos}")
    print(f"Offset       : {offset}")
    print(f"Z ceiling    : {MAX_Z}  Z floor: {MIN_Z}")
    print(f"Cup Z        : {cup_z}")
    print("Tracking ball — press Ctrl+C to stop.")

    while True:
        ee_pos   = get_vec(client, keys["pos_cur"])
        ball_pos = get_vec(client, ball_pos_key)

        if ee_pos is not None and ball_pos is not None:
            throw_state = detector.update(ball_pos)

            # FIX: clear history on held→throwing so the predictor only sees
            # in-flight samples, not stationary pre-throw data biasing vz.
            if prev_state == "held" and throw_state in ("throwing", "in_flight"):
                ball_history.clear()
                flight_cycle_count = 0
                print("[Track] Ball history cleared for fresh prediction.")

            ball_history.append(ball_pos.copy())
            if len(ball_history) > 40:   # ~400 ms @ 100 Hz
                ball_history.pop(0)

            if throw_state == "held":
                held_stable_count += 1
                # FIX: recompute offset after enough stable held frames so it
                # stays valid across multiple catch-and-reset cycles.
                if held_stable_count == OFFSET_REGEN_FRAMES:
                    offset = ee_pos - ball_pos
                    print(f"[Track] Offset recomputed: {offset}")
                landing_target = None
                desired = ball_pos + offset

            elif throw_state == "throwing":
                held_stable_count   = 0
                flight_cycle_count += 1
                if landing_target is None:
                    landing_target = predict_landing(
                        ball_history, target_z=cup_z, dt=dt,
                        elapsed_cycles=flight_cycle_count
                    )
                    if landing_target is not None:
                        print(f"[Prediction] Early estimate: {landing_target}")
                desired = landing_target.copy() if landing_target is not None else ball_pos + offset

            elif throw_state == "in_flight":
                held_stable_count   = 0
                flight_cycle_count += 1
                # Continuously refine prediction as new data arrives
                refined = predict_landing(
                    ball_history, target_z=cup_z, dt=dt,
                    elapsed_cycles=flight_cycle_count
                )
                if refined is not None:
                    landing_target = refined
                desired = landing_target.copy() if landing_target is not None else ball_pos + offset

            else:
                desired = ball_pos + offset  # safety fallback; should not be reached

            prev_state = throw_state

            # FIX: clamp Z between MIN_Z and MAX_Z (not ee_pos[2]) so the
            # robot can descend to the cup height during a catch.

            # desired[0] = np.clip(desired[0], MIN_X, MAX_X)
            desired[0] = min(desired[0], MAX_X)  # Clamp X to workspace
            desired[0] = max(desired[0], MIN_X)  # Clamp X to workspace
            desired[1] = min(desired[1], MAX_Y)  # Clamp Y to workspace
            desired[1] = max(desired[1], MIN_Y)  # Clamp Y to workspace
            desired[2] = min(desired[2], MAX_Z) # Clamp Z to ceiling
            desired[2] = max(desired[2], MIN_Z) # Clamp Z to floor

            # desired[1] = np.clip(desired[1], MIN_Y, MAX_Y)
            # desired[2] = np.clip(desired[2], MIN_Z, MAX_Z)

            delta = desired - target
            dist  = np.linalg.norm(delta)
            if dist > max_step:
                delta *= max_step / dist

            target = target + delta

            set_vec(client, keys["pos_goal"], target)
            set_vec(client, keys["ori_goal"], fixed_ori)

        # FIX: rate-limited print — avoids flooding stdout at 100 Hz
        cycle += 1
        
        if cycle % PRINT_EVERY_N_CYCLES == 0:
            print("Offset:", offset)
            print(
                f"[{throw_state:10s}] "
                f"EE={np.round(ee_pos, 3) if ee_pos is not None else 'N/A'}  "
                f"target={np.round(target, 3)}  "
                f"ball={np.round(ball_pos, 3) if ball_pos is not None else 'N/A'}"
            )

        time.sleep(dt)


def main():
    parser = argparse.ArgumentParser(description="Track KendamaBall using Cartesian control.")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int,   default=6379)
    parser.add_argument("--rate",     type=float, default=100.0)
    parser.add_argument("--ball-pos", default="KendamaBall::pos")
    parser.add_argument("--max-step", type=float, default=0.003,
                        help="Maximum Cartesian motion per cycle (meters).")
    # FIX: cup-z promoted to CLI argument — no more hardcoded constant to edit.
    parser.add_argument("--cup-z",    type=float, default=0.2,
                        help="Z height of the cup/catch target (meters). "
                             "⚠️  Measure and set this accurately before running.")
    args = parser.parse_args()

    client = redis.Redis(host=args.host, port=args.port)
    keys   = make_keys(ROBOT_NAME)

    def _shutdown(sig, frame):
        print("\nShutting down.")
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _shutdown)

    go_home(client, keys, DEFAULT_HOME_JOINTS)
    time.sleep(1.0)
    track_ball(client, keys, args.ball_pos, args.max_step,
               1.0 / args.rate, cup_z=args.cup_z)


if __name__ == "__main__":
    main()
