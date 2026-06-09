"""
swing_and_catch.py — Combined swing + ball-tracking loop for the kendama demo (v1).

================================================================================
WHAT THIS SCRIPT DOES (high level)
================================================================================
This is a phase state machine that loops the following sequence:

    CALIBRATE  ->  SWING  ->  HANDOFF  ->  TRACK  ->  (return home, repeat)

1. CALIBRATE  Park at home in JOINT mode, then ask you to seat the ball in the
              cup. On Enter, capture  offset = ee_pos - ball_pos . This offset
              encodes "ball sitting in the cup." You then let the ball hang.
2. SWING      Replay the logged joint trajectory (logs/close_enough.csv) in
              JOINT mode. While replaying, watch the live ball Z each cycle.
              When ball_z crosses Z_THRESHOLD (the ball swinging up at the end
              of the arc), abort the replay and hand off.
3. HANDOFF    Switch JOINT -> CARTESIAN. Seed the Cartesian goal from the LIVE
              EE pose so the arm does not snap, and capture the swing's ending
              orientation as the fixed tracking orientation.
4. TRACK      Shadow the ball:  desired = ball_pos + offset , clamped to the
              workspace box and rate-limited per cycle, holding fixed_ori. Runs
              for TRACK_TIMEOUT seconds.
5. LOOP       Return home (JOINT) and repeat.

================================================================================
RELATION TO THE EXISTING SCRIPTS — WHAT'S REUSED, WHAT'S NEW
================================================================================
REUSED (imported directly from replay_logged_motion.py — tested, proven code):
  - load_joint_trajectory   : CSV parsing (handles q required, dq/ddq optional)
  - smooth_joint_samples     : optional moving-average smoothing
  - truncate_joint_samples   : the --stop-time cutoff behavior
  - joint_feedforward        : velocity / velocity-acceleration feed-forward
  - set_joint_command        : writes goal pos/vel/accel to Redis
  - set_active_controller    : controller switching with confirmation
  - make_keys (as rlm_make_keys) / RedisKeys : the Redis key layout
  We deliberately reuse all the *fiddly* trajectory-loading and feed-forward
  code rather than rewriting it. See SWING phase.

REWRITTEN HERE (small, so you can edit the swing/track loops directly):
  - The swing replay LOOP itself. replay_logged_motion.replay_joint_samples is
    a self-contained blocking loop with NO per-cycle ball check. We need to
    watch ball Z *during* the swing to trigger the handoff, so the ~15-line
    replay loop lives here (see run_swing) where you fully control the abort.
  - The TRACK loop is copied from move_with_ball.py (the simple shadowing
    tracker, NOT the predictive one). Same offset math, same clamps.

SAME AS move_with_ball.py:
  - offset = ee_pos - ball_pos, desired = ball_pos + offset
  - workspace clamps (MIN/MAX X/Y/Z), max_step rate limiting, fixed orientation.
DIFFERENT FROM move_with_ball.py:
  - The offset is captured during CALIBRATE (ball in cup), BEFORE the swing,
    instead of at the start of tracking.
  - fixed_ori is captured at HANDOFF (whatever the swing ended in), so you can
    swap in a new swing CSV with a different ending orientation freely.

================================================================================
HOW TO MODIFY THIS SCRIPT (quick index — all knobs are near the top)
================================================================================
* Run a DIFFERENT logged motion : pass --swing-csv path/to/file.csv  (or change
                                   DEFAULT_SWING_CSV). It must be a JOINT-mode
                                   log with a 'q' column, same as close_enough.
* Tune the swing->track trigger  : Z_THRESHOLD (and Z_AXIS if the world is
                                   rotated — see "WORLD AXIS" below).
* Change the WORLD AXIS          : Z_AXIS picks which coordinate the threshold
                                   checks. If the robot/world frame is rotated
                                   so "up" is not index 2, change Z_AXIS. The
                                   workspace clamps use explicit indices too;
                                   adjust those if your up-axis changes.
* Where the OFFSET is defined    : run_calibration(), the line
                                   offset = ee_pos - ball_pos
* Where the TRACKING motion is   : run_track(). Edit the shadowing math /
                                   clamps / max_step there.
* Where to add a PREDICTIVE model: see the big "INSERT PREDICTION HERE" comment
                                   inside run_track(). You would replace
                                   `desired = ball_pos + offset` with a call to
                                   move_with_ball_predictive.predict_landing()
                                   plus its ThrowDetector. Left out of v1 on
                                   purpose to keep this simple.
* Loop / retry behavior          : TRACK_TIMEOUT controls how long each catch
                                   attempt tracks before returning home and
                                   re-swinging. v1 has NO catch-success check.
================================================================================
"""

import argparse
import math
import time

import numpy as np
import redis

# We import the TESTED machinery from replay_logged_motion.py rather than
# re-implementing CSV parsing and feed-forward. These are used unchanged.
from replay_logged_motion import (
    RedisKeys,
    make_keys as rlm_make_keys,
    load_joint_trajectory,
    smooth_joint_samples,
    truncate_joint_samples,
    joint_feedforward,
    set_joint_command,
    set_joint_goal,
    set_active_controller as rlm_set_active_controller,
    read_current_joints,
)

# ============================================================================
# TUNABLES — edit these freely during testing.
# ============================================================================
ROBOT_NAME = "Titania"

# Home pose. Matches move_with_ball.py's DEFAULT_HOME_JOINTS (kendama pointing
# perpendicular to the ground), NOT main.py's DEFAULT_JOINTS. Change if your
# swing's start pose differs.
DEFAULT_HOME_JOINTS = np.array([
    math.radians(49.22),
    math.radians(-98.22),
    math.radians(-97.69),
    math.radians(83.55),
    math.radians(98.78),
    math.radians(-6.29),
    math.radians(-33.43),
])

# --- Swing -> Track trigger -------------------------------------------------
# WORLD AXIS: which coordinate of the ball position is "up". 2 = Z (default).
# If the robot/world frame is rotated, change this so the threshold checks the
# real vertical axis.
Z_AXIS = 2
# Ball crosses this height (in world coords, on Z_AXIS) at the top of the swing
# -> hand off to tracking. Tune by hand; start near the EE's height.
Z_THRESHOLD = 0.30

# --- Tracking (copied from move_with_ball.py) -------------------------------
MAX_Z = 0.319234   # Z ceiling
MIN_Y = -0.35
MAX_Y = 0.35
MIN_X = 0.4
MAX_X = 0.8

# How long to shadow the ball per catch attempt before re-homing and looping.
TRACK_TIMEOUT = 5.0

# Swing replay defaults (mirrors run_close_enough.sh).
DEFAULT_SWING_CSV = "logs/close_enough.csv"
DEFAULT_STOP_TIME = 5.327          # --stop-time cutoff used in the demo
DEFAULT_FEEDFORWARD = "velocity-acceleration"
DEFAULT_MAX_ACCEL = 15.0           # rad/s^2 clip for ff acceleration
DEFAULT_SMOOTH_WINDOW = 0.0        # seconds; 0 = no smoothing


# ============================================================================
# Redis helpers. These mirror move_with_ball.py's get_vec/set_vec (which use
# decode_responses=False). replay_logged_motion.py uses decode_responses=True
# and json-based helpers; to keep ONE client and avoid mixing the two byte/str
# conventions, we open the client WITHOUT decode_responses (bytes), matching
# move_with_ball.py, and the reused replay helpers (set_joint_command etc.)
# only WRITE json strings, which works fine on a bytes client.
# ============================================================================
import json


def get_vec(client, key):
    raw = client.get(key)
    if raw is None:
        return None
    return np.array(json.loads(raw.decode("utf-8")))


def set_vec(client, key, vec):
    client.set(key, json.dumps(np.asarray(vec).tolist()))


def go_home(client, keys: RedisKeys, home_joints, max_step_deg=0.5,
            threshold=0.2, settle_count=20):
    """Drive slowly to home in JOINT mode. Same interpolation idea as
    move_with_ball.py.go_home, but uses the reused RedisKeys/controller name."""
    print("[Home] Switching to joint controller...")
    rlm_set_active_controller(client, keys, keys.joint_controller_name, timeout=5.0)

    q = None
    while q is None:
        q = read_current_joints(client, keys)
        time.sleep(0.01)

    target = q.copy()
    max_step = math.radians(max_step_deg)
    print("[Home] Moving slowly to home...")

    stable = 0
    while True:
        delta = home_joints - target
        dist = np.linalg.norm(delta)
        if dist > max_step:
            delta *= max_step / dist
        target = target + delta
        set_joint_goal(client, keys, target)

        q = read_current_joints(client, keys)
        if q is not None:
            err = np.linalg.norm(q - home_joints)
            if err < threshold:
                stable += 1
                if stable >= settle_count:
                    print(f"[Home] Reached (err={err:.4f} rad).")
                    return
            else:
                stable = 0
        time.sleep(0.01)


# ============================================================================
# PHASE 1 — CALIBRATION.  This is WHERE THE OFFSET IS DEFINED.
# ============================================================================
def run_calibration(client, keys, ball_pos_key):
    """Home, then capture offset = ee_pos - ball_pos with the ball in the cup.

    We switch to Cartesian briefly ONLY to read a valid EE Cartesian position
    (pos_cur). After capture we go back to joint mode for the swing.
    """
    go_home(client, keys, DEFAULT_HOME_JOINTS)
    time.sleep(0.5)

    input("[Calibrate] Seat the ball IN THE CUP, then press Enter...")

    # Need a Cartesian read of the EE; switch controller so pos_cur is fresh.
    rlm_set_active_controller(client, keys, keys.cartesian_controller_name, timeout=5.0)
    time.sleep(0.2)

    ee_pos = get_vec(client, keys.cartesian_current_position)
    ball_pos = get_vec(client, ball_pos_key)
    if ee_pos is None or ball_pos is None:
        raise RuntimeError("Could not read EE or ball position during calibration.")

    offset = ee_pos - ball_pos            # <-- OFFSET DEFINED HERE
    print(f"[Calibrate] EE   : {np.round(ee_pos, 4)}")
    print(f"[Calibrate] Ball : {np.round(ball_pos, 4)}")
    print(f"[Calibrate] Offset captured: {np.round(offset, 4)}")
    input("[Calibrate] Now let the ball HANG. Press Enter to begin the swing...")
    return offset


# ============================================================================
# PHASE 2 — SWING.  Reuses replay_logged_motion loaders; the LOOP is local so
# we can watch ball Z and abort. This is the only "new" replay code.
# ============================================================================
def load_swing(csv_path, stop_time, smooth_window):
    """Load + shape the swing using the TESTED replay_logged_motion functions."""
    samples, _first_row = load_joint_trajectory(csv_path, time_source="robot")
    samples = smooth_joint_samples(samples, smooth_window)
    samples = truncate_joint_samples(samples, stop_time)
    return samples


def run_swing(client, keys, samples, ball_pos_key, dt, speed,
              feedforward_mode, max_accel):
    """Stream the joint trajectory while watching ball Z. Returns True if the
    Z threshold was crossed (hand off to track), False if the swing finished
    without crossing (fallback)."""
    print("[Swing] Switching to joint controller and replaying...")
    rlm_set_active_controller(client, keys, keys.joint_controller_name, timeout=5.0)

    start = time.perf_counter()
    for sample in samples:
        # --- timing (same scheme as replay_joint_samples) ---
        target_time = start + sample.t / speed
        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        # --- command with reused feed-forward ---
        velocity, acceleration = joint_feedforward(sample, feedforward_mode, max_accel)
        if velocity is not None:
            velocity = velocity * speed
        if acceleration is not None:
            acceleration = acceleration * speed * speed
        set_joint_command(client, keys, sample.q, velocity=velocity,
                          acceleration=acceleration)

        # --- THE ABORT CHECK: watch the live ball's up-axis position ---
        ball_pos = get_vec(client, ball_pos_key)
        if ball_pos is not None and ball_pos[Z_AXIS] > Z_THRESHOLD:
            print(f"[Swing] Ball crossed Z_THRESHOLD "
                  f"({ball_pos[Z_AXIS]:.3f} > {Z_THRESHOLD}) — handing off.")
            return True

    print("[Swing] Replay finished without crossing threshold (fallback).")
    return False


# ============================================================================
# PHASE 3 — HANDOFF.  Switch to Cartesian and seed the goal from the LIVE EE
# pose so nothing snaps. Captures the swing's ending orientation.
# ============================================================================
def run_handoff(client, keys):
    print("[Handoff] Switching to Cartesian controller...")
    rlm_set_active_controller(client, keys, keys.cartesian_controller_name, timeout=5.0)
    time.sleep(0.2)

    ee_pos = get_vec(client, keys.cartesian_current_position)
    ee_ori = get_vec(client, keys.cartesian_current_orientation)
    if ee_pos is None or ee_ori is None:
        raise RuntimeError("Could not read EE pose at handoff.")

    # Seed goal = current pose => no jump. Hold this orientation through track.
    set_vec(client, keys.cartesian_goal_position, ee_pos)
    set_vec(client, keys.cartesian_goal_orientation, ee_ori.reshape(3, 3))
    print(f"[Handoff] Seeded Cartesian goal at {np.round(ee_pos, 4)}")
    return ee_pos.copy(), ee_ori.copy()


# ============================================================================
# PHASE 4 — TRACK.  Copied from move_with_ball.py's shadowing loop.
# ============================================================================
def run_track(client, keys, ball_pos_key, offset, fixed_ori, start_target,
              max_step, dt, timeout):
    print("[Track] Shadowing ball...")
    target = start_target.copy()
    end_time = time.perf_counter() + timeout

    while time.perf_counter() < end_time:
        ball_pos = get_vec(client, ball_pos_key)
        if ball_pos is not None:

            # ----------------------------------------------------------------
            # INSERT PREDICTION HERE (future work):
            #   Replace the line below with predictive interception, e.g.
            #   feed ball_pos into a ThrowDetector + predict_landing() from
            #   move_with_ball_predictive.py and set `desired` to the predicted
            #   catch point instead of the live offset shadow. Left out of v1.
            # ----------------------------------------------------------------
            desired = ball_pos + offset

            # Workspace clamps — same as move_with_ball.py.
            desired[0] = min(max(desired[0], MIN_X), MAX_X)
            desired[1] = min(max(desired[1], MIN_Y), MAX_Y)
            desired[2] = min(desired[2], MAX_Z)
            desired[2] = max(desired[2], start_target[2])  # don't dip below start

            delta = desired - target
            dist = np.linalg.norm(delta)
            if dist > max_step:
                delta *= max_step / dist
            target = target + delta

            set_vec(client, keys.cartesian_goal_position, target)
            set_vec(client, keys.cartesian_goal_orientation, fixed_ori.reshape(3, 3))

        time.sleep(dt)

    print("[Track] Timeout — ending this catch attempt.")


# ============================================================================
# MAIN — the loop.
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Swing + catch loop (v1).")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--ball-pos", default="KendamaBall::pos")
    parser.add_argument("--max-step", type=float, default=0.003,
                        help="Max Cartesian motion per cycle (m).")
    parser.add_argument("--swing-csv", default=DEFAULT_SWING_CSV)
    parser.add_argument("--stop-time", type=float, default=DEFAULT_STOP_TIME)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--smooth-window", type=float, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--feedforward", default=DEFAULT_FEEDFORWARD,
                        choices=("none", "velocity", "velocity-acceleration"))
    parser.add_argument("--max-accel", type=float, default=DEFAULT_MAX_ACCEL)
    args = parser.parse_args()

    dt = 1.0 / args.rate
    client = redis.Redis(host=args.host, port=args.port)  # bytes (no decode)
    keys = rlm_make_keys(ROBOT_NAME, "joint_controller", "cartesian_controller")

    # Load swing ONCE up front using the tested loaders.
    samples = load_swing(args.swing_csv, args.stop_time, args.smooth_window)
    print(f"[Init] Loaded {len(samples)} swing samples from {args.swing_csv}")

    # Calibrate offset once (ball in cup), then loop swing->catch.
    offset = run_calibration(client, keys, args.ball_pos)

    try:
        while True:
            crossed = run_swing(client, keys, samples, args.ball_pos, dt,
                                args.speed, args.feedforward, args.max_accel)
            if not crossed:
                print("[Loop] Threshold not crossed; handing off anyway.")
            ee_pos, fixed_ori = run_handoff(client, keys)
            run_track(client, keys, args.ball_pos, offset, fixed_ori, ee_pos,
                      args.max_step, dt, TRACK_TIMEOUT)

            print("[Loop] Returning home for next attempt.")
            go_home(client, keys, DEFAULT_HOME_JOINTS)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Exit] Stopping; parking home.")
        go_home(client, keys, DEFAULT_HOME_JOINTS)


if __name__ == "__main__":
    main()
