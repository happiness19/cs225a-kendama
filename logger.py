import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import flexivrdk
import redis


DEFAULT_ROBOT_NAME = "Titania"
DEFAULT_ROBOT_SN = "Rizon4R_062043"
DEFAULT_RATE_HZ = 150.0
DEFAULT_DURATION_SEC = 30.0


def vec(values) -> str:
    return " ".join(f"{v:.10g}" for v in values)


def get_attr(obj, name: str):
    return getattr(obj, name, [])


def decode_redis_value(raw: bytes | None) -> str:
    return "" if raw is None else raw.decode("utf-8")


def wait_until_operational(robot, timeout: float) -> None:
    start = time.perf_counter()
    while not robot.operational():
        if time.perf_counter() - start > timeout:
            raise TimeoutError("Timed out waiting for robot to become operational.")
        time.sleep(0.1)


def start_free_drive(robot, primitive_name: str, timeout: float) -> None:
    if robot.fault():
        if not robot.ClearFault():
            raise RuntimeError("Robot fault could not be cleared.")

    robot.Enable()
    wait_until_operational(robot, timeout)

    robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
    robot.ExecutePrimitive(primitive_name, dict())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start Flexiv Cartesian free drive and log robot kinematics/dynamics to CSV."
    )
    parser.add_argument("--robot-name", default=DEFAULT_ROBOT_NAME, help="Human/OpenSai robot name.")
    parser.add_argument("--robot-sn", default=DEFAULT_ROBOT_SN, help="Flexiv RDK robot serial number.")
    parser.add_argument("--redis-host", default="localhost", help="Redis host.")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port.")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="Log rate in Hz.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC, help="Log duration in seconds.")
    parser.add_argument(
        "--primitive",
        default="FloatingCartesian",
        help="Free-drive primitive to execute before logging.",
    )
    parser.add_argument(
        "--skip-free-drive",
        action="store_true",
        help="Only log states; do not switch modes or execute a floating primitive.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the robot to become operational.",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.duration < 0:
        raise ValueError("--duration must be non-negative")

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    out = log_dir / f"{datetime.now():%Y_%m_%d_%H_%M_%S}.csv"
    latest = log_dir / "latest.csv"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out.name)

    fields = [
        "t_wall",
        "robot_name",
        "robot_sn",
        "primitive",
        "robot_sec",
        "robot_nsec",
        "q",
        "dq",
        "theta",
        "dtheta",
        "tau",
        "tau_des",
        "tau_dot",
        "tau_ext",
        "tau_interact",
        "tcp_pose",
        "tcp_vel",
        "flange_pose",
        "flange_transform",
        "cartesian_task_current_orientation",
        "kendama_bot_pos",
        "kendama_bot_ori",
        "ext_wrench_in_tcp",
        "ext_wrench_in_world",
        "ext_wrench_in_world_raw",
        "ft_sensor_raw",
    ]

    robot = flexivrdk.Robot(args.robot_sn)
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port)
    flange_transform_key = f"opensai::sensors::{args.robot_name}::flange_transform"
    cartesian_task_current_orientation_key = (
        f"opensai::controllers::{args.robot_name}"
        "::cartesian_controller::cartesian_task::current_orientation"
    )
    kendama_bot_pos_key = "KendamaBot::pos"
    kendama_bot_ori_key = "KendamaBot::ori"

    try:
        if not args.skip_free_drive:
            print(f"Starting free drive with primitive {args.primitive!r}.")
            start_free_drive(robot, args.primitive, args.startup_timeout)
            print("Free drive started; recording.")
        else:
            print("Skipping free-drive startup; recording current robot state.")

        period = 1.0 / args.rate
        t0 = time.perf_counter()
        next_t = t0

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            while time.perf_counter() - t0 < args.duration:
                state = robot.states()
                timestamp = get_attr(state, "timestamp")

                writer.writerow(
                    {
                        "t_wall": f"{time.time():.9f}",
                        "robot_name": args.robot_name,
                        "robot_sn": args.robot_sn,
                        "primitive": "" if args.skip_free_drive else args.primitive,
                        "robot_sec": timestamp[0] if len(timestamp) > 0 else "",
                        "robot_nsec": timestamp[1] if len(timestamp) > 1 else "",
                        "q": vec(get_attr(state, "q")),
                        "dq": vec(get_attr(state, "dq")),
                        "theta": vec(get_attr(state, "theta")),
                        "dtheta": vec(get_attr(state, "dtheta")),
                        "tau": vec(get_attr(state, "tau")),
                        "tau_des": vec(get_attr(state, "tau_des")),
                        "tau_dot": vec(get_attr(state, "tau_dot")),
                        "tau_ext": vec(get_attr(state, "tau_ext")),
                        "tau_interact": vec(get_attr(state, "tau_interact")),
                        "tcp_pose": vec(get_attr(state, "tcp_pose")),
                        "tcp_vel": vec(get_attr(state, "tcp_vel")),
                        "flange_pose": vec(get_attr(state, "flange_pose")),
                        "flange_transform": decode_redis_value(redis_client.get(flange_transform_key)),
                        "cartesian_task_current_orientation": decode_redis_value(
                            redis_client.get(cartesian_task_current_orientation_key)
                        ),
                        "kendama_bot_pos": decode_redis_value(redis_client.get(kendama_bot_pos_key)),
                        "kendama_bot_ori": decode_redis_value(redis_client.get(kendama_bot_ori_key)),
                        "ext_wrench_in_tcp": vec(get_attr(state, "ext_wrench_in_tcp")),
                        "ext_wrench_in_world": vec(get_attr(state, "ext_wrench_in_world")),
                        "ext_wrench_in_world_raw": vec(get_attr(state, "ext_wrench_in_world_raw")),
                        "ft_sensor_raw": vec(get_attr(state, "ft_sensor_raw")),
                    }
                )

                next_t += period
                sleep_time = next_t - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.skip_free_drive:
            robot.Stop()

    print(f"Wrote log to {out}")


if __name__ == "__main__":
    main()
