"""Publish synthetic Kendama ball poses into Redis for simulation."""

import argparse
import json
import math
import time

import redis


def build_pose(x: float, y: float, z: float, orientation: list[float]) -> dict:
    return {
        "position": [x, y, z],
        "orientation": orientation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a ball position stream in Redis.")
    parser.add_argument("--key", default="opensai::sensors::KendamaBall::object_pose", help="Redis key to publish the pose")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--rate", type=float, default=100.0, help="Update rate in Hz")
    parser.add_argument("--radius", type=float, default=0.15, help="Horizontal radius of the circular trajectory")
    parser.add_argument("--center-x", type=float, default=0.0, help="Center X offset")
    parser.add_argument("--center-y", type=float, default=0.3, help="Center Y offset")
    parser.add_argument("--center-z", type=float, default=0.25, help="Center Z height")
    parser.add_argument("--frequency", type=float, default=0.25, help="Orbit frequency in Hz")
    parser.add_argument("--orientation", default="0,0,0,1", help="Orientation quaternion (comma-separated list)")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run before exiting; infinite if omitted")
    parser.add_argument("--log", action="store_true", help="Print each pose to stdout")
    args = parser.parse_args()

    orientation = [float(x) for x in args.orientation.split(",")]
    if len(orientation) != 4:
        raise ValueError("Orientation must be four comma-separated floats (quaternion)")

    redis_client = redis.Redis(host=args.host, port=args.port)

    dt = 1.0 / args.rate if args.rate > 0 else 0.01
    start_time = time.perf_counter()

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if args.duration and elapsed > args.duration:
                break

            angle = 2 * math.pi * args.frequency * elapsed
            x = args.center_x + args.radius * math.sin(angle)
            y = args.center_y
            z = args.center_z

            pose = build_pose(x, y, z, orientation)
            redis_client.set(args.key, json.dumps(pose))

            if args.log:
                print(f"[{time.time():.3f}] {pose}")

            time.sleep(dt)
    except KeyboardInterrupt:
        print("Simulation interrupted.")


if __name__ == "__main__":
    main()
