"""Simple redis poller that prints the Kendama ball pose."""

import argparse
import json
import time

import redis


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the KendamaBall position/orientation from Redis.")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--key-pos", default="KendamaBall::pos", help="Redis key storing the ball position")
    parser.add_argument("--key-ori", default="KendamaBall::ori", help="Redis key storing the ball orientation")
    parser.add_argument("--rate", type=float, default=10.0, help="Poll rate in Hz")
    args = parser.parse_args()

    client = redis.Redis(host=args.host, port=args.port)
    dt = 1.0 / args.rate if args.rate > 0 else 0.1

    print("Polling Redis for KendamaBall pose. Ctrl+C to stop.")
    try:
        while True:
            raw_pos = client.get(args.key_pos)
            raw_ori = client.get(args.key_ori)

            pos = None
            ori = None
            if raw_pos:
                try:
                    pos = json.loads(raw_pos.decode("utf-8"))
                except Exception:
                    pos = raw_pos.decode("utf-8", errors="ignore")
            if raw_ori:
                try:
                    ori = json.loads(raw_ori.decode("utf-8"))
                except Exception:
                    ori = raw_ori.decode("utf-8", errors="ignore")

            timestamp = time.time()
            print(f"[{timestamp:.3f}] position={pos} orientation={ori}")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("Stopping pose monitor.")


if __name__ == "__main__":
    main()
