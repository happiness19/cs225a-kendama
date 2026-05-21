"""
Simple Redis-based object pose logger.

Usage example:
    python3 sense_object_pose.py \
        --key opensai::sensors::KendamaBall::object_pose \
        --output object_pose_log.jsonl \
        --rate 100

Writes JSON Lines with fields: `t` (unix time), `position` (list) and `orientation` (list or null).
"""
import argparse
import json
import time
import redis
from typing import Tuple, Optional


def parse_pose_raw(raw_bytes) -> Tuple[Optional[list], Optional[list]]:
    if raw_bytes is None:
        return None, None
    try:
        s = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
        obj = json.loads(s)
    except Exception:
        return None, None

    # obj may be a dict, list, or simple array. Be permissive.
    if isinstance(obj, dict):
        # common shapes: {"position": [...], "orientation": [...]} or {"pose": {...}}
        if "position" in obj and "orientation" in obj:
            return obj["position"], obj["orientation"]
        if "pose" in obj and isinstance(obj["pose"], dict):
            p = obj["pose"].get("position") or obj["pose"].get("pos")
            o = obj["pose"].get("orientation") or obj["pose"].get("ori")
            return p, o
        # fallback: try keys that look like arrays
        for k in ("pos", "position", "p"):
            if k in obj:
                p = obj[k]
                o = obj.get("orientation") or obj.get("ori")
                return p, o
        return None, None

    if isinstance(obj, list):
        # [position, orientation]
        if len(obj) == 2 and (isinstance(obj[0], list) or isinstance(obj[0], tuple)):
            return list(obj[0]), list(obj[1])
        # flat array where first 3 are position
        if len(obj) >= 3 and all(isinstance(x, (int, float)) for x in obj[:3]):
            p = [float(x) for x in obj[:3]]
            o = [float(x) for x in obj[3:]] if len(obj) > 3 else None
            return p, o

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default="opensai::sensors::KendamaBall::object_pose", help="Redis key that holds the object pose (single-key format)")
    parser.add_argument("--pos-key", default=None, help="Redis key that holds the position array (overrides --key)")
    parser.add_argument("--ori-key", default=None, help="Redis key that holds the orientation array (overrides --key)")
    parser.add_argument("--output", default="object_pose_log.jsonl", help="Output JSONL file")
    parser.add_argument("--rate", type=float, default=100.0, help="Logging rate in Hz")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    args = parser.parse_args()

    redis_client = redis.Redis(host=args.host, port=args.port)

    dt = 1.0 / float(args.rate) if args.rate > 0 else 0.01

    with open(args.output, "a", buffering=1) as fout:
        try:
            print(f"Logging `{args.key}` -> {args.output} at {args.rate} Hz (dt={dt:.4f}s)")
            while True:
                pos = None
                ori = None
                raw_debug = None

                # If user provided separate keys, read them directly
                if args.pos_key or args.ori_key:
                    if args.pos_key:
                        raw_p = redis_client.get(args.pos_key)
                        try:
                            pos = json.loads(raw_p.decode('utf-8')) if raw_p is not None else None
                        except Exception:
                            pos = None
                    if args.ori_key:
                        raw_o = redis_client.get(args.ori_key)
                        try:
                            ori = json.loads(raw_o.decode('utf-8')) if raw_o is not None else None
                        except Exception:
                            ori = None
                    raw_debug = None
                else:
                    raw = redis_client.get(args.key)
                    raw_debug = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
                    pos, ori = parse_pose_raw(raw)

                entry = {"t": time.time(), "position": pos, "orientation": ori}
                if pos is None or ori is None:
                    entry["raw"] = raw_debug

                fout.write(json.dumps(entry) + "\n")
                time.sleep(dt)
        except KeyboardInterrupt:
            print("Interrupted by user, closing log file.")


if __name__ == "__main__":
    main()
