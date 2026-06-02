import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_LOG = Path("logs/latest.csv")
DEFAULT_CUP_OFFSET_F = np.array([0.0, 0.0, 0.1143])


def parse_vec(text: str, expected: int | None = None) -> np.ndarray:
    values = np.array([float(x) for x in text.replace(",", " ").split()], dtype=float)
    if expected is not None and values.size != expected:
        raise ValueError(f"Expected {expected} values, got {values.size}")
    return values


def format_vec(values: np.ndarray) -> str:
    return " ".join(f"{v:.10g}" for v in values)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("Quaternion has near-zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def parse_transform(text: str) -> tuple[np.ndarray, np.ndarray]:
    obj = json.loads(text)
    arr = np.array(obj, dtype=float)
    if arr.shape == (4, 4):
        return arr[:3, 3].copy(), arr[:3, :3].copy()
    if arr.shape == (3, 4):
        return arr[:3, 3].copy(), arr[:3, :3].copy()
    if arr.size == 16:
        mat = arr.reshape(4, 4)
        return mat[:3, 3].copy(), mat[:3, :3].copy()
    if arr.size == 12:
        mat = arr.reshape(3, 4)
        return mat[:3, 3].copy(), mat[:3, :3].copy()
    raise ValueError(f"Cannot parse transform payload with shape {arr.shape}")


def parse_pose(text: str) -> tuple[np.ndarray, np.ndarray]:
    values = parse_vec(text)
    if values.size < 7:
        raise ValueError("Pose must have at least position and quaternion values")
    return values[:3].copy(), quat_wxyz_to_matrix(values[3:7])


def parse_row_flange_pose(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    for column in ("flange_pose", "tcp_pose"):
        pose_text = row.get(column, "").strip()
        if pose_text:
            return parse_pose(pose_text)

    transform_text = row.get("flange_transform", "").strip()
    if transform_text:
        return parse_transform(transform_text)

    raise ValueError("Row has no usable flange_transform, flange_pose, or tcp_pose")


def cup_center_from_row(row: dict[str, str], cup_offset_f: np.ndarray) -> np.ndarray:
    flange_position, flange_rotation = parse_row_flange_pose(row)
    return flange_position + flange_rotation @ cup_offset_f


def default_output_path(input_path: Path) -> Path:
    if input_path.name == "latest.csv":
        return input_path.with_name("latest_cartesian_delta.csv")
    return input_path.with_name(f"{input_path.stem}_cartesian_delta{input_path.suffix}")


def process_log(input_path: Path, output_path: Path, cup_offset_f: np.ndarray) -> int:
    previous_position = None
    rows_written = 0

    with open(input_path, newline="") as src, open(output_path, "w", newline="") as dst:
        reader = csv.DictReader(src)
        fieldnames = [
            "row_index",
            "t_wall",
            "robot_sec",
            "robot_nsec",
            "raw_cartesian_position",
            "cartesian_delta",
            "cartesian_delta_norm",
            "raw_cartesian_x",
            "raw_cartesian_y",
            "raw_cartesian_z",
            "cartesian_delta_x",
            "cartesian_delta_y",
            "cartesian_delta_z",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, row in enumerate(reader):
            position = cup_center_from_row(row, cup_offset_f)
            if previous_position is None:
                delta = np.zeros(3)
            else:
                delta = position - previous_position
            previous_position = position

            writer.writerow(
                {
                    "row_index": row_index,
                    "t_wall": row.get("t_wall", ""),
                    "robot_sec": row.get("robot_sec", ""),
                    "robot_nsec": row.get("robot_nsec", ""),
                    "raw_cartesian_position": format_vec(position),
                    "cartesian_delta": format_vec(delta),
                    "cartesian_delta_norm": f"{np.linalg.norm(delta):.10g}",
                    "raw_cartesian_x": f"{position[0]:.10g}",
                    "raw_cartesian_y": f"{position[1]:.10g}",
                    "raw_cartesian_z": f"{position[2]:.10g}",
                    "cartesian_delta_x": f"{delta[0]:.10g}",
                    "cartesian_delta_y": f"{delta[1]:.10g}",
                    "cartesian_delta_z": f"{delta[2]:.10g}",
                }
            )
            rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process a logger CSV into cup-center Cartesian positions and per-step deltas."
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG,
        help="Input logger CSV. Defaults to logs/latest.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV. Defaults to <input>_cartesian_delta.csv, or logs/latest_cartesian_delta.csv.",
    )
    parser.add_argument(
        "--cup-offset",
        default=format_vec(DEFAULT_CUP_OFFSET_F),
        help="Cup-center offset in the flange frame as 'x y z' meters. Defaults to physical cup offset.",
    )
    args = parser.parse_args()

    input_path = args.logfile
    output_path = args.output or default_output_path(input_path)
    cup_offset_f = parse_vec(args.cup_offset, expected=3)

    rows_written = process_log(input_path, output_path, cup_offset_f)
    print(f"Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
