#!/usr/bin/env bash
set -euo pipefail

stop_time="${STOP_TIME:-5.327}"

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            echo "Usage: $0 [STOP_TIME_SECONDS|full|none|off] [extra replay args...]"
            echo
            echo "Default cutoff is 4.793 seconds. Override with a first argument or STOP_TIME=..."
            echo "Examples:"
            echo "  $0 4.65 --yes"
            echo "  STOP_TIME=4.9 $0 --yes"
            echo "  $0 full --yes"
            exit 0
            ;;
        full|none|off)
            stop_time=""
            shift
            ;;
        --*)
            ;;
        *)
            stop_time="$1"
            shift
            ;;
    esac
fi

stop_args=()
if [[ -n "$stop_time" ]]; then
    stop_args=(--stop-time "$stop_time")
fi

uv run python replay_logged_motion.py logs/close_enough.csv \
    --mode joint \
    --expected-config kendama_joint_replay.xml \
    --joint-feedforward velocity-acceleration \
    --speed 1.0 \
    "${stop_args[@]}" \
    --execute \
    "$@"
