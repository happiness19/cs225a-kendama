python3 replay_logged_motion.py logs/latest.csv \
    --mode joint \
    --expected-config kendama_joint_replay.xml \
    --joint-feedforward velocity-acceleration \
    --speed 1.0 \
    --execute

