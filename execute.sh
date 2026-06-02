python3 replay_logged_motion.py logs/latest_joint_replay_150hz_ff_amp135.csv \
    --mode joint \
    --expected-config kendama_joint_replay.xml \
    --joint-feedforward velocity-acceleration \
    --speed 1.0 \
    --execute
