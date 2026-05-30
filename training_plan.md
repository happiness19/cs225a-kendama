# MuJoCo RL Training Environment Plan

## Summary

Build a MuJoCo-based RL training setup on the `training` branch with an OpenSai-compatible interface. The environment will use Gymnasium + Stable-Baselines3 PPO, Cartesian cup-delta actions, scripted reset/launch for v1, and checkpoint/video visualization throughout training.

The design should follow the lessons in `training.md`: decompose the task, start with a reduced Cartesian action space, reward catch geometry rather than simple distance, add sensing/contact randomization, and preserve the OpenSai/Redis control contract.

Generated training artifacts will be organized by run:

- parameters: `checkpoints/<run_name>/params/`
- videos: `checkpoints/<run_name>/videos/`
- latest symlink: `checkpoints/latest`

`<run_name>` defaults to local time formatted as `YYYYMMDD_HHMMSS`.

## Key Implementation Changes

- Add a Python RL package using:
  - Gymnasium for the environment API.
  - Stable-Baselines3 PPO as the default trainer.
  - MuJoCo Python bindings for physics and rendering.
  - `imageio`/ffmpeg for MP4 video generation.
- Add `requirements-training.txt` and update `.gitignore` for generated artifacts: `checkpoints/`, TensorBoard/log dirs, temporary converted MuJoCo assets, and rendered videos.
- Convert or author a MuJoCo MJCF model based on the existing Rizon/kendama assets:
  - `Rizon4r_with_kendama_parallel.urdf`
  - `world_single_rizon.urdf`
  - robot visual/collision assets under `rizon4r/`
- Preserve core environment details:
  - 7 revolute arm joints.
  - gravity `0 0 -9.81`.
  - floor.
  - free `KendamaBall`.
  - `kendama_big_cup` body/site for Cartesian control and reward computation.
- Use a single torque-limit policy: `real_driver`.
  - Do not expose a `urdf_sim` or raw-effort profile.
  - Clip all commanded torques to 98% of URDF effort for robustness and closer real-driver behavior.
  - Torque limits are `[120.54, 120.54, 62.72, 62.72, 38.22, 38.22, 38.22]` Nm.
  - Enforce limits both in the controller wrapper before `data.ctrl` assignment and in MJCF actuator `ctrlrange`/`forcerange` where applicable.

## OpenSai-Compatible Environment Interface

- Implement `KendamaOpenSaiEnv` as a Gymnasium environment.
- RL action:
  - 3D Cartesian delta `[dx, dy, dz]` for the cup goal.
  - Clip each component to `+/-0.015 m` per RL step.
  - Hold cup orientation fixed in v1.
- Timing:
  - RL step rate: 200 Hz.
  - MuJoCo physics/control rate: 1 kHz.
  - Apply 5 low-level simulation/control steps per RL action.
- Low-level control:
  - Convert Cartesian cup goal to joint torques using a task-space PD/Jacobian-based controller.
  - Add a nullspace posture target near the scripted launch/catch-safe posture.
  - Clip torques with the fixed `real_driver` 98% limits before applying to MuJoCo.
- Implement an in-memory OpenSai-compatible bus with the same key shapes used by the current scripts:
  - `opensai::controllers::<robot>::active_controller_name`
  - joint task goal/current position, velocity, and acceleration keys.
  - Cartesian task goal/current position and orientation keys.
  - `opensai::sensors::KendamaBall::object_pose`
  - `opensai::sensors::KendamaBall::object_velocity`
  - `::sai-interfaces-webui::config_file_name`
- Observation should include:
  - robot joint position and velocity.
  - current cup pose and velocity.
  - current Cartesian goal.
  - ball pose and velocity.
  - active controller state.
  - optional delayed/noisy ball observation for robustness training.
- Episode v1:
  - scripted reset.
  - scripted launch/throw.
  - RL controls the catch phase.
  - terminate on catch success, ball drop, unsafe state, or timeout.
- Reward should include:
  - ball descending through the cup opening from above.
  - XY alignment near rim crossing.
  - low relative velocity at catch.
  - final ball retention in the cup.
  - penalties for joint-limit proximity, torque clipping, unsafe speed, and dropped ball.

## Checkpoints, Videos, And README Logging

- Default training command:

  ```bash
  python -m kendama_rl.train --total-timesteps 1000000
  ```

- Save exactly 10 checkpoints per run at evenly spaced progress points:
  - for 1M timesteps: every 100k timesteps.
  - in general: `10%, 20%, ..., 100%` of `--total-timesteps`.
- At each checkpoint:
  - save params to `checkpoints/<run_name>/params/step_<timestep>.zip`.
  - generate an evaluation video at `checkpoints/<run_name>/videos/step_<timestep>.mp4`.
  - run evaluation with deterministic policy actions and fixed eval seeds.
- Maintain `checkpoints/<run_name>/metadata.json` with:
  - run name.
  - command/config.
  - seed.
  - total timesteps.
  - checkpoint timesteps.
  - package versions.
  - torque limits.
  - reward/success summaries.
  - paths to params and videos.
- Maintain `checkpoints/latest` as a symlink to the newest run. Replace it atomically when a new run starts.
- Update `README.md` with:
  - setup instructions.
  - training command.
  - artifact layout.
  - OpenSai compatibility notes.
  - fixed `real_driver` torque limits.
  - link/reference to `training.md`.
  - append-only “Training Runs” entries generated at run start/end.

## Test Plan

- Unit test OpenSai-compatible key generation against keys used in `kendama_throw_and_catch.py` and `kendama_cup_flip.py`.
- Smoke test MuJoCo model loading, named joint/body/site lookup, ball free body state, and one physics step.
- Validate Gymnasium API compliance:
  - `reset()`
  - `step()`
  - observation spaces.
  - action clipping.
  - reward output.
  - termination/truncation.
  - deterministic seeding.
- Test torque clipping:
  - input torques above/below limits are clipped to `[120.54, 120.54, 62.72, 62.72, 38.22, 38.22, 38.22]`.
  - no alternate torque-limit profile exists.
  - clipped torque values are reflected in diagnostics/reward penalties.
- Test tiny training run:

  ```bash
  python -m kendama_rl.train --total-timesteps 1000 --run-name smoke_test
  ```

  It should produce exactly 10 params files, 10 videos, `metadata.json`, and a valid `checkpoints/latest` symlink.
- Test README logging with a dry-run mode before long training runs.

## Assumptions

- Gymnasium + Stable-Baselines3 PPO is the v1 RL stack.
- Cartesian cup deltas are the v1 trainable action space.
- “Iterations” means SB3 environment timesteps.
- The first trainable task is catch-phase control with scripted reset/launch.
- Torque limits always use the 98% `real_driver` values; raw URDF effort limits are not exposed as a runtime option.
- Generated checkpoints, videos, and logs are ignored by git.

