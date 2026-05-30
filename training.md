# RL Training Notes

## Lessons From Similar Work

### Decompose the kendama task

The closest prior work on robotic cup-and-ball/kendama-style tasks does not treat the whole behavior as one monolithic end-to-end policy. It separates the launch/swing/throw behavior from the catch behavior. That is a good fit for this repo because the existing scripts already have explicit phases: reset, throw/jolt, release, flip/track, and hold.

Initial RL work should focus on the hard phase with the clearest feedback: tracking and catching the falling ball. Reset and launch can remain scripted until the catch policy is stable.

### Start with a reduced action space

Raw torque control makes exploration hard and creates a large sim-to-real gap. The current OpenSai setup exposes higher-level task goals through Redis, especially Cartesian goal position/orientation for the cup link. A MuJoCo RL environment should start by reproducing that interface:

- action = Cartesian cup target delta, velocity, or next goal pose
- observation = ball state, cup pose, robot joint state, and current task goal
- low-level control = internal PD/task-space controller in MuJoCo

Raw torque control can be added later for final policy refinement, but it should not be the first training interface.

### Reward catch geometry, not just distance

A naive reward based only on distance between ball and cup can produce bad behavior, such as approaching from the side or below. The reward should encode the physical catch event:

- ball is above the cup before catch
- ball is descending through the cup opening
- ball/cup XY alignment at rim crossing
- relative velocity is reasonable at impact
- ball remains inside the cup after contact
- robot stays within joint, velocity, and torque limits

Sparse catch success should be used for evaluation. Shaped terms are useful during early training.

### Randomize contact and sensing

Catching is sensitive to contact and perception mismatch. Domain randomization should include:

- ball mass and radius
- cup collision geometry
- contact friction and restitution
- ball initial position and velocity
- controller tracking error
- action latency
- observation latency
- observation noise

Latency is especially important. Real deployments observe state through Redis/OpenSai and possibly cameras; policies trained on instantaneous perfect state are likely to learn brittle timing.

### Preserve the real control contract

The real robot path uses OpenSai controller state and goals through Redis keys. A useful training environment should mirror that contract even if Redis itself is not used during training:

- same robot names where useful: `Rizon4r` for sim, `Titania` for real
- same controller names: `joint_controller`, `cartesian_controller`
- same task concepts: joint goal, Cartesian goal, current task pose, active controller
- same update rates or an integer-ratio approximation
- same saturation behavior

Driver logs show torque saturation near 98% of the URDF effort values for observed joints. The MuJoCo controller should model this clipping.

### Use privileged state first

Start with simulator state rather than pixels:

- ball pose and velocity
- cup pose and velocity
- robot joint position and velocity
- active controller/task goal

After the environment works, degrade the observation with noise, latency, dropped samples, and estimated ball velocity.

### Build progressively

A practical sequence:

1. Scripted reset and scripted launch; RL controls only catch XY/Z cup motion.
2. Scripted reset; RL controls throw timing and catch motion.
3. RL controls Cartesian cup pose across the full episode.
4. Add observation delay/noise and randomized physics.
5. Add real-robot safety constraints and deployment wrappers.
6. Consider torque-level policies only after the higher-level policy works.

