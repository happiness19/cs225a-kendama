# Kendama Swing-Up Model

This repo now has an explicit flange-space model in `kendama_swingup_model.py`.
It is meant to replace raw joint oscillation with a model that can be calibrated,
monitored, and then armed on Titania.

## Frames

- `W`: world frame.
- `F`: Titania flange frame from `opensai::sensors::Titania::flange_transform`.
- `A`: string anchor point.
- `C`: cup center.
- `B`: ball center.

The fixed geometry is:

```text
p_A = p_F + R_WF r_FA
p_C = p_F + R_WF r_FC
```

The default physical cup offset is `r_FC = [0, 0, 0.1143]` m, the 4.5 inch
flange-to-cup estimate. The simulated parallel URDF offset is available with
`--urdf-cup-offset` and uses `[0, -0.04, 0.20]` m.

## Swing-Up

During swing-up, the ball is modeled as a taut-string pendulum in a selected
vertical swing plane:

```text
B = A + L e_r(theta)
theta_ddot = ((g - a_A) dot e_theta) / L
E = 0.5 (L theta_dot)^2 + g L (1 - cos(theta))
E_dot = -L theta_dot (a_A dot e_theta)
```

The controller pumps energy by choosing anchor acceleration opposite the
current tangential ball velocity when the pendulum energy is below target. A
recenter term keeps the commanded anchor motion bounded around the initial
anchor position.

## Catch

The live catch mode predicts where the descending ball crosses the catch plane
by rolling the taut pendulum model forward with the current anchor held fixed:

```text
theta_dot_dot = (g dot e_theta) / L
B(t) = A + L e_r(theta(t))
z_B(t*) = z_C + ball_center_above_cup
p_C,des = [x_B(t*), y_B(t*), z_C]
p_F,des = p_C,des - R_WF r_FC
```

The forecast horizon is `--catch-time-horizon`, and the integration step is
`--catch-prediction-dt`.

The script keeps a ballistic predictor for free-flight reference checks:

```text
z_B(t) = z_B0 + v_Bz t - 0.5 g t^2
p_C,des = [x_B0 + v_Bx t, y_B0 + v_By t, z_C]
p_F,des = p_C,des - R_WF r_FC
```

The live controller latches the last valid catch target briefly, so one bad
sensor frame does not freeze the catch. It times out back to swing-up if the
prediction is lost. The damping dip starts only when the ball is descending,
near catch height, and horizontally aligned with the measured cup center.

## Command Guardrails

Redis modes use the same duration flag as the dry runs. The default is an
8 second run; pass `--duration 0` for an infinite monitor or command loop.

The flange command is bounded two ways:

- `--max-step`: maximum flange goal movement per controller cycle.
- `--max-flange-displacement`: maximum distance from the starting flange
  position.

These limits bound the commanded goal, not the robot's actual tracking error.
The printed measured cup position should still be watched during first trials.
Command mode also checks the initial ball-anchor distance before arming. If
`|distance - --string-length| > --max-string-length-error`, the script exits
before switching to the Cartesian controller. This prevents applying the
taut-string swing-up model when the sensed ball is not plausibly attached at
the configured string length.

The same validity check runs every Redis cycle. If the live ball state violates
the taut-string assumptions, or if the ball speed exceeds `--max-ball-speed`,
the loop clears any catch target and stops using pendulum/catch updates until
the state becomes plausible again. The default invalid-state action is
`--invalid-state-action recenter`, which walks the flange goal back toward the
starting pose through the same `--max-step` and `--max-flange-displacement`
limits. Use `--invalid-state-action hold` if a trial should freeze the last
flange goal instead.

## Ball Sensing

The default ball pose key is a candidate list:

```text
opensai::sensors::KendamaBall::object_pose,KendamaBall::pos
```

That preserves the OpenSai-style key while also accepting the current tracker
key observed in Redis. The default velocity key remains:

```text
opensai::sensors::KendamaBall::object_velocity
```

If the velocity key is unavailable, pass `--estimate-ball-velocity` to estimate
linear velocity from consecutive pose samples. This is useful for monitor and
preflight checks, but first command trials should be conservative because
finite-difference velocity can be noisy.

When Redis has multiple plausible ball keys, score all candidates against the
current cup/anchor geometry:

```bash
python3 kendama_swingup_model.py \
  --preflight \
  --estimate-ball-velocity \
  --diagnose-ball-candidates \
  --scan-ball-candidates
```

The diagnostic prints each candidate's position, cup distance, anchor distance,
and `string_err = anchor_dist - --string-length`. Command mode uses the same
check and refuses to arm if the selected ball pose is not plausibly on the
configured string.

`--scan-ball-candidates` scans Redis using pose-like ball patterns, filters out
orientation/velocity keys, and appends parseable position keys to the
diagnostic list. It does not change the control key; use `--ball-pose-key` to
select the candidate used by the monitor or controller after the geometry check
looks right.

## Real-Arm Sequence

1. Run the hardware-free payload/calibration fixture:

   ```bash
   python3 kendama_swingup_model.py --dry-run-calibration
   ```

2. Check live Redis parsing before commanding anything:

   ```bash
   python3 kendama_swingup_model.py --preflight --estimate-ball-velocity
   ```

   This should at least print a valid flange position and computed cup center.
   If the ball pose or velocity keys are missing, the preflight output lists
   them explicitly. Also check `ball_anchor_distance` and
   `string_length_error`; a large string error means the sensed ball is not in
   a configuration where the taut-string swing-up model is valid.

   If that error is large, run the same command with
   `--diagnose-ball-candidates --scan-ball-candidates` to find whether a
   different Redis ball key is geometrically consistent.

3. Put the ball in the cup with the cup normal approximately upward.
4. Estimate the physical cup offset from live Redis sensor data:

   ```bash
   python3 kendama_swingup_model.py --calibrate-cup-from-ball
   ```

5. With the ball hanging still on a taut string, estimate the string anchor
   offset:

   ```bash
   python3 kendama_swingup_model.py \
     --calibrate-anchor-from-hanging-ball \
     --ball-pose-key KEY \
     --string-length L
   ```

   This assumes the direction from anchor to ball is world down. If the
   hanging string is deliberately tilted, set
   `--hanging-anchor-to-ball-direction-world X,Y,Z`.

6. Monitor without commanding:

   ```bash
   python3 kendama_swingup_model.py \
     --redis \
     --estimate-ball-velocity \
     --cup-offset-f X,Y,Z \
     --anchor-offset-f X,Y,Z \
     --duration 8
   ```

7. Arm the flange Cartesian controller only after the printed cup center,
   pendulum angle, energy, and catch targets look consistent:

   ```bash
   python3 kendama_swingup_model.py \
     --command \
     --cup-offset-f X,Y,Z \
     --anchor-offset-f X,Y,Z \
     --duration 4 \
     --max-anchor-accel 2.0 \
     --max-anchor-speed 0.20 \
     --max-anchor-offset 0.05 \
     --max-step 0.0015 \
     --max-flange-displacement 0.08
   ```

The first real trials should use conservative values for `--max-anchor-accel`,
`--max-anchor-speed`, `--max-anchor-offset`, `--max-step`, and
`--max-flange-displacement`.

Useful offline checks:

```bash
python3 kendama_swingup_model.py --duration 2
python3 kendama_swingup_model.py --dry-run-catch --duration 1
```
