import json
import math
import unittest

import numpy as np

from kendama_swingup_model import (
    ball_candidate_keys_text,
    KendamaGeometry,
    MemoryRedis,
    PendulumState,
    RedisKeys,
    SwingParams,
    Transform,
    ball_anchor_distance,
    ball_anchor_string_error,
    bounded_flange_goal,
    build_model_snapshot,
    catch_cup_target,
    collect_anchor_offset_samples,
    collect_cup_offset_samples,
    diagnose_ball_pose_candidates,
    discover_pose_keys,
    cup_center,
    estimate_anchor_offset_from_hanging_ball,
    estimate_cup_offset_from_ball,
    estimate_velocity_from_pose_samples,
    evaluate_ball_state_validity,
    flange_position_for_cup,
    invalid_ball_state_desired_flange,
    parse_transform_value,
    pendulum_units,
    publish_ball_in_cup_fixture,
    swingup_anchor_accel,
    transform_payload,
    update_catch_latch,
)


class KendamaSwingupModelTest(unittest.TestCase):
    def test_parse_transform_matrix_payload(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [0.4, -0.2, 0.7]

        parsed = parse_transform_value(json.dumps(matrix.tolist()))

        np.testing.assert_allclose(parsed.position, matrix[:3, 3])
        np.testing.assert_allclose(parsed.rotation, np.eye(3))

    def test_flange_position_for_cup_inverts_cup_center(self):
        geometry = KendamaGeometry(cup_offset_f=np.array([0.0, -0.04, 0.1143]))
        rotation = np.eye(3)
        cup_target = np.array([0.55, 0.10, 0.32])

        flange_target = flange_position_for_cup(cup_target, rotation, geometry)
        recovered = cup_center(Transform(flange_target, rotation), geometry)

        np.testing.assert_allclose(recovered, cup_target)

    def test_estimate_cup_offset_from_ball_in_cup(self):
        angle = math.radians(30.0)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Transform(np.array([0.4, -0.1, 0.2]), rotation)
        true_offset = np.array([0.03, -0.02, 0.12])
        ball_center_above_cup = 0.039
        ball_position = transform.position + transform.rotation @ true_offset
        ball_position = ball_position + np.array([0.0, 0.0, ball_center_above_cup])

        estimated = estimate_cup_offset_from_ball(transform, ball_position, ball_center_above_cup)

        np.testing.assert_allclose(estimated, true_offset, atol=1e-12)

    def test_estimate_anchor_offset_from_hanging_ball(self):
        angle = math.radians(-35.0)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Transform(np.array([0.3, 0.2, 0.4]), rotation)
        true_offset = np.array([0.02, -0.03, 0.13])
        string_length = 0.42
        anchor = transform.position + transform.rotation @ true_offset
        ball = anchor + string_length * np.array([0.0, 0.0, -1.0])

        estimated = estimate_anchor_offset_from_hanging_ball(transform, ball, string_length)

        np.testing.assert_allclose(estimated, true_offset, atol=1e-12)

    def test_swingup_accel_pumps_energy_when_below_target(self):
        params = SwingParams(string_length=0.4, energy_fraction=0.92, pump_gain=2.2)
        state = PendulumState(theta=math.radians(20.0), theta_dot=1.5)
        swing_axis = np.array([1.0, 0.0, 0.0])
        _, tangent = pendulum_units(state.theta, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

        accel, energy, target_energy = swingup_anchor_accel(state, params, swing_axis)

        self.assertLess(energy, target_energy)
        self.assertLess(np.dot(accel, tangent), 0.0)

    def test_catch_target_uses_future_descending_crossing(self):
        geometry = KendamaGeometry(ball_center_above_cup=0.039)
        ball_position = np.array([0.50, 0.10, 0.70])
        ball_velocity = np.array([0.20, -0.10, -1.00])

        catch = catch_cup_target(ball_position, ball_velocity, 0.30, geometry, horizon=2.0)

        self.assertIsNotNone(catch)
        cup_target, crossing_time = catch
        self.assertGreater(crossing_time, 0.0)
        self.assertLess(crossing_time, 2.0)
        self.assertAlmostEqual(cup_target[2], 0.30)
        self.assertAlmostEqual(cup_target[0], ball_position[0] + ball_velocity[0] * crossing_time)
        self.assertAlmostEqual(cup_target[1], ball_position[1] + ball_velocity[1] * crossing_time)

    def test_catch_latch_survives_short_prediction_gap(self):
        first = (np.array([0.5, 0.1, 0.3]), 0.25)

        latched = update_catch_latch(first, None, now=1.0, timeout=0.2)
        reused = update_catch_latch(None, latched, now=1.1, timeout=0.2)

        self.assertIsNotNone(reused)
        np.testing.assert_allclose(reused.cup_center_w, first[0])
        self.assertAlmostEqual(reused.crossing_time, first[1])
        self.assertAlmostEqual(reused.updated_at, 1.0)

    def test_catch_latch_expires_after_timeout(self):
        first = (np.array([0.5, 0.1, 0.3]), 0.25)

        latched = update_catch_latch(first, None, now=1.0, timeout=0.2)
        expired = update_catch_latch(None, latched, now=1.3, timeout=0.2)

        self.assertIsNone(expired)

    def test_bounded_flange_goal_limits_step_and_envelope(self):
        base = np.array([0.0, 0.0, 0.0])
        current = np.array([0.0, 0.0, 0.0])
        desired = np.array([1.0, 0.0, 0.0])

        next_goal = bounded_flange_goal(
            current,
            desired,
            base,
            max_step=0.05,
            max_displacement=0.18,
        )

        np.testing.assert_allclose(next_goal, [0.05, 0.0, 0.0])

        far_current = np.array([0.17, 0.0, 0.0])
        next_goal = bounded_flange_goal(
            far_current,
            desired,
            base,
            max_step=0.05,
            max_displacement=0.18,
        )

        np.testing.assert_allclose(next_goal, [0.18, 0.0, 0.0])

    def test_memory_redis_ball_in_cup_fixture_recovers_offset(self):
        client = MemoryRedis()
        keys = RedisKeys("Titania")
        angle = math.radians(-20.0)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Transform(np.array([0.35, 0.12, 0.44]), rotation)
        geometry = KendamaGeometry(cup_offset_f=np.array([0.02, -0.04, 0.12]))
        ball_pose_key = "opensai::sensors::KendamaBall::object_pose"

        publish_ball_in_cup_fixture(client, keys, transform, geometry, ball_pose_key)
        samples, mean, std = collect_cup_offset_samples(
            client,
            keys,
            ball_pose_key,
            sample_count=3,
            sample_period=0.0,
            ball_center_above_cup=geometry.ball_center_above_cup,
            cup_normal_w=np.array([0.0, 0.0, 1.0]),
            sleep_fn=lambda _: None,
        )

        self.assertEqual(samples.shape, (3, 3))
        np.testing.assert_allclose(mean, geometry.cup_offset_f, atol=1e-12)
        np.testing.assert_allclose(std, [0.0, 0.0, 0.0], atol=1e-12)

    def test_memory_redis_hanging_ball_fixture_recovers_anchor_offset(self):
        client = MemoryRedis()
        keys = RedisKeys("Titania")
        angle = math.radians(15.0)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = Transform(np.array([0.35, 0.12, 0.44]), rotation)
        anchor_offset = np.array([0.01, -0.02, 0.11])
        string_length = 0.38
        anchor = transform.position + transform.rotation @ anchor_offset
        ball = anchor + string_length * np.array([0.0, 0.0, -1.0])
        ball_pose_key = "KendamaBall::pos"
        client.set(keys.flange_transform, transform_payload(transform))
        client.set(ball_pose_key, json.dumps(ball.tolist()))

        samples, mean, std = collect_anchor_offset_samples(
            client,
            keys,
            ball_pose_key,
            sample_count=3,
            sample_period=0.0,
            string_length=string_length,
            anchor_to_ball_direction_w=np.array([0.0, 0.0, -1.0]),
            sleep_fn=lambda _: None,
        )

        self.assertEqual(samples.shape, (3, 3))
        np.testing.assert_allclose(mean, anchor_offset, atol=1e-12)
        np.testing.assert_allclose(std, [0.0, 0.0, 0.0], atol=1e-12)

    def test_model_snapshot_reports_missing_ball_keys(self):
        client = MemoryRedis()
        keys = RedisKeys("Titania")
        transform = Transform(np.array([0.35, 0.12, 0.44]), np.eye(3))
        client.set(keys.flange_transform, transform_payload(transform))

        snapshot = build_model_snapshot(
            client,
            keys,
            KendamaGeometry(),
            SwingParams(),
            "opensai::sensors::KendamaBall::object_pose",
            "opensai::sensors::KendamaBall::object_velocity",
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertIsNotNone(snapshot.flange)
        self.assertIsNotNone(snapshot.cup_w)
        self.assertIsNone(snapshot.ball_w)
        self.assertEqual(len(snapshot.missing_keys), 2)
        np.testing.assert_allclose(snapshot.flange.position, transform.position)

    def test_model_snapshot_uses_fallback_ball_pose_key(self):
        client = MemoryRedis()
        keys = RedisKeys("Titania")
        transform = Transform(np.array([0.0, 0.0, 0.0]), np.eye(3))
        client.set(keys.flange_transform, transform_payload(transform))
        client.set("KendamaBall::pos", json.dumps([0.1, 0.2, 0.3]))

        snapshot = build_model_snapshot(
            client,
            keys,
            KendamaGeometry(),
            SwingParams(),
            "opensai::sensors::KendamaBall::object_pose,KendamaBall::pos",
            "opensai::sensors::KendamaBall::object_velocity",
            np.array([1.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(snapshot.ball_w, [0.1, 0.2, 0.3])
        self.assertEqual(snapshot.missing_keys, ["opensai::sensors::KendamaBall::object_velocity"])

    def test_model_snapshot_with_ball_state_computes_pendulum_and_catch(self):
        client = MemoryRedis()
        keys = RedisKeys("Titania")
        geometry = KendamaGeometry(cup_offset_f=np.array([0.0, 0.0, 0.10]))
        transform = Transform(np.array([0.0, 0.0, 0.0]), np.eye(3))
        ball_pose_key = "opensai::sensors::KendamaBall::object_pose"
        ball_velocity_key = "opensai::sensors::KendamaBall::object_velocity"

        client.set(keys.flange_transform, transform_payload(transform))
        client.set(ball_pose_key, json.dumps({"position": [0.05, 0.0, 0.50]}))
        client.set(ball_velocity_key, json.dumps([0.0, 0.0, -1.0]))

        snapshot = build_model_snapshot(
            client,
            keys,
            geometry,
            SwingParams(string_length=0.4),
            ball_pose_key,
            ball_velocity_key,
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertEqual(snapshot.missing_keys, [])
        self.assertIsNotNone(snapshot.pendulum)
        self.assertIsNotNone(snapshot.energy)
        self.assertIsNotNone(snapshot.target_energy)
        self.assertIsNotNone(snapshot.catch)
        self.assertIsNotNone(snapshot.ball_anchor_distance)
        self.assertIsNotNone(snapshot.string_length_error)
        np.testing.assert_allclose(snapshot.cup_w, [0.0, 0.0, 0.10])
        np.testing.assert_allclose(snapshot.flange.rotation, transform.rotation)

    def test_ball_anchor_string_error(self):
        anchor = np.array([0.0, 0.0, 0.0])
        ball = np.array([0.3, 0.4, 0.0])

        self.assertAlmostEqual(ball_anchor_distance(anchor, ball), 0.5)
        self.assertAlmostEqual(ball_anchor_string_error(anchor, ball, 0.4), 0.1)

    def test_ball_pose_candidate_diagnostics_scores_all_candidates(self):
        client = MemoryRedis()
        cup = np.array([0.0, 0.0, 0.10])
        anchor = np.array([0.0, 0.0, 0.0])
        client.set("good", json.dumps([0.0, 0.0, 0.40]))
        client.set("far", json.dumps([1.0, 0.0, 0.0]))
        client.set("bad", "not-json")

        diagnostics = diagnose_ball_pose_candidates(
            client,
            "missing,good,far,bad",
            cup,
            anchor,
            string_length=0.4,
        )

        self.assertEqual([item.key for item in diagnostics], ["missing", "good", "far", "bad"])
        self.assertFalse(diagnostics[0].available)
        np.testing.assert_allclose(diagnostics[1].position_w, [0.0, 0.0, 0.4])
        self.assertAlmostEqual(diagnostics[1].cup_distance, 0.3)
        self.assertAlmostEqual(diagnostics[1].anchor_distance, 0.4)
        self.assertAlmostEqual(diagnostics[1].string_length_error, 0.0)
        self.assertAlmostEqual(diagnostics[2].string_length_error, 0.6)
        self.assertIsNotNone(diagnostics[3].parse_error)

    def test_ball_candidate_keys_text_adds_default_tracker_fallback(self):
        keys_text = ball_candidate_keys_text(
            "opensai::sensors::KendamaBall::object_pose,KendamaBall::pos,KendamaBall::pos",
            None,
        )

        self.assertEqual(
            keys_text,
            "opensai::sensors::KendamaBall::object_pose,KendamaBall::pos,KendamaBallBlue::pos",
        )

    def test_discover_pose_keys_scans_redis_and_filters_non_pose_payloads(self):
        client = MemoryRedis()
        client.set("KendamaBall::pos", json.dumps([0.1, 0.2, 0.3]))
        client.set("KendamaBall::ori", json.dumps([0.0, 0.0, 0.0, 1.0]))
        client.set("KendamaBall::velocity", json.dumps([0.0, 0.0, 0.0]))
        client.set("KendamaBallBlue::pos", json.dumps({"position": [0.0, 0.0, 0.0]}))
        client.set("OtherBall::pos", "not-json")

        discovered = discover_pose_keys(
            client,
            "*KendamaBall*,*KendamaBallBlue*,*OtherBall*",
            max_scan_keys=20,
        )

        self.assertEqual(discovered, ["KendamaBall::pos", "KendamaBallBlue::pos"])

    def test_ball_candidate_keys_text_appends_discovered_keys_once(self):
        keys_text = ball_candidate_keys_text(
            "KendamaBall::pos",
            "explicit",
            discovered_keys=["KendamaBall::pos", "scanned", "scanned"],
        )

        self.assertEqual(keys_text, "explicit,KendamaBall::pos,scanned")

    def test_estimate_velocity_from_pose_samples_uses_pose_candidates(self):
        class SequenceRedis(MemoryRedis):
            def __init__(self, key, values):
                super().__init__()
                self.key = key
                self.values_sequence = list(values)
                self.index = 0

            def get(self, key):
                if key != self.key:
                    return None
                value = self.values_sequence[min(self.index, len(self.values_sequence) - 1)]
                self.index += 1
                return value

        pose_key = "KendamaBall::pos"
        positions = [
            json.dumps([0.0, 0.0, 0.0]),
            json.dumps([0.1, 0.0, 0.0]),
            json.dumps([0.2, 0.0, 0.0]),
        ]
        client = SequenceRedis(pose_key, positions)

        velocity = estimate_velocity_from_pose_samples(
            client,
            "opensai::sensors::KendamaBall::object_pose,KendamaBall::pos",
            sample_count=3,
            sample_period=0.1,
            sleep_fn=lambda _: None,
        )

        np.testing.assert_allclose(velocity, [1.0, 0.0, 0.0], atol=1e-12)

    def test_model_snapshot_uses_estimated_velocity_override(self):
        client = MemoryRedis()
        keys = RedisKeys("Titania")
        geometry = KendamaGeometry(cup_offset_f=np.array([0.0, 0.0, 0.10]))
        transform = Transform(np.array([0.0, 0.0, 0.0]), np.eye(3))
        ball_pose_key = "KendamaBall::pos"

        client.set(keys.flange_transform, transform_payload(transform))
        client.set(ball_pose_key, json.dumps([0.05, 0.0, 0.50]))

        snapshot = build_model_snapshot(
            client,
            keys,
            geometry,
            SwingParams(string_length=0.4),
            ball_pose_key,
            "opensai::sensors::KendamaBall::object_velocity",
            np.array([1.0, 0.0, 0.0]),
            ball_velocity_override=np.array([0.0, 0.0, -1.0]),
        )

        self.assertEqual(snapshot.missing_keys, [])
        self.assertTrue(snapshot.ball_velocity_estimated)
        self.assertIsNotNone(snapshot.pendulum)


if __name__ == "__main__":
    unittest.main()
