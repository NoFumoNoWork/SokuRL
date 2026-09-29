"""Check that hidden renderer values cannot change projected observations."""
from dataclasses import replace
import unittest
import random

from soku_rl.render_state import RenderEntity, RenderSnapshot
from soku_rl.visibility import VisibilityConfig, screen_entity, visible_entities, quantize_gauge
from soku_rl.contours import Contour, SAMPLES, visible_fraction


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = VisibilityConfig(8, 0.5, 0.02, 0.1, 48., 96., 16., .25)
        self.player = RenderEntity(480., 0., 1., 1, 1)
        self.scene = RenderSnapshot(-320., 420., 1., 21,
                                    (self.player, self.player), ((), ()), False)

    def test_calibrated_screen_projection(self):
        visible = screen_entity(self.player, self.scene, self.config)
        self.assertEqual((visible.visible, visible.x, visible.y), (True, 160., 424.))

    def test_hidden_attributes_cannot_leak(self):
        invisible = replace(self.player, alpha=0.)
        other = replace(invisible, x=900., y=800., facing=-1)
        self.assertEqual(screen_entity(invisible, self.scene, self.config),
                         screen_entity(other, self.scene, self.config))
        offscreen = replace(self.player, x=99999.)
        self.assertFalse(screen_entity(offscreen, self.scene, self.config).visible)

    def test_weather_does_not_blank_all_public_pose_information(self):
        scene = replace(self.scene, weather=11)
        self.assertEqual(screen_entity(self.player, scene, self.config),
                         screen_entity(self.player, self.scene, self.config))

    def test_hidden_object_slots_disappear(self):
        hidden = replace(self.player, alpha=0.)
        one = replace(self.scene, objects=((self.player,), ()))
        two = replace(self.scene, objects=((hidden, self.player, hidden), ()))
        self.assertEqual(visible_entities(one, self.config), visible_entities(two, self.config))

    def test_gauge_precision_and_invalid_values(self):
        self.assertEqual(quantize_gauge(9990, 10000, .02), 1.)
        self.assertEqual(quantize_gauge(9820, 10000, .02), .98)
        with self.assertRaises(ValueError):
            quantize_gauge(1, 0, .02)
        with self.assertRaises(ValueError):
            screen_entity(replace(self.player, x=float("nan")), self.scene, self.config)

    def test_complete_overlap_hides_both_without_assuming_draw_order(self):
        players, _ = visible_entities(self.scene, self.config)
        self.assertFalse(any(player.visible for player in players))

    def test_partial_overlap_preserves_enough_contour(self):
        scene = replace(self.scene, players=(self.player, replace(self.player, x=520.)))
        players, _ = visible_entities(scene, self.config)
        self.assertTrue(all(player.visible for player in players))

    def test_transparent_player_does_not_block_visible_player(self):
        scene = replace(self.scene, players=(self.player, replace(self.player, alpha=0.)))
        players, _ = visible_entities(scene, self.config)
        self.assertTrue(players[0].visible)
        self.assertFalse(players[1].visible)

    def test_object_inside_character_contour_is_hidden(self):
        scene = replace(self.scene, players=(self.player, replace(self.player, x=800.)),
                        objects=((replace(self.player, y=48.),), ()))
        players, objects = visible_entities(scene, self.config)
        self.assertTrue(all(player.visible for player in players))
        self.assertEqual(objects, ((), ()))

    def test_visibility_shortcuts_match_the_original_quadrature_exactly(self):
        rng = random.Random(92817)
        for _ in range(1000):
            target = Contour(rng.uniform(-100, 740), rng.uniform(-100, 580),
                             rng.uniform(1, 50), rng.uniform(1, 100), 1.)
            blockers = tuple(Contour(target.x + rng.uniform(-100, 100),
                target.y + rng.uniform(-100, 100), rng.uniform(1, 50), rng.uniform(1, 100),
                rng.choice((.5, .75, 1.))) for _ in range(rng.randrange(10)))
            nearby = tuple(other for other in blockers if target.overlaps(other))
            remaining = 0.
            for dx, dy in SAMPLES:
                x, y = target.x + dx * target.radius_x, target.y + dy * target.radius_y
                if not (0 <= x < 640 and 0 <= y < 480):
                    continue
                weight = 1.
                for other in nearby:
                    if other.contains(x, y):
                        weight *= 1 - other.alpha
                remaining += weight
            self.assertEqual(visible_fraction(target, blockers), remaining / len(SAMPLES))


if __name__ == "__main__":
    unittest.main()
