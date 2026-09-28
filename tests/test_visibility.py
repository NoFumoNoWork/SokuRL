"""Check that hidden renderer values cannot change projected observations."""
from dataclasses import replace
import unittest

from soku_rl.render_state import RenderEntity, RenderSnapshot
from soku_rl.visibility import VisibilityConfig, screen_entity, visible_entities, quantize_gauge


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = VisibilityConfig(8, 0.5, 0.02, 0.1)
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

    def test_unknown_weather_does_not_disclose_pose(self):
        scene = replace(self.scene, weather=11)
        self.assertFalse(screen_entity(self.player, scene, self.config).visible)

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


if __name__ == "__main__":
    unittest.main()
