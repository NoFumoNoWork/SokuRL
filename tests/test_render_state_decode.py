"""Keep renderer slot decoding identical when unused slots are skipped."""
import pytest

from soku_rl.render_state import CAMERA, ENTITY, COUNTS, MAX_OBJECTS, RenderEntity, RenderSnapshot


@pytest.mark.parametrize("counts", [(0, 0), (1, 0), (0, 1), (7, 19), (64, 64)])
def test_object_counts_retain_the_correct_fixed_slot_offsets(counts):
    rows = tuple((float(i), float(2 * i), .5, -1, 1) for i in range(2 + 2 * MAX_OBJECTS))
    data = CAMERA.pack(10., 20., 1., 21) + b"".join(ENTITY.pack(*row) for row in rows)
    decoded = RenderSnapshot.decode(data + COUNTS.pack(*counts, 0))
    expected = tuple(RenderEntity(*row) for row in rows)
    assert decoded.players == expected[:2]
    assert decoded.objects == (expected[2:2 + counts[0]],
                               expected[2 + MAX_OBJECTS:2 + MAX_OBJECTS + counts[1]])
    assert (decoded.camera_x, decoded.camera_y, decoded.camera_scale, decoded.weather) == (10., 20., 1., 21)


@pytest.mark.parametrize("counts", [(65, 0, 0), (0, 65, 0), (0, 0, 2)])
def test_invalid_counts_fail_before_entity_decoding(counts):
    data = CAMERA.pack(0., 0., 1., 0) + bytes(ENTITY.size * (2 + 2 * MAX_OBJECTS))
    with pytest.raises(ValueError, match="object counts"):
        RenderSnapshot.decode(data + COUNTS.pack(*counts))
