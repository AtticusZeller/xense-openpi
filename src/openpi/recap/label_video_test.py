import numpy as np
import pytest

from openpi.recap import label_video as _label_video


def test_label_strip_colors_columns_by_the_label_of_their_frame():
    strip = _label_video.label_strip(np.array([True, False, False, True]), width=8, height=3)

    assert strip.shape == (3, 8, 3)
    assert strip.dtype == np.uint8
    for column, positive in enumerate([True, True, False, False, False, False, True, True]):
        expected = _label_video.POSITIVE_COLOR if positive else _label_video.NEGATIVE_COLOR
        assert (strip[:, column] == expected).all()


def test_label_strip_handles_more_frames_than_columns():
    strip = _label_video.label_strip(np.arange(100) >= 50, width=4, height=1)
    # Columns show frames 0, 25, 50 and 75.
    assert [tuple(color) for color in strip[0]] == [_label_video.NEGATIVE_COLOR] * 2 + [_label_video.POSITIVE_COLOR] * 2


def test_label_strip_rejects_an_empty_episode():
    with pytest.raises(ValueError, match="non-empty"):
        _label_video.label_strip(np.array([], bool), width=4, height=1)


@pytest.mark.parametrize(
    ("positive", "color"), [(True, _label_video.POSITIVE_COLOR), (False, _label_video.NEGATIVE_COLOR)]
)
def test_annotate_frame_borders_the_image_in_the_label_color(positive, color):
    image = np.full((6, 10, 3), 7, np.uint8)
    strip = _label_video.label_strip(np.zeros(20, bool), width=14, height=4)

    frame = _label_video.annotate_frame(image, strip, 0, 20, positive=positive, border=2)

    assert frame.shape == (6 + 4 + 4, 14, 3)
    assert (frame[2:8, 2:12] == 7).all()
    for edge in (frame[:2], frame[8:10], frame[:10, :2], frame[:10, 12:]):
        assert (edge == color).all()


def test_annotate_frame_moves_the_cursor_with_the_frame_index():
    image = np.zeros((6, 10, 3), np.uint8)
    strip = _label_video.label_strip(np.zeros(20, bool), width=14, height=4)

    frame = _label_video.annotate_frame(image, strip, 10, 20, positive=False, border=2)

    timeline = frame[10:]
    assert (timeline[:, 6:9] == 255).all()  # Frame 10 of 20 is column 7; the cursor is three columns wide.
    assert (timeline[:, :6] == _label_video.NEGATIVE_COLOR).all()
    assert (timeline[:, 9:] == _label_video.NEGATIVE_COLOR).all()


def test_annotate_frame_draws_the_caption_inside_the_image():
    image = np.full((60, 200, 3), 7, np.uint8)
    strip = _label_video.label_strip(np.ones(5, bool), width=216, height=4)

    plain = _label_video.annotate_frame(image, strip, 0, 5, positive=True)
    captioned = _label_video.annotate_frame(image, strip, 0, 5, positive=True, caption="t=0")

    changed = (plain != captioned).any(axis=2)
    assert changed.any()
    assert not changed[:8].any()
    assert not changed[68:].any()


def test_annotate_frame_rejects_a_strip_of_the_wrong_width():
    with pytest.raises(ValueError, match="strip must be"):
        _label_video.annotate_frame(np.zeros((6, 10, 3), np.uint8), np.zeros((4, 10, 3), np.uint8), 0, 1, positive=True)


def test_forced_frames_take_the_third_color_on_the_strip_and_the_border():
    positive, forced = np.array([True, True, False, True]), np.array([False, True, False, True])
    strip = _label_video.label_strip(positive, width=4, height=2, forced=forced)
    assert [tuple(color) for color in strip[0]] == [
        _label_video.POSITIVE_COLOR,
        _label_video.FORCED_COLOR,
        _label_video.NEGATIVE_COLOR,
        _label_video.FORCED_COLOR,
    ]

    image = np.zeros((6, 4, 3), np.uint8)
    wide = _label_video.label_strip(positive, width=8, height=2, forced=forced)
    frame = _label_video.annotate_frame(image, wide, 1, 4, positive=True, forced=True, border=2)
    assert tuple(frame[0, 0]) == _label_video.FORCED_COLOR
