"""Video frames that show the advantage label of every moment of an episode.

Each frame is the camera image inside a border in the color of its label, with a timeline of the whole
episode's labels and a cursor below it. Watching the video shows directly which moments are labeled positive.
"""

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Same colors as the label backgrounds of scripts/recap/plot_value.py (matplotlib tab:green and tab:red).
POSITIVE_COLOR = (44, 160, 44)
NEGATIVE_COLOR = (214, 39, 40)


def label_strip(positive: np.ndarray, width: int, height: int) -> np.ndarray:
    """Draws the labels of a whole episode as a timeline.

    Column `x` takes the color of the label of frame `x * len(positive) // width`.

    Returns:
        uint8 image of shape (height, width, 3).
    """
    positive = np.asarray(positive, bool)
    if positive.ndim != 1 or len(positive) == 0:
        raise ValueError(f"positive must be a non-empty 1-D array, got shape {positive.shape}")
    frames = np.arange(width) * len(positive) // width
    colors = np.where(positive[frames, None], POSITIVE_COLOR, NEGATIVE_COLOR).astype(np.uint8)
    return np.broadcast_to(colors, (height, width, 3)).copy()


def annotate_frame(
    image: np.ndarray,
    strip: np.ndarray,
    frame_index: int,
    num_frames: int,
    *,
    positive: bool,
    caption: str = "",
    border: int = 8,
) -> np.ndarray:
    """Frames one image in the color of its label and puts the timeline with a cursor below it.

    Args:
        image: HWC uint8 camera image.
        strip: Timeline from `label_strip`, as wide as the image plus both borders.
        frame_index: Position of this frame in the episode; places the white cursor on the timeline.
        num_frames: Length of the episode.
        positive: Label of this frame.
        caption: Text drawn in the top left corner of the image.
        border: Border width in pixels.

    Returns:
        uint8 image of shape (image height + 2 * border + strip height, image width + 2 * border, 3).
    """
    height, width = image.shape[:2]
    if strip.shape[1] != width + 2 * border:
        raise ValueError(f"strip must be {width + 2 * border} wide, got {strip.shape[1]}")
    framed = height + 2 * border
    frame = np.empty((framed + strip.shape[0], width + 2 * border, 3), np.uint8)
    frame[:framed] = POSITIVE_COLOR if positive else NEGATIVE_COLOR
    frame[border : border + height, border : border + width] = image
    frame[framed:] = strip
    cursor = frame_index * strip.shape[1] // num_frames
    frame[framed:, max(cursor - 1, 0) : cursor + 2] = 255
    if not caption:
        return frame
    canvas = Image.fromarray(frame)
    draw, font, origin = ImageDraw.Draw(canvas), ImageFont.load_default(size=18), (border + 6, border + 4)
    left, top, right, bottom = draw.textbbox(origin, caption, font=font)
    draw.rectangle((left - 4, top - 3, right + 4, bottom + 3), fill=(0, 0, 0))
    draw.text(origin, caption, fill=(255, 255, 255), font=font)
    return np.asarray(canvas)
