"""Color masks preserve all channel values, views, and the caller's image."""
import numpy as np
import pytest

from kindle._qbert_guidance import rgb_mask


@pytest.mark.parametrize('color', [(0, 0, 0), (255, 255, 255), (45, 87, 176),
                                  (210, 210, 64), (181, 83, 40), (146, 70, 192),
                                  (0, 127, 255)])
@pytest.mark.parametrize('layout', ['contiguous', 'reversed', 'transposed'])
def test_mask_matches_every_single_channel_value_without_mutation(color, layout):
    image = np.broadcast_to(np.array(color, dtype=np.uint8), (3, 256, 3)).copy()
    for channel in range(3):
        image[channel, :, channel] = np.arange(256, dtype=np.uint8)
    if layout == 'reversed':
        image = image[:, ::-1]
    elif layout == 'transposed':
        image = image.transpose(1, 0, 2)
    before = image.copy()
    expected = np.all(image == color, axis=-1)
    actual = rgb_mask(image, color)
    assert actual.dtype == np.bool_ and actual.shape == image.shape[:-1]
    assert actual.sum() == 3
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)
