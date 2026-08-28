import torch

from src.datasets.ownership_transforms import compose_target_frame_ownership


def test_reference_ownership_follows_cl39_frame_geometry():
    probabilities = torch.zeros(6, 16, 16)
    probabilities[5] = 1
    probabilities[0, 4:12, 4:12] = 1
    probabilities[5, 4:12, 4:12] = 0
    transformed, bbox = compose_target_frame_ownership(
        probabilities, [4, 4, 12, 12], [6, 6, 10, 10],
        canvas_size=16, target_face_fraction=0.25, position_offset=(0.1, -0.1),
    )
    torch.testing.assert_close(
        transformed.sum(0), torch.ones(16, 16), atol=1e-6, rtol=0,
    )
    x0, y0, x1, y1 = (int(round(value)) for value in bbox)
    assert float(transformed[0, y0:y1, x0:x1].mean()) > 0.75
