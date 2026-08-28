import torch

from src.model.photomaker_branched.extensions.correspondence_confidence import (
    compute_cycle_confidence,
)
from src.model.photomaker_branched.extensions.global_local_balancer import (
    compute_global_appearance_delta,
)
from src.model.photomaker_branched.extensions.intrinsic_identity_sidecar import (
    IntrinsicIDTokenProjector,
)
from src.model.photomaker_branched.extensions.ot_reference_transport import (
    StageSplitReferenceTransport,
)
from src.model.photomaker_branched.extensions.roi_reference_route import (
    FaceRoiReferenceRoute,
)
from src.model.photomaker_branched.extensions.valid_kv_attention import (
    packed_attention_oracle,
    valid_key_sdpa,
)


def test_valid_key_attention_matches_packed_oracle_and_falls_back():
    torch.manual_seed(1)
    query, key, value = (torch.randn(2, 2, 4, 3) for _ in range(3))
    valid = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    fallback = torch.randn_like(query)
    result = valid_key_sdpa(
        query, key, value, valid, fallback=fallback, return_entropy=True,
    )
    oracle = packed_attention_oracle(query[:1], key[:1], value[:1], valid[:1])
    torch.testing.assert_close(result.message[:1], oracle)
    torch.testing.assert_close(result.message[1:], fallback[1:])
    assert torch.isfinite(result.entropy).all()


def test_cycle_confidence_prefers_exact_cycle():
    query = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    result = compute_cycle_confidence(
        query, query, torch.ones(1, 4, 1), torch.ones(1, 4, dtype=torch.bool),
        floor=0.25, margin_center=0.04, margin_temperature=0.02,
        cycle_sigma_cells=1.5, chunk_size=2,
    )
    assert float(result.cycle_distance.max()) == 0.0
    assert float(result.cycle_score.min()) == 1.0
    assert float(result.confidence.min()) > 0.25


def test_stage_split_transport_is_finite_and_shape_preserving():
    torch.manual_seed(2)
    query, key, value = (torch.randn(1, 2, 16, 4) for _ in range(3))
    mask = torch.ones(1, 16, 1)
    result = StageSplitReferenceTransport(
        grid_size=4, epsilon=0.10, iterations=4, coordinate_weight=0.15,
        transition_start=0.5, transition_end=0.7, late_top_k=2,
        min_valid_tokens=2,
    )(
        query_heads=query, key_heads=key, value_heads=value,
        target_mask=mask, reference_mask=mask, progress=torch.zeros(1, 1, 1),
        fallback=torch.zeros_like(query),
    )
    assert result.message.shape == query.shape
    assert torch.isfinite(result.message).all()


def test_roi_route_has_exact_zero_effect_initialization():
    torch.manual_seed(3)
    target, reference, native = (torch.randn(1, 64, 4) for _ in range(3))
    mask = torch.zeros(1, 64, 1)
    mask[:, 27:29] = 1
    route = FaceRoiReferenceRoute(
        roi_size=4, face_area_threshold=0.10, box_expansion=0.2,
        gate_max=0.2, delta_native_cap=0.35, boundary_ring_cells=1,
    )
    result = route(
        target_hidden=target, reference_hidden=reference,
        target_mask=mask, reference_mask=mask, native_out=native, heads=1,
        project_q=lambda value: value, project_native_k=lambda value: value,
        project_native_v=lambda value: value,
        project_reference_k=lambda value: value,
        project_reference_v=lambda value: value, project_output=lambda value: value,
    )
    assert torch.count_nonzero(result.delta) == 0
    assert result.eligible_fraction > 0
    result.delta.sum().backward()
    assert route.gate_raw.grad is not None and torch.isfinite(route.gate_raw.grad)


def test_intrinsic_projector_maps_null_identity_to_zero():
    projector = IntrinsicIDTokenProjector(input_dim=8, token_dim=16, num_tokens=2, hidden_dim=12)
    output = projector(torch.zeros(3, 8))
    assert output.shape == (3, 2, 16)
    assert torch.count_nonzero(output) == 0


def test_global_local_branch_is_finite_and_bounded():
    torch.manual_seed(4)
    query, key, value = (torch.randn(1, 1, 16, 4) for _ in range(3))
    face = torch.zeros(1, 16, 1)
    face[:, 5:11] = 1
    native = torch.randn(1, 16, 4)
    result = compute_global_appearance_delta(
        query_heads=query, reference_key_heads=key, reference_value_heads=value,
        target_face_mask=face, reference_face_mask=face,
        native_message=native, native_out=native, project_output=lambda value: value,
        progress=torch.zeros(1, 1, 1), dilation_cells=1,
        early_scale=0.3, late_scale=0.1, native_cap=0.2, local_exclusion=0.5,
    )
    assert result.delta.shape == native.shape
    assert torch.isfinite(result.delta).all()
