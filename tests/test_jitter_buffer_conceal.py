import pytest
import media.jitter_buffer_conceal as mod

CASES = [
    ("target_hysteresis", (1,)),
    ("phase_continuous", (1,)),
    ("drop_on_frame_boundary", (1,)),
    ("depth_by_time", (1,)),
    ("atomic_index", (1,)),
    ("fade_on_underrun", (1,)),
    ("push", (1,)),
    ("pop", (1,)),
    ("conceal", (1,)),
    ("decode", (1,)),
    ("validate_buffer", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
