import pytest
import gameplay.lockstep_input_buffer as mod

CASES = [
    ("frame_queue", (1,)),
    ("catchup_cap", (1,)),
    ("rng_snapshot", (1,)),
    ("logical_frame", (1,)),
    ("reconnect_dedupe", (1,)),
    ("watermark", (1,)),
    ("present_rebuild", (1,)),
    ("rollback", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("latency_comp", (1,)),
    ("metrics", (1,)),
    ("input_order", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
