import types
import time

import torch

from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.sampling.sampling_params import SamplingParams


class DummyTreeCache:
    def __init__(self, page_size=1):
        self.page_size = page_size

    def match_prefix(self, rid=None, key=None):
        # No prefix hit by default
        return torch.empty(0, dtype=torch.bool), None, None, 0


def make_req(rid: str, input_len: int, queue_time_start: float = None):
    sp = SamplingParams()
    # construct origin_input_ids of given length
    origin_ids = list(range(input_len))
    from sglang.srt.managers.schedule_batch import Req

    r = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=origin_ids,
        sampling_params=sp,
    )
    r.queue_time_start = queue_time_start if queue_time_start is not None else time.perf_counter()
    return r


def test_wfq_short_first_ordering_and_aging():
    tree = DummyTreeCache(page_size=1)
    pol = SchedulePolicy("wfq-short-first", tree_cache=tree, enable_hierarchical_cache=False,
                         enable_priority_scheduling=False, schedule_low_priority_values_first=False)

    now = time.perf_counter()
    # S1 (<=512)
    r1 = make_req("r1", 100, now - 1.0)
    # S2 (<=2048)
    r2 = make_req("r2", 1500, now - 1.0)
    # S3 (>2048) but very old to test aging effect
    r3 = make_req("r3", 5000, now - 30.0)

    q = [r2, r3, r1]
    pol.calc_priority(q)

    # Expect S1 first always
    assert q[0].rid == "r1"
    # Very old S3 should outrank S2 with the aging term
    assert q[1].rid == "r3"
    assert q[2].rid == "r2"


def test_soft_cap_on_joiners_only_and_ratio_nudge(monkeypatch):
    # Use the scheduler method unbound, with a dummy self carrying required attrs
    from sglang.srt.managers import scheduler as sched_mod

    class DummySelf:
        pass

    self = DummySelf()
    # Required by get_num_allocatable_reqs
    self._clamp_active = True
    self._decode_capacity_baseline = 8
    class SA:
        decode_soft_cap_factor = 0.5
        decode_soft_cap_min_share = 0.5
    self.server_args = SA()
    self.pp_size = 1
    class Pool:
        def available_size(self):
            return 999
    self.req_to_token_pool = Pool()

    # running_bs=4; baseline*factor=4; allowed joiners = 4-4 = 0
    allowed = sched_mod.Scheduler.get_num_allocatable_reqs(self, running_bs=4)
    assert allowed <= 0

    # Now test ratio nudge via clamp machine in metrics mixin
    from sglang.srt.managers import scheduler_metrics_mixin as mixin_mod

    class ServerArgsObj:
        enable_metrics = False
        ttft_min_samples = 10
        ttft_ema_alpha = 0.5
        decode_lat_rolling_size = 20
        max_micro_batch_size = 8
        enable_ttft_clamp = True
        ttft_target_p95_ms = 100
        ttft_target_p99_ms = None
        max_decode_step_p95_ms = None
        ttft_clamp_sustain_ms = 100
        ttft_clamp_hysteresis_pct = 0.85
        ttft_clamp_cooldown_ms = 10
        ttft_clamp_max_duration_ms = 2000
        decode_log_interval = 1

    class DummySched(mixin_mod.SchedulerMetricsMixin):
        pass

    ds = DummySched()
    ds.enable_metrics = False
    ds.server_args = ServerArgsObj()
    # init required fields
    ds.init_metrics(tp_rank=0, pp_rank=0, dp_rank=None)
    ds.new_token_ratio = 0.5

    # Monkeypatch time to simulate sustained breach
    base = 1000.0
    def fake_time():
        return fake_time.t
    fake_time.t = base
    monkeypatch.setattr(mixin_mod, "time", types.SimpleNamespace(perf_counter=fake_time))

    # Feed TTFT samples above threshold for sustain period
    for _ in range(ds.server_args.ttft_min_samples):
        ds._observe_ttft(0.2)  # 200ms > 100ms
    # Before sustain window elapsed, clamp should not activate
    ds._maybe_update_ttft_clamp()
    assert not ds._clamp_active
    # Advance time to cross sustain window
    fake_time.t = base + (ds.server_args.ttft_clamp_sustain_ms / 1000.0)
    ds._maybe_update_ttft_clamp()
    assert ds._clamp_active
    # Ratio bumped once per call
    r0 = ds.new_token_ratio
    ds._maybe_update_ttft_clamp()
    assert ds.new_token_ratio > r0


def test_internal_state_projected_kv_after_admit_pages():
    # Build a dummy self object to invoke Scheduler.get_internal_state (unbound)
    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.managers.io_struct import GetInternalStateReq

    class DummyModelRunner:
        weight_load_mem_usage = 0.0
        graph_mem_usage = 0.0

    class DummyWorker:
        model_runner = DummyModelRunner()

    class DummyTp:
        worker = DummyWorker()

    class DummyKV:
        mem_usage = 0.0
        def get_kvcache(self):
            return self

    class DummyRunningBatch:
        def __init__(self):
            self.reqs = []
        def new_page_count_next_decode(self):
            return 0

    class DummySelf:
        pass

    self = DummySelf()
    self.tp_worker = DummyTp()
    self.token_to_kv_pool_allocator = DummyKV()
    self.max_total_num_tokens = 10000
    self.page_size = 1
    # No clamp
    self._decode_capacity_baseline = 8
    self.running_batch = DummyRunningBatch()
    self.req_to_token_pool = types.SimpleNamespace(available_size=lambda: 999)
    self.pp_size = 1
    # Provide scheduling policy and queue
    tree = DummyTreeCache(page_size=1)
    self.policy = SchedulePolicy("wfq-short-first", tree_cache=tree, enable_hierarchical_cache=False,
                                 enable_priority_scheduling=False, schedule_low_priority_values_first=False)
    self.waiting_queue = [make_req("s1", 100), make_req("s3", 5000)]
    # Minimal methods used by get_internal_state
    def _get_token_info():
        # available + evictable = headroom tokens
        available = 1000
        evictable = 0
        num_used = self.max_total_num_tokens - (available + evictable)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available, evictable
    self._get_token_info = _get_token_info
    self.get_num_allocatable_reqs = lambda running_bs: 10

    out = sched_mod.Scheduler.get_internal_state(self, GetInternalStateReq())
    ret = out.internal_state
    # With headroom=1000, projected admissions ~ 100 + 5000 = 5100 -> projected headroom pages clipped to 0
    assert ret.get("projected_kv_after_admit_pages", 0) == 0
