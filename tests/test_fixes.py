"""六个修复点的行为测试：滞回 / 保相位 / 帧边界 / 时间深度 / 原子指针 / 淡入。"""

import threading

import pytest

import media.jitter_buffer_conceal as mod
from media.jitter_buffer_conceal import (
    AtomicIndex,
    HysteresisController,
    JitterBuffer,
    Packet,
    atomic_index,
    audit,
    conceal,
    crossfade_junction,
    decode,
    depth_by_time,
    determinism,
    drop_on_frame_boundary,
    fade_on_underrun,
    fallback,
    make_packet,
    metrics,
    phase_continuous,
    pop,
    push,
    target_hysteresis,
)

SR = mod.SAMPLE_RATE


# ---------------------------------------------------------------------------
# 修复点 1：目标延迟滞回
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_target_holds_inside_band(self):
        ctrl = HysteresisController(nominal=80.0, margin=20.0, dwell=3)
        # 深度在目标附近反复抖动（滞回带 60~100ms），目标不得翻跳
        for depth in (75.0, 85.0, 70.0, 95.0, 80.0) * 4:
            assert ctrl.update(depth) == 80.0

    def test_needs_dwell_before_adjusting(self):
        ctrl = HysteresisController(nominal=80.0, margin=20.0, dwell=3)
        # 单次/两次越界不调整
        assert ctrl.update(40.0) == 80.0
        assert ctrl.update(40.0) == 80.0
        # 连续第 3 次越界才上调一个步长
        assert ctrl.update(40.0) == 100.0

    def test_schmitt_bands_are_asymmetric(self):
        ctrl = HysteresisController(nominal=80.0, margin=20.0, dwell=1)
        # 低于低水位 -> 上调；回到带内 -> 保持；高于新高水位 -> 下调
        assert ctrl.update(50.0) == 100.0
        assert ctrl.update(95.0) == 100.0
        assert ctrl.update(130.0) == 80.0

    def test_stateless_function_entry(self):
        assert target_hysteresis(80.0) == 80.0
        state = {}
        for _ in range(3):
            target = target_hysteresis(10.0, state)
        assert target > 80.0

    def test_clamped_to_range(self):
        ctrl = HysteresisController(nominal=80.0, min_ms=60.0, max_ms=100.0, dwell=1)
        for _ in range(20):
            ctrl.update(0.0)
        assert ctrl.target <= 100.0
        for _ in range(20):
            ctrl.update(10_000.0)
        assert ctrl.target >= 60.0


# ---------------------------------------------------------------------------
# 修复点 5：指针原子更新
# ---------------------------------------------------------------------------


class TestAtomicIndex:
    def test_concurrent_advance_no_lost_update(self):
        idx = AtomicIndex(0)
        n_threads, per_thread = 8, 5_000

        def worker():
            for _ in range(per_thread):
                idx.advance()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert idx.get() == n_threads * per_thread

    def test_function_entry(self):
        assert atomic_index(1) == 2
        idx = AtomicIndex(10)
        assert atomic_index(idx, 5) == 15
        assert idx.get() == 15

    def test_compare_and_set(self):
        idx = AtomicIndex(3)
        assert idx.compare_and_set(3, 7) is True
        assert idx.get() == 7
        assert idx.compare_and_set(3, 9) is False
        assert idx.get() == 7

    def test_snapshot_consistency(self):
        buf = JitterBuffer()
        for seq in range(5):
            push(buf, make_packet(seq))
        snap = buf.pointers()
        # 同一把锁内取的快照内部自洽：队列帧数不超过已写入数
        assert snap["write"] >= snap["queued_frames"]
        assert snap["read"] == 0


# ---------------------------------------------------------------------------
# 修复点 4：深度按时间，而不是按包数
# ---------------------------------------------------------------------------


class TestDepthByTime:
    def test_same_packet_count_different_time_depth(self):
        twenty = [make_packet(i, duration_ms=20.0) for i in range(5)]
        sixty = [make_packet(i, duration_ms=60.0) for i in range(5)]
        assert len(twenty) == len(sixty) == 5
        assert depth_by_time(twenty) == pytest.approx(100.0)
        assert depth_by_time(sixty) == pytest.approx(300.0)

    def test_buffer_depth_is_milliseconds(self):
        buf = JitterBuffer()
        push(buf, make_packet(0, duration_ms=60.0))
        push(buf, make_packet(1, duration_ms=20.0))
        assert depth_by_time(buf) == pytest.approx(80.0)
        assert metrics(buf)["depth_by_packets"] == 2

    def test_empty_is_zero(self):
        assert depth_by_time(JitterBuffer()) == 0.0


# ---------------------------------------------------------------------------
# 修复点 3：按帧边界丢弃
# ---------------------------------------------------------------------------


class TestFrameBoundaryDrop:
    def test_drops_only_whole_frames(self):
        buf = JitterBuffer()
        for seq in range(10):
            push(buf, make_packet(seq, duration_ms=20.0))
        dropped = drop_on_frame_boundary(buf, target_ms=100.0)
        # 每帧 20ms，深 200ms -> 丢 5 个整帧；队列里剩余帧全部仍是完整帧
        assert len(dropped) == 5
        assert [p.seq for p in dropped] == [0, 1, 2, 3, 4]
        for f in buf._frames:
            assert len(f.samples) == int(20.0 * SR / 1000.0)
        assert depth_by_time(buf) == pytest.approx(100.0)

    def test_never_splits_a_frame_with_mixed_durations(self):
        buf = JitterBuffer()
        push(buf, make_packet(0, duration_ms=20.0))
        push(buf, make_packet(1, duration_ms=60.0))
        push(buf, make_packet(2, duration_ms=20.0))
        dropped = drop_on_frame_boundary(buf, target_ms=90.0)
        # 20+60+20=100 > 90：丢最旧的 20ms 后余 80 <= 90，且不存在半帧
        assert [p.seq for p in dropped] == [0]
        assert all(
            len(f.samples) == int(f.duration_ms * SR / 1000.0)
            for f in buf._frames
        )

    def test_decode_rejects_half_frame(self):
        half = Packet(
            seq=1,
            ts_ms=20.0,
            duration_ms=20.0,
            samples=[0.0] * (int(20.0 * SR / 1000.0) - 7),
        )
        with pytest.raises(ValueError):
            decode(half)

    def test_push_rejects_half_frame(self):
        half = Packet(
            seq=0,
            ts_ms=0.0,
            duration_ms=20.0,
            samples=[0.0] * 3,
        )
        buf = JitterBuffer()
        result = push(buf, half)
        assert result["status"] == "rejected"
        assert depth_by_time(buf) == 0.0


# ---------------------------------------------------------------------------
# 修复点 2：丢包保相位
# ---------------------------------------------------------------------------


class TestPhaseContinuity:
    def test_periodic_extension_matches_at_wrap(self):
        template = make_packet(0).samples
        synth, phase = phase_continuous(template, len(template), sample_rate=SR)
        # 周期延拓在回绕点上与模板同位置取值相等（自身连续）
        period = mod.estimate_pitch_period(template, SR)
        assert period is not None
        assert synth[period] == pytest.approx(template[0])

    def test_multi_frame_concealment_keeps_phase(self):
        template = make_packet(0).samples
        n = len(template)
        first, phase = phase_continuous(template, n, sample_rate=SR)
        second, _ = phase_continuous(template, n, sample_rate=SR, phase=phase)
        # 第二段首样本 = 第一段末样本的下一个周期位置，跨帧无相位跳变
        period = mod.estimate_pitch_period(template, SR)
        assert second[0] == pytest.approx(template[(n) % period])

    def test_crossfade_is_continuous_at_junction(self):
        template = make_packet(0).samples
        concealed_tail = template[-100:]
        real = make_packet(2).samples
        blended = crossfade_junction(concealed_tail, real, SR)
        n = min(len(real), len(concealed_tail), mod._fade_sample_count(SR))
        # 接缝首样本权重完全在合成段：与重叠窗口起点逐值相等，无阶跃
        assert blended[0] == pytest.approx(concealed_tail[len(concealed_tail) - n])
        # 斜坡爬升：前几个样本由合成段主导
        assert blended[1] != pytest.approx(real[1])
        # 重叠窗口之外回归真实帧
        assert blended[n] == pytest.approx(real[n])
        assert blended[-1] == pytest.approx(real[-1])

    def test_pipeline_conceals_gap_then_crossfades_recovery(self):
        buf = JitterBuffer()
        push(buf, make_packet(0))
        push(buf, make_packet(1))
        # seq 2、3 丢失，直接给 seq 4
        push(buf, make_packet(4))
        out = [pop(buf) for _ in range(5)]
        assert [f.seq for f in out] == [0, 1, 2, 3, 4]
        assert out[2].concealed and out[3].concealed
        assert not out[4].concealed
        assert metrics(buf)["stats"]["crossfades"] >= 1
        # 恢复帧头部被淡化：首样本完全等于合成段尾部（无阶跃），且明显
        # 小于该点原始真实信号
        assert abs(out[4].samples[0]) < 0.8


# ---------------------------------------------------------------------------
# 修复点 6：欠载淡入
# ---------------------------------------------------------------------------


class TestFadeOnUnderrun:
    def test_fade_in_starts_from_zero_gain(self):
        faded = fade_on_underrun([1.0] * 1000, sample_rate=SR)
        assert faded[0] == pytest.approx(0.0, abs=1e-9)
        # 斜坡单调上升，斜坡结束后恢复原值
        assert faded[1] > faded[0]
        assert faded[-1] == 1.0

    def test_fade_out_ends_at_zero_gain(self):
        faded = fade_on_underrun([1.0] * 1000, sample_rate=SR, direction="out")
        assert faded[-1] == pytest.approx(0.0, abs=1e-9)
        assert faded[0] == 1.0

    def test_underrun_packet_is_faded(self):
        buf = JitterBuffer()
        underrun = pop(buf)  # 空缓冲直接出队 -> PLC 欠载
        assert underrun.concealed
        assert underrun.samples[0] == pytest.approx(0.0, abs=1e-9)

    def test_smoke_int_entry(self):
        # 契约测试风格：裸整数入参也必须返回非空结果
        assert len(fade_on_underrun(10)) == 10


# ---------------------------------------------------------------------------
# 端到端：完整流水线 + 确定性
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_scripted_loss_sequence(self):
        buf = JitterBuffer()
        lost = {3, 4, 8}
        for seq in range(12):
            if seq not in lost:
                push(buf, make_packet(seq))
        frames = [decode(pop(buf), SR) for _ in range(12)]
        assert [f.seq for f in frames] == list(range(12))
        assert {f.seq for f in frames if "concealed" in f.flags} == lost
        m = metrics(buf)
        assert m["stats"]["underruns"] == 3
        assert m["stats"]["crossfades"] >= 1
        assert mod.audit(buf)["ok"]

    def test_duplicate_and_late_packets(self):
        buf = JitterBuffer()
        p0 = make_packet(0)
        assert push(buf, p0)["status"] == "stored"
        assert push(buf, p0)["status"] == "duplicate"
        pop(buf)
        assert push(buf, make_packet(0))["status"] == "late"

    def test_capacity_drop_is_frame_aligned(self):
        buf = JitterBuffer()
        for seq in range(40):  # 40 * 20ms = 800ms > 400ms 硬容量
            push(buf, make_packet(seq))
        for f in buf._frames:
            assert len(f.samples) == int(20.0 * SR / 1000.0)
        # 硬容量溢出按整帧丢回容量线，最多多欠一个帧时长
        assert depth_by_time(buf) <= buf.capacity_ms + 20.0 + 1e-9
        assert metrics(buf)["stats"]["dropped"] >= 1

    def test_long_loss_falls_back_to_comfort_noise(self):
        buf = JitterBuffer()
        push(buf, make_packet(0))
        first = pop(buf)
        frames = [pop(buf) for _ in range(mod.MAX_CONCEAL_FRAMES + 2)]
        assert any(f.comfort_noise for f in frames)
        assert metrics(buf)["stats"]["fallback"] >= 1

    def test_determinism_report(self):
        report = determinism()
        assert report["deterministic"] is True
        assert len(report["signature"]) == 64

    def test_conceal_and_fallback_smoke(self):
        pkt = conceal(20)
        assert pkt.concealed and len(pkt.samples) > 0
        cn = fallback(20)
        assert cn.comfort_noise and len(cn.samples) > 0
        assert decode(1).seq == 1
