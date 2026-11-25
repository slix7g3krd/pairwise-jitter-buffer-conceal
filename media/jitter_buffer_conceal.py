"""音频抖动缓冲与丢包隐藏（jitter buffer + PLC）。

对应六个修复点：

1. 目标延迟滞回 --------- ``target_hysteresis`` / ``HysteresisController``
2. 丢包保相位 ----------- ``phase_continuous`` + 恢复帧交叉淡化
3. 按帧边界丢弃 --------- ``drop_on_frame_boundary``（只丢整帧，不切半帧）
4. 缓冲深度按时间计算 --- ``depth_by_time``（毫秒，而非包数）
5. 读写指针原子更新 ----- ``AtomicIndex`` / ``atomic_index``
6. 欠载淡入 ------------- ``fade_on_underrun``

``push`` / ``pop`` / ``conceal`` / ``decode`` 等函数在上述原语之上组成完整
收包→缓冲→出队→隐藏→解码流水线。仅使用标准库，线程安全（单生产者-单消费者
模型下指针更新加锁、单调计数，杜绝撕裂的环形索引）。
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Packet",
    "DecodedFrame",
    "AtomicIndex",
    "HysteresisController",
    "JitterBuffer",
    "make_packet",
    "target_hysteresis",
    "phase_continuous",
    "drop_on_frame_boundary",
    "depth_by_time",
    "atomic_index",
    "fade_on_underrun",
    "push",
    "pop",
    "conceal",
    "decode",
    "validate_buffer",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
DEFAULT_FRAME_MS = 20.0
NOMINAL_TARGET_MS = 80.0      # 标称目标延迟
MIN_TARGET_MS = 40.0          # 滞回控制器允许的最低目标
MAX_TARGET_MS = 240.0         # 滞回控制器允许的最高目标
CAPACITY_MS = 400.0           # 硬容量（超过即按帧边界丢弃）
HYSTERESIS_STEP_MS = 20.0     # 每次调整步长
HYSTERESIS_MARGIN_MS = 20.0   # 滞回带半宽：低水位 target-margin，高水位 target+margin
DWELL_READINGS = 3            # 连续越界读数次数，防止单次抖动造成目标抖动
FADE_MS = 5.0                 # 欠载淡入 / 交叉淡化时长
MAX_CONCEAL_FRAMES = 4        # 超过该连续丢帧数改用舒适噪声兜底
COMFORT_NOISE_LEVEL = 0.02

# 全局锁：让 atomic_index 对裸整数入参也具备可验证的原子语义
_GLOBAL_INDEX_LOCK = threading.Lock()

# 默认周期模板（440Hz 正弦一个周期），无历史帧时 PLC 的信号源
_DEFAULT_TEMPLATE_CACHE: Dict[int, List[float]] = {}


def _default_template(sample_rate: int = SAMPLE_RATE) -> List[float]:
    if sample_rate not in _DEFAULT_TEMPLATE_CACHE:
        period = max(1, int(round(sample_rate / 440.0)))
        _DEFAULT_TEMPLATE_CACHE[sample_rate] = [
            math.sin(2.0 * math.pi * i / period) for i in range(period)
        ]
    return list(_DEFAULT_TEMPLATE_CACHE[sample_rate])


def _fade_sample_count(sample_rate: int, fade_ms: float = FADE_MS) -> int:
    return max(1, int(round(fade_ms * sample_rate / 1000.0)))


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Packet:
    """一帧音频 = 一个不可分割的缓冲单元（半帧永不入队/出队）。"""

    seq: int
    ts_ms: float
    duration_ms: float
    samples: List[float] = field(default_factory=list)
    concealed: bool = False       # 是否为 PLC 合成帧
    comfort_noise: bool = False   # 是否为舒适噪声兜底帧


@dataclass
class DecodedFrame:
    seq: int
    samples: List[float]
    sample_rate: int
    duration_ms: float
    flags: Tuple[str, ...] = ()


class AtomicIndex:
    """带锁的单调索引。

    收包线程只推进写指针，解码线程只推进读指针；每次更新在锁内完成，
    外部读到的永远是一个完整的整数，不会出现更新到一半的撕裂索引。
    """

    def __init__(self, initial: int = 0) -> None:
        self._lock = threading.Lock()
        self._value = int(initial)

    def get(self) -> int:
        with self._lock:
            return self._value

    def advance(self, delta: int = 1) -> int:
        with self._lock:
            self._value += int(delta)
            return self._value

    def compare_and_set(self, expected: int, new: int) -> bool:
        with self._lock:
            if self._value == expected:
                self._value = int(new)
                return True
            return False

    def __int__(self) -> int:
        return self.get()


class HysteresisController:
    """目标延迟滞回控制器（Schmitt 触发 + 驻留计数）。

    深度低于 ``target - margin`` 且连续 ``dwell`` 次才上调目标；
    深度高于 ``target + margin`` 且连续 ``dwell`` 次才下调目标。
    上调/下调阈值不对称，网络延迟在目标值附近抖动时目标不会来回翻。
    """

    def __init__(
        self,
        nominal: float = NOMINAL_TARGET_MS,
        min_ms: float = MIN_TARGET_MS,
        max_ms: float = MAX_TARGET_MS,
        step: float = HYSTERESIS_STEP_MS,
        margin: float = HYSTERESIS_MARGIN_MS,
        dwell: int = DWELL_READINGS,
    ) -> None:
        self.nominal = float(nominal)
        self.min_ms = float(min_ms)
        self.max_ms = float(max_ms)
        self.step = float(step)
        self.margin = float(margin)
        self.dwell = int(dwell)
        self.target = float(nominal)
        self._low_count = 0
        self._high_count = 0
        self._lock = threading.Lock()

    def update(self, depth_ms: float) -> float:
        depth_ms = float(depth_ms)
        with self._lock:
            low_watermark = self.target - self.margin
            high_watermark = self.target + self.margin
            if depth_ms < low_watermark:
                self._low_count += 1
                self._high_count = 0
                if self._low_count >= self.dwell and self.target < self.max_ms:
                    self.target = min(self.max_ms, self.target + self.step)
                    self._low_count = 0
            elif depth_ms > high_watermark:
                self._high_count += 1
                self._low_count = 0
                if self._high_count >= self.dwell and self.target > self.min_ms:
                    self.target = max(self.min_ms, self.target - self.step)
                    self._high_count = 0
            else:
                # 落在滞回带内：清零驻留计数，目标保持不动
                self._low_count = 0
                self._high_count = 0
            return self.target

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "target_ms": self.target,
                "low_watermark_ms": self.target - self.margin,
                "high_watermark_ms": self.target + self.margin,
                "low_count": self._low_count,
                "high_count": self._high_count,
            }


class JitterBuffer:
    """按时间计深、帧对齐、指针原子的环形抖动缓冲。"""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        start_seq: int = 0,
        nominal_ms: float = NOMINAL_TARGET_MS,
        min_ms: float = MIN_TARGET_MS,
        max_ms: float = MAX_TARGET_MS,
        capacity_ms: float = CAPACITY_MS,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.capacity_ms = float(capacity_ms)
        self._lock = threading.RLock()
        self._frames: Deque[Packet] = deque()
        self._by_seq: Dict[int, Packet] = {}
        # 单调指针：读指针=下一个待出队序号，写指针=累计入槽数
        self.read_index = AtomicIndex(start_seq)
        self.write_index = AtomicIndex(0)
        self.controller = HysteresisController(
            nominal=nominal_ms, min_ms=min_ms, max_ms=max_ms
        )
        # PLC 状态
        self.phase = 0
        self.period: Optional[int] = None
        self.frame_ms = DEFAULT_FRAME_MS
        self.last_good: List[float] = []
        self.loss_run = 0
        self.pending_tail: List[float] = []
        self.stats: Dict[str, int] = {
            "stored": 0,
            "duplicates": 0,
            "late": 0,
            "rejected": 0,
            "dropped": 0,
            "popped": 0,
            "underruns": 0,
            "concealed": 0,
            "fallback": 0,
            "crossfades": 0,
        }

    # -- 内部工具（调用方需持锁） -------------------------------------------

    def _depth_ms_locked(self) -> float:
        return float(sum(f.duration_ms for f in self._frames))

    def pointers(self) -> Dict[str, Any]:
        """在同一把锁内取一对一致的指针快照，避免读到跨更新的组合。"""
        with self._lock:
            return {
                "read": self.read_index.get(),
                "write": self.write_index.get(),
                "queued_frames": len(self._frames),
                "depth_ms": self._depth_ms_locked(),
            }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def make_packet(
    seq: int,
    duration_ms: float = DEFAULT_FRAME_MS,
    sample_rate: int = SAMPLE_RATE,
    freq: float = 440.0,
) -> Packet:
    """生成一帧相位连续的确定性正弦 PCM（测试/兜底用途的模拟编码器）。"""
    n = int(round(duration_ms * sample_rate / 1000.0))
    start_sample = int(round(seq * duration_ms * sample_rate / 1000.0))
    samples = [
        math.sin(2.0 * math.pi * freq * (start_sample + i) / sample_rate)
        for i in range(n)
    ]
    return Packet(
        seq=int(seq),
        ts_ms=seq * duration_ms,
        duration_ms=float(duration_ms),
        samples=samples,
    )


def _frame_sample_count(duration_ms: float, sample_rate: int) -> int:
    return int(round(duration_ms * sample_rate / 1000.0))


def _is_full_frame(packet: Packet, sample_rate: int) -> bool:
    """整帧校验：样本数必须恰好等于帧时长，半帧一律拒绝。"""
    return len(packet.samples) == _frame_sample_count(
        packet.duration_ms, sample_rate
    )


def _as_packet(obj: Any, sample_rate: int = SAMPLE_RATE) -> Packet:
    if isinstance(obj, Packet):
        return obj
    if isinstance(obj, dict):
        return Packet(
            seq=int(obj["seq"]),
            ts_ms=float(obj.get("ts_ms", 0.0)),
            duration_ms=float(obj["duration_ms"]),
            samples=list(obj.get("samples", [])),
            concealed=bool(obj.get("concealed", False)),
            comfort_noise=bool(obj.get("comfort_noise", False)),
        )
    if isinstance(obj, int) and not isinstance(obj, bool):
        return make_packet(obj, sample_rate=sample_rate)
    raise TypeError(f"无法把 {type(obj)!r} 解释为 Packet")


def _as_buffer(obj: Any) -> JitterBuffer:
    if isinstance(obj, JitterBuffer):
        return obj
    return JitterBuffer()


def _collect_packets(obj: Any) -> List[Packet]:
    if isinstance(obj, JitterBuffer):
        with obj._lock:
            return list(obj._frames)
    if isinstance(obj, (list, tuple)):
        return [_as_packet(item) for item in obj]
    return []


def estimate_pitch_period(
    samples: Sequence[float], sample_rate: int = SAMPLE_RATE
) -> Optional[int]:
    """归一化自相关基音估计；找不到可靠周期时返回 None（调用方退化为整帧重复）。

    取第一个相关度超过阈值的滞后，避免高次谐波和整帧自相关造成周期误判。
    """
    n = len(samples)
    min_lag = max(16, sample_rate // 300)
    max_lag = min(int(n * 0.75), max(min_lag, sample_rate // 80))
    if max_lag <= min_lag:
        return None
    for lag in range(min_lag, max_lag + 1):
        m = n - lag
        energy_a = 0.0
        energy_b = 0.0
        cross = 0.0
        for i in range(m):
            a = samples[i]
            b = samples[i + lag]
            energy_a += a * a
            energy_b += b * b
            cross += a * b
        denom = math.sqrt(energy_a * energy_b)
        if denom <= 1e-12:
            continue
        score = cross / denom
        if score > 0.85:
            return lag
    return None


# ---------------------------------------------------------------------------
# 修复点 5：指针原子更新
# ---------------------------------------------------------------------------


def atomic_index(index: Any, delta: int = 1) -> int:
    """原子推进索引并返回提交后的新值。

    - ``AtomicIndex``：在其内部锁内推进；
    - 裸整数：在模块级锁内完成加法（便于无状态调用与测试）。
    """
    if isinstance(index, AtomicIndex):
        return index.advance(delta)
    with _GLOBAL_INDEX_LOCK:
        return int(index) + int(delta)


# ---------------------------------------------------------------------------
# 修复点 1：目标延迟滞回
# ---------------------------------------------------------------------------


def target_hysteresis(
    depth_ms: Any,
    state: Any = None,
    *,
    nominal: float = NOMINAL_TARGET_MS,
    min_ms: float = MIN_TARGET_MS,
    max_ms: float = MAX_TARGET_MS,
    step: float = HYSTERESIS_STEP_MS,
    margin: float = HYSTERESIS_MARGIN_MS,
    dwell: int = DWELL_READINGS,
) -> float:
    """根据当前缓冲深度（毫秒）更新目标延迟，返回新目标（毫秒）。

    ``state`` 可以是 ``HysteresisController`` 或持久化 dict；不传则使用
    一次性控制器。滞回带 + 驻留计数保证深度在目标附近抖动时目标不翻跳。
    """
    if isinstance(state, HysteresisController):
        controller = state
    elif isinstance(state, dict):
        controller = state.get("controller")
        if controller is None:
            controller = HysteresisController(
                nominal=nominal, min_ms=min_ms, max_ms=max_ms,
                step=step, margin=margin, dwell=dwell,
            )
            state["controller"] = controller
    else:
        controller = HysteresisController(
            nominal=nominal, min_ms=min_ms, max_ms=max_ms,
            step=step, margin=margin, dwell=dwell,
        )
    return controller.update(float(depth_ms))


# ---------------------------------------------------------------------------
# 修复点 4：缓冲深度按时间（毫秒）计算，而不是按包数
# ---------------------------------------------------------------------------


def depth_by_time(buffer_or_packets: Any) -> float:
    """返回队列中所有完整帧的时长之和（毫秒）。

    20ms 帧与 60ms 帧即使包数相同，时间深度也差 3 倍——目标延迟必须与
    这个值比较，而不是与包数比较。
    """
    packets = _collect_packets(buffer_or_packets)
    return float(sum(p.duration_ms for p in packets))


# ---------------------------------------------------------------------------
# 修复点 6：欠载淡入（同时提供淡出与双向淡化，消除拼接点咔哒声）
# ---------------------------------------------------------------------------


def fade_on_underrun(
    samples: Any,
    fade_ms: float = FADE_MS,
    sample_rate: int = SAMPLE_RATE,
    direction: str = "in",
) -> List[float]:
    """对样本施加短淡入/淡出斜坡。

    欠载后第一批替代样本从 0 增益起步，避免与前段在拼接点形成阶跃；
    ``direction`` 取 ``"in"`` / ``"out"`` / ``"inout"``。
    裸整数入参被视为样本点数，对单位信号做淡入（无状态调用/测试）。
    """
    if isinstance(samples, int) and not isinstance(samples, bool):
        data = [1.0] * samples
    else:
        data = [float(x) for x in samples]
    if not data:
        return []
    if direction not in ("in", "out", "inout"):
        raise ValueError(f"未知 direction: {direction!r}")
    k = min(len(data), _fade_sample_count(sample_rate, fade_ms))

    if direction in ("in", "inout"):
        # 首样本严格为 0 增益，线性爬到 1.0，保证拼接点无阶跃
        for i in range(k):
            data[i] *= i / k
    if direction in ("out", "inout"):
        for i in range(k):
            # 末样本严格为 0 增益
            idx = len(data) - 1 - i
            data[idx] *= i / k
    return data


# ---------------------------------------------------------------------------
# 修复点 2：丢包保相位
# ---------------------------------------------------------------------------


def phase_continuous(
    previous: Any = None,
    needed: Optional[int] = None,
    sample_rate: int = SAMPLE_RATE,
    phase: int = 0,
    period: Optional[int] = None,
) -> Tuple[List[float], int]:
    """以相位连续方式周期延拓历史样本，合成 ``needed`` 个替代样本。

    返回 ``(samples, new_phase)``。合成信号在周期回绕处取值相同（自身
    连续），并通过 ``phase`` 累加器让多帧连续丢包时跨帧无缝衔接；
    恢复真实帧后再由交叉淡化抹平残余接缝。
    """
    if isinstance(previous, int) and not isinstance(previous, bool) and needed is None:
        needed = previous
        previous = None
    if previous is None:
        prev = _default_template(sample_rate)
    else:
        prev = [float(x) for x in previous]
    if not prev:
        prev = _default_template(sample_rate)
    if needed is None:
        needed = len(prev)
    if period is None or period <= 0:
        period = estimate_pitch_period(prev, sample_rate) or len(prev)
    period = min(int(period), len(prev))
    phase = int(phase) % period
    out = [prev[(phase + i) % period] for i in range(int(needed))]
    new_phase = (phase + int(needed)) % period
    return out, new_phase


def crossfade_junction(
    concealed_tail: Sequence[float],
    real_head: Sequence[float],
    sample_rate: int = SAMPLE_RATE,
    fade_ms: float = FADE_MS,
) -> List[float]:
    """真实帧回归时，在其头部与上一段合成帧尾部之间做线性交叉淡化。"""
    head = [float(x) for x in real_head]
    if not concealed_tail or not head:
        return head
    n = min(
        len(head),
        len(concealed_tail),
        _fade_sample_count(sample_rate, fade_ms),
    )
    tail_start = len(concealed_tail) - n
    for i in range(n):
        g = i / n  # 真实帧权重从 0 爬到 1：接缝首样本完全等于合成段尾部
        head[i] = concealed_tail[tail_start + i] * (1.0 - g) + head[i] * g
    return head


# ---------------------------------------------------------------------------
# 修复点 3：按帧边界丢弃
# ---------------------------------------------------------------------------


def _drop_locked(buffer: JitterBuffer, target_ms: float) -> List[Packet]:
    """持锁丢弃最旧的整帧，直到时间深度不超过目标。永不切分帧。"""
    dropped: List[Packet] = []
    while buffer._frames and buffer._depth_ms_locked() > target_ms:
        frame = buffer._frames.popleft()
        buffer._by_seq.pop(frame.seq, None)
        # 帧被整体摘除：出队到该序号时按序号缺口走 PLC，不产生半帧
        dropped.append(frame)
        buffer.stats["dropped"] += 1
    return dropped


def drop_on_frame_boundary(buffer: Any, target_ms: Optional[float] = None) -> List[Packet]:
    """溢出时只按整帧丢弃最旧包，返回被丢弃的帧列表。

    - 每次丢弃都是一个完整 ``Packet``，解码器永远不会从半个帧开始解；
    - 被丢帧标记为丢失，出队到该序号时走 PLC 而不是吐出残帧；
    - 混帧长场景下可能比目标多欠一个最长帧，这是帧对齐的固有代价。
    无缓冲区的裸整数入参返回空列表。
    """
    if not isinstance(buffer, JitterBuffer):
        if isinstance(buffer, list):
            # 无状态列表形式：按目标时长从最旧端整帧移除
            if target_ms is None:
                return []
            packets = buffer
            total = float(sum(p.duration_ms for p in packets))
            dropped: List[Packet] = []
            while packets and total > target_ms:
                frame = packets.pop(0)
                total -= frame.duration_ms
                dropped.append(frame)
            return dropped
        return []
    with buffer._lock:
        target = (
            buffer.controller.target if target_ms is None else float(target_ms)
        )
        return _drop_locked(buffer, target)


# ---------------------------------------------------------------------------
# 流水线：收包 / 出队 / 隐藏 / 兜底 / 解码
# ---------------------------------------------------------------------------


def _status(status: str, buffer: JitterBuffer, packet: Packet,
            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = {
        "status": status,
        "seq": packet.seq,
        "depth_ms": buffer._depth_ms_locked(),
        "target_ms": buffer.controller.target,
    }
    if extra:
        result.update(extra)
    return result


def push(buffer: Any = None, packet: Any = None) -> Dict[str, Any]:
    """收包入队（线程安全）。

    整帧校验 → 去重/迟到判定 → 序号有序插入 → 写指针原子推进 →
    超硬容量时按帧边界丢弃。裸整数入参会被当作序号合成一帧（无状态调用）。
    """
    buf = _as_buffer(buffer)
    if packet is None:
        seq = buffer if isinstance(buffer, int) and not isinstance(buffer, bool) else buf.read_index.get()
        packet = make_packet(seq, sample_rate=buf.sample_rate)
    pkt = _as_packet(packet, buf.sample_rate)

    with buf._lock:
        if not _is_full_frame(pkt, buf.sample_rate):
            buf.stats["rejected"] += 1
            return _status("rejected", buf, pkt, {
                "reason": "partial frame: 半帧禁止入队",
                "samples": len(pkt.samples),
                "expected": _frame_sample_count(pkt.duration_ms, buf.sample_rate),
            })
        if pkt.seq in buf._by_seq:
            buf.stats["duplicates"] += 1
            return _status("duplicate", buf, pkt)
        if pkt.seq < buf.read_index.get():
            buf.stats["late"] += 1
            return _status("late", buf, pkt)

        inserted = False
        for i, existing in enumerate(buf._frames):
            if existing.seq > pkt.seq:
                buf._frames.insert(i, pkt)
                inserted = True
                break
        if not inserted:
            buf._frames.append(pkt)
        buf._by_seq[pkt.seq] = pkt
        buf.write_index.advance(1)  # 原子推进写指针
        buf.stats["stored"] += 1
        buf.frame_ms = pkt.duration_ms

        dropped: List[Packet] = []
        if buf._depth_ms_locked() > buf.capacity_ms:
            # 硬容量溢出：按帧边界整帧丢弃到硬容量线；目标线由滞回控制器
            # 主动调用 drop_on_frame_boundary 时使用
            dropped = _drop_locked(buf, buf.capacity_ms)
        return _status("stored", buf, pkt, {
            "dropped_seqs": [f.seq for f in dropped],
        })


def _conceal_packet(buffer: JitterBuffer, seq: int, duration_ms: float) -> Packet:
    """PLC 合成帧：周期延拓 + 衰减 + 突发首帧淡入。"""
    n = _frame_sample_count(duration_ms, buffer.sample_rate)
    template = buffer.last_good or _default_template(buffer.sample_rate)
    synth, buffer.phase = phase_continuous(
        template,
        n,
        sample_rate=buffer.sample_rate,
        phase=buffer.phase,
        period=buffer.period,
    )
    # 连续丢包越多能量越低，防止重复帧固化成机器音
    gain = max(0.5, 0.9 ** buffer.loss_run)
    synth = [s * gain for s in synth]
    if buffer.loss_run == 1:
        # 欠载淡入：替代帧从 0 增益起步，消除拼接点咔哒声
        synth = fade_on_underrun(synth, sample_rate=buffer.sample_rate)
    tail_n = min(len(synth), _fade_sample_count(buffer.sample_rate))
    buffer.pending_tail = synth[-tail_n:]
    return Packet(
        seq=seq,
        ts_ms=seq * duration_ms,
        duration_ms=duration_ms,
        samples=synth,
        concealed=True,
    )


def _comfort_packet(buffer: JitterBuffer, seq: int, duration_ms: float) -> Packet:
    """长连丢兜底：确定性舒适噪声，两端淡化。"""
    n = _frame_sample_count(duration_ms, buffer.sample_rate)
    seed = hashlib.sha256(
        f"comfort:{int(seq)}:{buffer.sample_rate}".encode("utf-8")
    ).digest()
    rng = random.Random(seed)
    samples = [
        rng.uniform(-COMFORT_NOISE_LEVEL, COMFORT_NOISE_LEVEL) for _ in range(n)
    ]
    samples = fade_on_underrun(
        samples, sample_rate=buffer.sample_rate, direction="inout"
    )
    buffer.pending_tail = samples[-min(n, _fade_sample_count(buffer.sample_rate)):]
    return Packet(
        seq=seq,
        ts_ms=seq * duration_ms,
        duration_ms=duration_ms,
        samples=samples,
        concealed=True,
        comfort_noise=True,
    )


def pop(buffer: Any = None) -> Packet:
    """按解码节拍出队一帧（线程安全）。

    命中则返回真实整帧（丢包恢复时先做交叉淡化保相位）；未命中即欠载，
    读指针仍原子推进一个序号，由 PLC（短时）或舒适噪声（长连丢）补帧。
    """
    buf = _as_buffer(buffer)
    with buf._lock:
        expected = buf.read_index.get()
        frame = buf._by_seq.pop(expected, None)

        if frame is not None:
            try:
                buf._frames.remove(frame)
            except ValueError:
                pass
            buf.read_index.advance(1)  # 原子推进读指针
            buf.stats["popped"] += 1
            buf.last_good = list(frame.samples)
            buf.period = estimate_pitch_period(buf.last_good, buf.sample_rate)
            if buf.loss_run > 0:
                # 真实帧回归：与合成段尾部交叉淡化，保相位、去咔哒声
                frame.samples = crossfade_junction(
                    buf.pending_tail, frame.samples, buf.sample_rate
                )
                buf.stats["crossfades"] += 1
            buf.loss_run = 0
            buf.phase = 0
            buf.controller.update(buf._depth_ms_locked())
            return frame

        # 欠载：缓冲中没有 expected 序号的帧
        buf.loss_run += 1
        buf.stats["underruns"] += 1
        if buf.loss_run > MAX_CONCEAL_FRAMES:
            pkt = _comfort_packet(buf, expected, buf.frame_ms)
            buf.stats["fallback"] += 1
        else:
            pkt = _conceal_packet(buf, expected, buf.frame_ms)
            buf.stats["concealed"] += 1
        buf.read_index.advance(1)  # 解码时钟不停摆
        buf.controller.update(buf._depth_ms_locked())
        return pkt


def conceal(
    source: Any = None,
    duration_ms: float = DEFAULT_FRAME_MS,
    sample_rate: int = SAMPLE_RATE,
) -> Packet:
    """无状态/有状态丢包隐藏入口，返回一帧合成 ``Packet``。

    裸整数入参视为需要合成的时长（毫秒）。
    """
    if isinstance(source, int) and not isinstance(source, bool):
        duration_ms = float(source)
        source = None
    if isinstance(source, JitterBuffer):
        with source._lock:
            seq = source.read_index.get()
            source.loss_run += 1
            source.stats["underruns"] += 1
            pkt = _conceal_packet(source, seq, source.frame_ms)
            source.read_index.advance(1)
            source.stats["concealed"] += 1
            return pkt
    n = _frame_sample_count(duration_ms, sample_rate)
    synth, _ = phase_continuous(
        _default_template(sample_rate), n, sample_rate=sample_rate
    )
    synth = fade_on_underrun(synth, sample_rate=sample_rate)
    return Packet(
        seq=0,
        ts_ms=0.0,
        duration_ms=float(duration_ms),
        samples=synth,
        concealed=True,
    )


def fallback(
    source: Any = None,
    duration_ms: float = DEFAULT_FRAME_MS,
    sample_rate: int = SAMPLE_RATE,
) -> Packet:
    """舒适噪声兜底帧（确定性随机、两端淡化），裸整数入参视为时长毫秒。"""
    if isinstance(source, int) and not isinstance(source, bool):
        duration_ms = float(source)
        source = None
    if isinstance(source, JitterBuffer):
        sample_rate = source.sample_rate
    n = _frame_sample_count(duration_ms, sample_rate)
    seed = hashlib.sha256(
        f"comfort:{int(round(duration_ms))}:{sample_rate}".encode("utf-8")
    ).digest()
    rng = random.Random(seed)
    samples = [
        rng.uniform(-COMFORT_NOISE_LEVEL, COMFORT_NOISE_LEVEL) for _ in range(n)
    ]
    samples = fade_on_underrun(
        samples, sample_rate=sample_rate, direction="inout"
    )
    return Packet(
        seq=getattr(source, "read_index", AtomicIndex(0)).get(),
        ts_ms=0.0,
        duration_ms=float(duration_ms),
        samples=samples,
        concealed=True,
        comfort_noise=True,
    )


def decode(frame: Any = None, sample_rate: int = SAMPLE_RATE) -> DecodedFrame:
    """解码一帧。

    只接受样本数与帧时长严格相等的完整帧；遇到半帧直接抛 ``ValueError``，
    从根上杜绝“从半个帧开始解、噪声底突变”。裸整数入参会合成对应序号的
    标准测试帧后解码。
    """
    if frame is None:
        frame = make_packet(0, sample_rate=sample_rate)
    elif isinstance(frame, int) and not isinstance(frame, bool):
        frame = make_packet(frame, sample_rate=sample_rate)
    pkt = _as_packet(frame, sample_rate)
    expected_n = _frame_sample_count(pkt.duration_ms, sample_rate)
    if len(pkt.samples) != expected_n:
        raise ValueError(
            f"seq={pkt.seq} 是半帧/残帧: {len(pkt.samples)}/{expected_n} 样本，拒绝解码"
        )
    flags: List[str] = []
    if pkt.concealed:
        flags.append("concealed")
    if pkt.comfort_noise:
        flags.append("comfort_noise")
    return DecodedFrame(
        seq=pkt.seq,
        samples=list(pkt.samples),
        sample_rate=sample_rate,
        duration_ms=pkt.duration_ms,
        flags=tuple(flags),
    )


# ---------------------------------------------------------------------------
# 校验 / 指标 / 审计 / 确定性
# ---------------------------------------------------------------------------


def validate_buffer(buffer: Any = None) -> Dict[str, Any]:
    """检查缓冲不变量，返回 ``{"ok": bool, "issues": [...], ...}``。"""
    buf = _as_buffer(buffer)
    issues: List[str] = []
    with buf._lock:
        seqs = [f.seq for f in buf._frames]
        if len(seqs) != len(set(seqs)):
            issues.append("队列中存在重复序号")
        if seqs != sorted(seqs):
            issues.append("队列未按序号有序")
        if set(seqs) != set(buf._by_seq.keys()):
            issues.append("deque 与 by_seq 索引不一致")
        for f in buf._frames:
            if not _is_full_frame(f, buf.sample_rate):
                issues.append(f"seq={f.seq} 不是完整帧（半帧混入队列）")
        depth = buf._depth_ms_locked()
        if depth > buf.capacity_ms + 1e-9:
            issues.append(
                f"时间深度 {depth:.1f}ms 超过硬容量 {buf.capacity_ms:.1f}ms"
            )
        if buf.write_index.get() != buf.stats["stored"]:
            issues.append("写指针与累计入队计数不一致")
        if buf.read_index.get() < 0:
            issues.append("读指针倒退")
        target_info = buf.controller.snapshot()
        return {
            "ok": not issues,
            "issues": issues,
            "depth_ms": depth,
            "target_ms": buf.controller.target,
            "pointers": buf.pointers(),
            "hysteresis": target_info,
        }


def metrics(buffer: Any = None) -> Dict[str, Any]:
    """返回运行指标：计数全部基于时间深度与整帧事件。"""
    buf = _as_buffer(buffer)
    with buf._lock:
        stats = dict(buf.stats)
        total_outputs = stats["popped"] + stats["concealed"] + stats["fallback"]
        loss_rate = (
            (stats["underruns"] / total_outputs) if total_outputs else 0.0
        )
        return {
            "stats": stats,
            "depth_ms": buf._depth_ms_locked(),
            "depth_by_packets": len(buf._frames),
            "target_ms": buf.controller.target,
            "loss_run": buf.loss_run,
            "loss_rate": loss_rate,
            "pointers": buf.pointers(),
        }


def audit(buffer: Any = None) -> Dict[str, Any]:
    """校验 + 指标 + 处置建议的一站式审计报告。"""
    buf = _as_buffer(buffer)
    report = validate_buffer(buf)
    m = metrics(buf)
    recommendations: List[str] = []
    stats = m["stats"]
    total_outputs = stats["popped"] + stats["concealed"] + stats["fallback"]
    if total_outputs:
        underrun_ratio = stats["underruns"] / total_outputs
        drop_ratio = stats["dropped"] / max(1, stats["stored"])
        if underrun_ratio > 0.1:
            recommendations.append("欠载率偏高：上调标称目标延迟或扩大滞回带")
        if drop_ratio > 0.1:
            recommendations.append("丢帧率偏高：下调标称目标延迟或增大硬容量")
        if stats["fallback"]:
            recommendations.append("出现长连丢舒适噪声兜底：检查网络链路质量")
    if not report["ok"]:
        recommendations.append("缓冲不变量被破坏，参见 issues")
    report["metrics"] = m
    report["recommendations"] = recommendations
    report["health"] = "ok" if report["ok"] and not recommendations else "degraded"
    return report


def determinism(
    seed: Any = 1234,
    frames: int = 24,
    lost_seqs: Sequence[int] = (4, 5, 10),
) -> Dict[str, Any]:
    """同一脚本化场景跑两遍，逐样本比对，返回确定性报告。"""
    if isinstance(seed, int) and not isinstance(seed, bool):
        rng_seed = seed
    else:
        rng_seed = 1234
    lost = set(int(s) for s in lost_seqs)

    def run() -> Dict[str, Any]:
        random.Random(rng_seed).random()  # 预留随机源位，当前场景完全脚本化
        buf = JitterBuffer()
        outputs: List[Tuple[int, Tuple[str, ...], Tuple[float, ...]]] = []
        for seq in range(frames):
            if seq not in lost:
                push(buf, make_packet(seq, sample_rate=buf.sample_rate))
        for _ in range(frames):
            decoded = decode(pop(buf), buf.sample_rate)
            outputs.append(
                (
                    decoded.seq,
                    decoded.flags,
                    tuple(round(x, 9) for x in decoded.samples),
                )
            )
        return {"outputs": outputs, "metrics": metrics(buf)}

    first = run()
    second = run()
    identical = first == second
    signature = hashlib.sha256(repr(first).encode("utf-8")).hexdigest()
    return {
        "deterministic": identical,
        "frames": frames,
        "lost_seqs": sorted(lost),
        "signature": signature,
    }
