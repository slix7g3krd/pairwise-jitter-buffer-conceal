# pairwise-jitter-buffer-conceal

## 领域

游戏、图形与多媒体

## 主题

音频抖动缓冲与丢包隐藏

## 说明

延迟滞回、相位连续与定时深。

初始状态下 `media/jitter_buffer_conceal.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 修复点与实现对照

| 缺陷 | 修复 | 入口 |
| --- | --- | --- |
| 固定目标延迟，抖动时反复欠载/溢出 | 目标延迟滞回：Schmitt 双水位 + 驻留计数，带内不调、连续越界才按步长调整并钳位 | `target_hysteresis` / `HysteresisController` |
| 欠载重复上一帧，相位不连续产生咔哒声 | 丢包保相位：自相关基音估计 + 周期延拓，`phase` 累加器跨帧衔接；真实帧回归时交叉淡化 | `phase_continuous` / `crossfade_junction` |
| 溢出丢最旧包落在帧中间，半帧解码噪声突变 | 按帧边界丢：只摘除完整 `Packet`，半帧在入队和解码两处都被拒绝 | `drop_on_frame_boundary` / `decode` |
| 深度按包数算，20ms 与 60ms 帧实际缓冲差 3 倍 | 深度按时间：队列时长求和（毫秒），目标延迟只与毫秒值比较 | `depth_by_time` |
| 收发线程并发改指针，读到撕裂索引 | 指针原子更新：锁内单调推进，指针快照同锁读取 | `atomic_index` / `AtomicIndex` |
| 欠载替代帧拼接点阶跃 | 欠载淡入：替代帧从 0 增益线性起步，长连丢舒适噪声两端淡化 | `fade_on_underrun` |

流水线由 `push`（整帧校验→去重/迟到→有序插入→硬容量帧对齐丢弃）、
`pop`（命中出队/欠载 PLC，读指针始终原子推进）、`conceal`/`fallback`
（周期延拓隐藏/舒适噪声兜底）、`decode`（整帧校验解码）组成；
`validate_buffer`/`metrics`/`audit` 提供不变量校验、指标与处置建议，
`determinism` 用双跑逐样本比对保证可复现。
