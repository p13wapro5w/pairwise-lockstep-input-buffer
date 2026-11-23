# pairwise-lockstep-input-buffer

## 领域

游戏、图形与多媒体

## 主题

帧同步输入缓冲

## 说明

服务器帧号排队、回滚快照与重连去重。

`gameplay/lockstep_input_buffer.py` 实现了帧同步客户端的输入缓冲与回滚基础设施。
每个公开入口都返回一个可检查的结果对象，而不是 `None`。

```bash
python3 -m pytest -q
```

## 修复的六类问题

| 问题 | 入口 | 做法 |
| --- | --- | --- |
| 输入按本地帧进缓冲，抖动后顺序错乱 | `FrameQueue` / `frame_queue` / `input_order` | 只按 `server_frame` 排序与消费；同一 `(player, frame)` 去重；低于消费水位线的输入判为 stale。 |
| 追帧/补偿无上限，越追越慢 | `catchup_cap` / `compensation_cap` / `CatchupController` / `fallback` | 每 tick 的追帧步数、一次回滚的重放帧数都有硬上限；超过上限返回 `resync_required`，请求权威快照。 |
| 回滚时随机数状态没回退 | `rng_snapshot` / `rng_restore` / `Snapshot` / `RollbackBuffer` | 快照同时保存模拟状态、逻辑帧与 RNG 状态；恢复时校验摘要，状态与随机数一起回滚。 |
| 技能冷却用本地帧号，两端差一帧 | `logical_frame` / `Cooldown` / `CooldownTracker` | 冷却一律基于 `local_frame - input_delay`（并可由服务器确认帧钳制）；`validation` 会报告 `frame_basis="render"` 的冷却。 |
| 重连补帧把旧输入重复灌入队列 | `InputDeduplicator` / `reconnect_dedupe` / `watermark` / `audit` | 每个玩家维护已处理序号水位线；帧号低于水位线或序号不大于水位线的补发输入全部丢弃。 |
| 回滚后表现层继续外推旧时间线 | `PresentationBuffer` / `present_rebuild` | 丢弃回滚点之后的所有插值/特效样本，以权威快照为锚点重新开始插值。 |

## 端到端示例

```python
import random
from gameplay.lockstep_input_buffer import (
    FrameQueue, FrameInput, RollbackBuffer, PresentationBuffer,
    InputDeduplicator, logical_frame,
)

rng = random.Random(1)
queue = FrameQueue(start_frame=0, players=("a", "b"))
queue.push(FrameInput("a", 0, 1, ("move",)))
queue.push(FrameInput("b", 0, 1, ("move",)))
assert queue.pop_frame(0) is not None          # 一帧完整后才消费

snapshots = RollbackBuffer(max_compensation=8)
snapshots.capture(0, {"score": 0}, rng)        # 状态 + 逻辑帧 + RNG
result = snapshots.rollback(0, rng=rng)        # RNG 一并回退
assert result.success
```
