# pairwise-lockstep-input-buffer

## 领域

游戏、图形与多媒体

## 主题

帧同步输入缓冲

## 说明

服务器帧号排队、回滚快照与重连去重。

初始状态下 `gameplay/lockstep_input_buffer.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
