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
