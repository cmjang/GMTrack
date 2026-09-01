# RGMT 跌倒恢复复现

本文件只记录 RGMT `arXiv:2601.23080v1` Sec. II-D 的恢复协议，以及本仓库为了把“随机不稳定姿态”落到运行时所作的本地假设。

严格任务默认 `recovery_probability = 0`。只有专门的
`GMTrack-Stage1-Recovery-Flat-Unitree-G1` 才开启这个机制。

## 论文契约

RGMT Sec. II-D 的恢复机制可以压缩成四条：

| 项目 | 约束 |
|---|---|
| reset 比例 | 每次 reset 以 `0.15` 概率把该环境设为 recovery env |
| 初始化 | recovery env 重置为随机不稳定姿态 |
| 参考 | 保持普通 motion reference，reference 继续按原时间线前进 |
| 辅助力 | 对 anchor body 施加向上的 `U[0,200] N` 助力，并在训练中退火到可忽略 |
| 终止屏蔽 | recovery env 在 `3 s` 窗口内屏蔽失稳终止；窗口结束后恢复标准终止 |

## 明确排除

这个实现：

- 不使用 `fallAndGetUp` / get-up 演示
- 不使用 AMP 或 MJLab 的 recovery 数据
- 不把 reference 重定向到任何专门的起身片段

## 未发表的本地假设

RGMT 只说“randomized unstable poses”，没有公开具体采样器。下面这些值只是当前实现里把这句话实例化的运行时假设，不是论文参数。

| 项目 | 本地假设 |
|---|---|
| 根高度 | `0.35 m` 到 `0.65 m` 之间均匀采样 |
| 躯干倾角 | 绕随机水平轴取 `pi/3` 到 `2pi/3` 的倾斜，再加随机 yaw |
| 关节姿态 | 使用 reference joint pose，加 `±0.25 rad` 抖动，并裁剪到 soft limits |
| 速度 | 根速度和关节速度都置零 |
| 参考时钟 | reference 继续前进，不回跳到任何起身示范 |
| 参考边界 | 若采样点离普通片段结尾不足 3 s，只把起点前移到能容纳完整恢复窗口的位置 |
| 助力退火 | 2.4M env steps 内把助力退火到近零 |

## 运行备注

- recovery 不是严格任务的一部分，严格任务仍然保持 `recovery_probability = 0`
- 这个恢复任务用于复现 RGMT Sec. II-D；未公开的采样细节仍属于本地假设
- `webplay` 的 `Recovery Test` 面板可在普通动作播放中向选中机器人施加
  `±X/±Y` 水平速度冲击；`--random-recovery-start` 则单独检查无助力随机姿态恢复
- 如果以后换采样器，优先更新上面的“未发表的本地假设”，不要把它们误写成论文值
