# HumanoidBench 任务总结

HumanoidBench 是一个基于物理的人形机器人控制任务套件，提供了 27 个全身操作和运动控制任务，涵盖了从简单到复杂的各种人形机器人控制场景。

## 📋 快速参考：本项目基准任务

| 任务 | 命令 | 类别 | 难度 |
|------|------|------|------|
| **Stand** | `humanoid_h1hand-stand-v0` | 基础运动 | 🟢 |
| **Walk** | `humanoid_h1hand-walk-v0` | 基础运动 | 🟡 |
| **Run** | `humanoid_h1hand-run-v0` | 基础运动 | 🟡 |
| **Sit** | `humanoid_h1hand-sit_simple-v0` | 复杂运动 | 🟡 |
| **Slide** | `humanoid_h1hand-slide-v0` | 复杂运动 | 🔴 |
| **Pole** | `humanoid_h1hand-pole-v0` | 复杂运动 | 🔴 |
| **Hurdle** | `humanoid_h1hand-hurdle-v0` | 复杂运动 | 🔴 |

## 任务概览

- **总任务数**: 27 个主要任务
- **机器人类型**: H1（有手/无手）、G1、H1Simple、H1Touch
- **观察类型**: state（状态）、rgb（图像）、proprio（本体感觉）、tactile（触觉）
- **环境类型**: humanoid
- **任务类型**: 运动控制、操作、平衡

## 🎯 本项目基准任务

本文档中用于对比评估的 7 个核心运动控制任务：

### 基础运动任务
| 任务 | MFP 配置 | 描述 | 难度 |
|------|----------|------|------|
| **Stand** | `humanoid_h1hand-stand-v0` | 人形机器人保持站立姿态 | 🟢 入门 |
| **Walk** | `humanoid_h1hand-walk-v0` | 人形机器人向前行走 | 🟡 中级 |
| **Run** | `humanoid_h1hand-run-v0` | 人形机器人奔跑 | 🟡 中级 |

### 复杂运动任务
| 任务 | MFP 配置 | 描述 | 难度 |
|------|----------|------|------|
| **Sit** | `humanoid_h1hand-sit_simple-v0` | 人形机器人坐下（简单版本） | 🟡 中级 |
| **Slide** | `humanoid_h1hand-slide-v0` | 人形机器人滑行 | 🔴 高级 |
| **Pole** | `humanoid_h1hand-pole-v0` | 人形机器人爬杆 | 🔴 高级 |
| **Hurdle** | `humanoid_h1hand-hurdle-v0` | 人形机器人跨越障碍物 | 🔴 高级 |

### 快速启动命令

```bash
# 基础运动任务
python mfp/train.py env_type=humanoid task=humanoid_h1hand-stand-v0
python mfp/train.py env_type=humanoid task=humanoid_h1hand-walk-v0
python mfp/train.py env_type=humanoid task=humanoid_h1hand-run-v0

# 复杂运动任务
python mfp/train.py env_type=humanoid task=humanoid_h1hand-sit_simple-v0
python mfp/train.py env_type=humanoid task=humanoid_h1hand-slide-v0
python mfp/train.py env_type=humanoid task=humanoid_h1hand-pole-v0
python mfp/train.py env_type=humanoid task=humanoid_h1hand-hurdle-v0
```

### 批量实验脚本

```bash
# 创建实验脚本
for task in stand walk run sit_simple slide pole hurdle; do
    echo "Running humanoid_h1hand-${task}-v0..."
    python mfp/train.py \
        env_type=humanoid \
        task=humanoid_h1hand-${task}-v0 \
        seed=1 \
        steps=1_000_000
done
```

### 任务特性说明

- **Stand**: 测试机器人的静态平衡能力，是其他运动任务的基础
- **Walk**: 测试周期性步态生成和动态平衡，是最基础的 locomotion 任务
- **Run**: 需要更快的速度和更复杂的动力学控制
- **Sit**: 需要精确的关节控制和身体协调，有 simple/hard 两个版本
- **Slide**: 在光滑表面滑行，测试摩擦力利用和身体控制
- **Pole**: 攀爬垂直杆，需要双手协调和力量控制
- **Hurdle**: 跨越水平障碍物，需要起跳时机和身体姿态控制

## 机器人类型

### H1 系列（带灵巧手）
- **h1hand**: Unitree H1 机器人带灵巧手
- **主要测试平台**: 大部分基准任务使用此机器人

### H1 系列（无手）
- **h1**: Unitree H1 机器人（无手版本）
- **适用场景**: 纯运动控制任务

### G1 系列
- **g1**: Unitree G1 机器人带三指手
- **特点**: 更经济实用的人形机器人平台

### 其他变体
- **h1simplehand**: H1 机器人带低维手
- **h1touch**: H1 机器人带触觉传感器

## 任务分类

### 🏃 运动类任务 (Locomotion)

#### 基础运动
- **Walk（行走）**
  - `h1hand-walk-v0` - 人形机器人行走
  - `h1-walk-v0` - 无手版本行走
  - `g1-walk-v0` - G1 机器人行走

- **Run（奔跑）**
  - `h1hand-run-v0` - 人形机器人奔跑
  - `h1-run-v0` - 无手版本奔跑
  - `g1-run-v0` - G1 机器人奔跑

- **Stand（站立）**
  - `h1hand-stand-v0` - 人形机器人站立
  - `h1-stand-v0` - 无手版本站立
  - `g1-stand-v0` - G1 机器人站立

#### 复杂运动
- **Sit（坐下）**
  - `h1hand-sit_simple-v0` - 坐下（简单）
  - `h1hand-sit_hard-v0` - 坐下（困难）
  - `h1-sit_simple-v0` / `h1-sit_hard-v0` - 无手版本
  - `g1-sit_simple-v0` / `g1-sit_hard-v0` - G1 版本

- **Balance（平衡）**
  - `h1hand-balance_simple-v0` - 平衡（简单）
  - `h1hand-balance_hard-v0` - 平衡（困难）
  - `h1-balance_simple-v0` / `h1-balance_hard-v0` - 无手版本
  - `g1-balance_simple-v0` / `g1-balance_hard-v0` - G1 版本

- **Stair（爬楼梯）**
  - `h1hand-stair-v0` - 爬楼梯
  - `h1-stair-v0` - 无手版本
  - `g1-stair-v0` - G1 版本

- **Slide（滑行）**
  - `h1hand-slide-v0` - 滑行
  - `h1-slide-v0` - 无手版本
  - `g1-slide-v0` - G1 版本

#### 障碍穿越
- **Hurdle（跨栏）**
  - `h1hand-hurdle-v0` - 跨越障碍物
  - `h1-hurdle-v0` - 无手版本
  - `g1-hurdle-v0` - G1 版本

- **Crawl（爬行）**
  - `h1hand-crawl-v0` - 爬行
  - `h1-crawl-v0` - 无手版本
  - `g1-crawl-v0` - G1 版本

- **Maze（迷宫）**
  - `h1hand-maze-v0` - 迷宫导航
  - `h1-maze-v0` - 无手版本
  - `g1-maze-v0` - G1 版本

- **Pole（爬杆）**
  - `h1hand-pole-v0` - 爬杆
  - `h1-pole-v0` - 无手版本
  - `g1-pole-v0` - G1 版本

- **Highbar（高杠）**
  - `h1hand-highbar_hard-v0` - 高杠悬挂（强化版）
  - `h1-highbar_simple-v0` - 无手版本（简单）

### 🎯 操作类任务 (Manipulation)

#### 日常操作
- **Reach（ reaching）**
  - `h1hand-reach-v0` - 伸 reaching
  - `h1-reach-v0` - 无手版本
  - `g1-reach-v0` - G1 版本

- **Push（推）**
  - `h1hand-push-v0` - 推物体
  - `h1-push-v0` - 无手版本
  - `g1-push-v0` - G1 版本

- **Cube（立方体）**
  - `h1hand-cube-v0` - 操作立方体
  - `g1-cube-v0` - G1 版本

- **Package（包裹）**
  - `h1hand-package-v0` - 处理包裹
  - `h1-package-v0` - 无手版本
  - `g1-package-v0` - G1 版本

#### 家居环境
- **Cabinet（柜子）**
  - `h1hand-cabinet-v0` - 操作柜子
  - `g1-cabinet-v0` - G1 版本

- **Door（门）**
  - `h1hand-door-v0` - 开门
  - `h1-door-v0` - 无手版本
  - `g1-door-v0` - G1 版本

- **Window（窗户）**
  - `h1hand-window-v0` - 开关窗户
  - `g1-window-v0` - G1 版本

- **Kitchen（厨房）**
  - `h1hand-kitchen-v0` - 厨房任务
  - `g1-kitchen-v0` - G1 版本

- **Bookshelf（书架）**
  - `h1hand-bookshelf_simple-v0` - 书架任务（简单）
  - `h1hand-bookshelf_hard-v0` - 书架任务（困难）
  - `g1-bookshelf_simple-v0` / `g1-bookshelf_hard-v0` - G1 版本

- **Room（房间）**
  - `h1hand-room-v0` - 房间内任务
  - `g1-room-v0` - G1 版本

#### 物体操作
- **Basketball（篮球）**
  - `h1hand-basketball-v0` - 篮球任务
  - `h1-basketball-v0` - 无手版本
  - `g1-basketball-v0` - G1 版本

- **Spoon（勺子）**
  - `h1hand-spoon-v0` - 使用勺子
  - `g1-spoon-v0` - G1 版本

- **Truck（卡车）**
  - `h1hand-truck-v0` - 操作卡车
  - `h1-truck-v0` - 无手版本
  - `g1-truck-v0` - G1 版本

- **Insert（插入）**
  - `h1hand-insert_normal-v0` - 插入任务（正常）
  - `h1hand-insert_small-v0` - 插入任务（小物体）
  - `g1-insert_normal-v0` / `g1-insert_small-v0` - G1 版本

#### 力量任务
- **Powerlift（力量举）**
  - `h1hand-powerlift-v0` - 力量举
  - `g1-powerlift-v0` - G1 版本

## 在配置文件中使用

### 基本配置

在 `mfp/config.yaml` 中设置任务：

```yaml
# 环境类型
env_type: humanoid

# 任务选择（格式：robot-task-version）
task: humanoid_h1hand-walk-v0        # H1带手行走
task: humanoid_h1hand-stand-v0       # H1带手站立
task: humanoid_h1hand-run-v0         # H1带手奔跑
task: humanoid_h1hand-push-v0        # H1带手推物
task: humanoid_h1-walk-v0            # H1无手行走
task: humanoid_g1-walk-v0            # G1行走
```

### 观察类型配置

```yaml
obs: state    # 状态观察（默认，包含完整环境状态）
obs: rgb      # 图像观察
```

### 分层策略配置

对于需要分层策略的任务（如 push），可以配置低层策略：

```yaml
# 低层策略路径
policy_path: data/reach_one_hand/torch_model.pt
mean_path: data/reach_one_hand/mean.npy
var_path: data/reach_one_hand/var.npy
policy_type: reach_single  # 或 reach_double_relative
```

### 任务命名规范

任务的命名格式为：`robot-task-version`

- `robot`: 机器人类型（h1hand, h1, g1, h1simplehand, h1touch）
- `task`: 具体任务（walk, run, stand, push, cabinet, etc.）
- `version`: 版本号（通常为 v0）

在 MFP 框架中，需要添加 `humanoid_` 前缀：
- MFP 中：`humanoid_h1hand-walk-v0`
- 原始：`h1hand-walk-v0`

## 观察空间配置

### 完整状态观察（默认）
```yaml
obs: state
```
包含机器人状态和环境状态的完整信息

### 感觉模态观察
```python
# 在代码中设置
env = gym.make("h1touch-stand-v0", obs_wrapper=True, sensors="proprio,image,tactile")
```

可用的传感器类型：
- `proprio`: 本体感觉（关节位置、速度等）
- `image`: 视觉信息
- `tactile`: 触觉信息（需要使用 h1touch 机器人）

## 任务难度分级

### 🟢 入门级任务
适合快速测试和初步实验：
- `h1hand-stand-v0` - 基本站立
- `h1-walk-v0` - 简单行走（无手）
- `h1hand-reach-v0` - 基础 reaching

### 🟡 中级任务
需要一定的控制策略：
- `h1hand-walk-v0` - 行走
- `h1hand-push-v0` - 推物体
- `h1hand-sit_simple-v0` - 简单坐下
- `h1hand-cube-v0` - 操作立方体

### 🔴 高级任务
复杂的全身协调和操作：
- `h1hand-basketball-v0` - 篮球任务
- `h1hand-bookshelf_hard-v0` - 困难书架任务
- `h1hand-insert_small-v0` - 精细插入
- `h1hand-kitchen-v0` - 厨房复杂任务
- `h1hand-maze-v0` - 迷宫导航

## 环境特定配置

### SLURM 集群配置

在 SLURM 环境中，环境会自动设置 GPU：

```python
# 自动设置（无需手动配置）
EGL_DEVICE_ID 会自动从 SLURM_STEP_GPUS 或 SLURM_JOB_GPUS 读取
```

### MuJoCo 渲染配置

```python
# Linux 环境
MUJOCO_GL = "egl"  # 默认设置

# macOS 环境
MUJOCO_GL 不设置（使用默认值）
```

## 观察空间与动作空间

### 观察空间
- **state**: 高维状态向量（包含机器人全身状态和环境信息）
- **rgb**: 可选的图像观察
- **维度**: 根据任务不同而变化，通常为几百维

### 动作空间
- **类型**: 连续动作空间
- **范围**: 通过 wrapper 归一化
- **维度**: 
  - 无手版本：约 19-30 维（主要关节）
  - 带手版本：50+ 维（包含手指关节）

## 分层策略系统

### 低层策略

某些任务支持使用预训练的低层策略：

```yaml
# 单手 reaching 策略
policy_type: reach_single
policy_path: data/reach_one_hand/torch_model.pt
mean_path: data/reach_one_hand/mean.npy
var_path: data/reach_one_hand/var.npy

# 双手 reaching 策略
policy_type: reach_double_relative
policy_path: data/reach_two_hands/torch_model.pt
mean_path: data/reach_two_hands/mean.npy
var_path: data/reach_two_hands/var.npy
```

### 可用的低层策略
- `reach_one_hand`: 单手 reaching 策略
- `reach_two_hands`: 双手 reaching 策略

## 测试环境

### 随机动作测试
```bash
# 测试环境是否正常工作
python -m humanoid_bench.test_env --env h1hand-walk-v0

# 使用分层策略测试
export POLICY_PATH="data/reach_two_hands/torch_model.pt"
export MEAN_PATH="data/reach_two_hands/mean.npy"
export VAR_PATH="data/reach_two_hands/var.npy"

python -m humanoid_bench.test_env \
  --env h1hand-push-v0 \
  --policy_path ${POLICY_PATH} \
  --mean_path ${MEAN_PATH} \
  --var_path ${VAR_PATH} \
  --policy_type "reach_double_relative"
```

### 使用特定传感器测试
```bash
python -m humanoid_bench.test_env \
  --env h1touch-stand-v0 \
  --obs_wrapper True \
  --sensors "proprio,image,tactile"
```

## 相关文件

- **环境实现**: [mfp/envs/humanoid.py](mfp/envs/humanoid.py)
- **环境定义**: [humanoid-bench/humanoid_bench/envs/](humanoid-bench/humanoid_bench/envs/)
- **训练脚本**: [mfp/train.py](mfp/train.py)
- **配置文件**: [mfp/config.yaml](mfp/config.yaml)
- **低层策略**: [humanoid-bench/data/](humanoid-bench/data/)


## 参考资料

- [HumanoidBench 论文](https://arxiv.org/abs/2403.10506)
- [HumanoidBench 网站](https://sferrazza.cc/humanoidbench_site/)
- [HumanoidBench GitHub](https://github.com/carlosferrazza/humanoid-bench)
- [Unitree H1 官方文档](https://github.com/unitreerobotics/unitree_ros)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
