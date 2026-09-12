---
date: 2026-07-30
topic: robotics
source: trending
repo: rsasaki0109/lidar_slam_ros2
file: scanmatcher/include/scanmatcher/map_update_policy.hpp
permalink: https://github.com/rsasaki0109/lidar_slam_ros2/blob/18712d2c9e4a7828450b61bf9192796db5e644ac/scanmatcher/include/scanmatcher/map_update_policy.hpp#L14-L111
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, slam, keyframe]
---

# lidar_slam_ros2 map policy：地图更新先变成小判定 / lidar_slam_ros2 Map Policy: Turn Map Updates into Small Decisions

> **一句话 / In one line**: `map_update_policy.hpp` 把 keyframe 触发、局部子图选择、发布周期和自适应阈值拆成可测试的纯函数。 / `map_update_policy.hpp` splits keyframe triggering, local submap selection, publish timing, and adaptive thresholds into testable pure functions.

## 为什么重要 / Why this matters

SLAM 节点很容易被 ROS callback、点云、TF 和优化器缠在一起。这个文件反过来做：把“是否该更新地图”的决策抽成无副作用函数，让策略能单独测试。

SLAM nodes easily entangle ROS callbacks, point clouds, TF, and optimizers. This file does the opposite: it extracts the decision of whether to update the map into side-effect-free functions that can be tested alone.

## 代码 / The code

`rsasaki0109/lidar_slam_ros2` — [`scanmatcher/include/scanmatcher/map_update_policy.hpp`](https://github.com/rsasaki0109/lidar_slam_ros2/blob/18712d2c9e4a7828450b61bf9192796db5e644ac/scanmatcher/include/scanmatcher/map_update_policy.hpp#L14-L111)

```cpp
// Keyframe trigger: a map update starts once the robot moved far enough,
// no async update is in flight, and pose acceptance did not suppress it.
inline bool shouldTriggerMapUpdate(
  const double trans,
  const double trans_for_mapupdate,
  const bool mapping_in_flight,
  const bool suppress_map_update)
{
  return trans >= trans_for_mapupdate && !mapping_in_flight && !suppress_map_update;
}

// Async warmup gate: run the map update on a worker thread only after the
// map holds enough submaps (warmup_submaps <= 0 disables the warmup).
inline bool useAsyncMapUpdate(
  const bool async_map_update,
  const int async_map_update_warmup_submaps,
  const int num_submaps)
{
  return async_map_update &&
         (async_map_update_warmup_submaps <= 0 ||
         num_submaps >= async_map_update_warmup_submaps);
}

// VoxelHashMap mode: local points within a spatial radius form the
// registration target, capped at 50 m.
inline double voxelHashLocalRadius(const double voxel_hash_map_max_distance)
{
  return std::min(voxel_hash_map_max_distance, 50.0);
}

// Spatial local map: walk submaps newest-first and keep those within the
// radius of the current position, up to num_targeted_cloud - 1 entries
// (the live scan itself is the remaining one). Returns indices in the
// historical iteration order (newest to oldest).
inline std::vector<int> selectSpatialSubmapIndices(
  const std::vector<Eigen::Vector3d> & submap_positions,
  const Eigen::Vector3d & current_position,
  const double spatial_local_map_radius,
  const int num_targeted_cloud)
{
  std::vector<int> indices;
  const int num_submaps = static_cast<int>(submap_positions.size());
  int added = 0;
  for (int i = num_submaps - 1; i >= 0 && added < num_targeted_cloud - 1; i--) {
    const double dist = (submap_positions[i] - current_position).norm();
    if (dist <= spatial_local_map_radius) {
      indices.push_back(i);
      added++;
    }
  }
  return indices;
}

// Temporal local map: the num_targeted_cloud - 1 most recent submaps,
// newest first (original behavior).
inline std::vector<int> selectTemporalSubmapIndices(
  const int num_submaps,
  const int num_targeted_cloud)
{
  std::vector<int> indices;
  for (int i = 0; i < num_targeted_cloud - 1; i++) {
    if (num_submaps - 1 - i < 0) {continue;}
    indices.push_back(num_submaps - 1 - i);
  }
  return indices;
}

// Periodic full-map publish gate.
inline bool shouldPublishMap(const double dt_sec, const double map_publish_period)
{
  return dt_sec > map_publish_period;
}

// Adaptive correspondence-distance EMA update after alignment. A
// non-positive mean correspondence distance leaves the EMA unchanged; the
// first positive sample seeds it.
inline double updateAdaptiveCorrespondenceEma(
  const double current_ema,
  const double mean_correspondence_distance,
  const double ema_alpha)
{
  if (mean_correspondence_distance > 0.0) {
    if (current_ema <= 0.0) {
      return mean_correspondence_distance;  // Initialize
    }
    return ema_alpha * mean_correspondence_distance + (1.0 - ema_alpha) * current_ema;
  }
  return current_ema;
}

// Max correspondence distance applied before alignment when the adaptive
// threshold is active (the shell only applies it while the EMA is seeded).
inline double adaptiveMaxCorrespondenceDistance(
  const double adaptive_corr_dist_multiplier,
  const double adaptive_corr_dist_ema)
{
  return adaptive_corr_dist_multiplier * adaptive_corr_dist_ema;
}
```

## 逐行讲解 / What's happening

1. **第 14-23 行 / Lines 14-23 (`shouldTriggerMapUpdate`)**:
   - 中文: 触发 keyframe 要同时满足移动距离、无异步更新、未被 pose acceptance 抑制。
   - English: A keyframe triggers only when travel distance is enough, no async update is running, and pose acceptance has not suppressed it.
2. **第 27-35 行 / Lines 27-35 (`useAsyncMapUpdate`)**:
   - 中文: 异步更新有 warmup gate：子图数量够了才放到 worker thread。
   - English: Async updating has a warmup gate: enough submaps must exist before work moves to a worker thread.
3. **第 48-79 行 / Lines 48-79 (local submap selection)**:
   - 中文: 空间模式按半径从新到旧选子图；时间模式只拿最近 N 个。
   - English: Spatial mode selects recent submaps within a radius; temporal mode simply takes the latest N.
4. **第 90-111 行 / Lines 90-111 (adaptive EMA)**:
   - 中文: 有效匹配距离会更新 EMA，之后用 multiplier 生成下一轮最大 correspondence distance。
   - English: Valid correspondence distance updates an EMA, then a multiplier turns it into the next max correspondence distance.

## 类比 / The analogy

这像扫地机器人决定什么时候更新房间地图：走得不够远不更新，后台还在处理也不更新，只挑附近房间做参考。

It is like a robot vacuum deciding when to update its room map: do not update before moving far enough, do not update while background work is running, and use nearby rooms as reference.

## 自己跑一遍 / Try it yourself

```python
def should_trigger(trans, threshold, in_flight, suppressed):
    return trans >= threshold and not in_flight and not suppressed

def select_spatial(positions, current, radius, target_clouds):
    out = []
    for i in range(len(positions) - 1, -1, -1):
        if len(out) >= target_clouds - 1:
            break
        if abs(positions[i] - current) <= radius:
            out.append(i)
    return out

def update_ema(current, sample, alpha):
    if sample <= 0:
        return current
    return sample if current <= 0 else alpha * sample + (1 - alpha) * current

print(should_trigger(1.2, 1.0, False, False))
print(select_spatial([0.0, 0.5, 2.0, 2.8], 2.5, 0.7, 3))
print(round(update_ema(0.4, 0.8, 0.25), 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
True
[3, 2]
0.5
```

纯函数让 SLAM 策略变得可测：不用启动 ROS，也能验证边界行为。

Pure functions make SLAM policy testable: boundary behavior can be checked without launching ROS.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Keyframe SLAM** / **Keyframe SLAM**: 多数前端都用位移/旋转阈值决定何时加入关键帧。 / Most frontends use translation/rotation thresholds to decide when to add keyframes.
- **Adaptive ICP gates** / **Adaptive ICP gates**: 配准阈值随最近匹配质量调整，是生产 SLAM 常见策略。 / Registration thresholds adapting to recent match quality are common in production SLAM.

## 注意事项 / Caveats / when it breaks

- **阈值决定地图密度** / **Thresholds decide map density**: 距离阈值太小会频繁更新，太大又会丢局部细节。 / Too small a distance threshold updates too often; too large loses local detail.
- **空间选择依赖姿态质量** / **Spatial choice depends on pose quality**: 当前位姿错了，半径选出来的局部地图也会错。 / If the current pose is wrong, radius-based local map selection is wrong too.

## 延伸阅读 / Further reading

- [lidar_slam_ros2 repository](https://github.com/rsasaki0109/lidar_slam_ros2)
- [Source permalink](https://github.com/rsasaki0109/lidar_slam_ros2/blob/18712d2c9e4a7828450b61bf9192796db5e644ac/scanmatcher/include/scanmatcher/map_update_policy.hpp#L14-L111)
