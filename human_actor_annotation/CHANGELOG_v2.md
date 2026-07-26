# 第二版修改说明

## 1. 问题定义

第一问改为：

> 自车接下来约 2 秒的安全动作，是否受到交通参与者的直接约束？

界面同时显示关键对象定义：假设对象不存在后，自车的减速、停车、避让或保持车距等安全动作会明显变化。红绿灯、道路结构和交通规则本身不属于 Actor。

内部保存值保持不变，因此原有统计脚本仍兼容。

## 2. Top RGB 编号布局

- 小锚点表示 actor 的实际投影位置；
- 圆形 `A#` 标签通过引导线连接锚点；
- 标签在多个候选方向和距离上自动搜索空闲位置；
- 标签始终约束在图像内部；
- 靠近图像边缘的对象使用边缘锚点；
- 新缓存签名会自动绕过第一版旧图片和旧视频缓存。

## 3. 约 4 秒视频

默认使用：

```yaml
history_frames: 20
future_frames: 20
fps: 10
```

共 41 帧，第一帧到最后一帧的时间跨度约 4 秒。

## 4. 行驶状态优先采样

从每帧 `measurements` 中读取实际自车速度。默认规则：

- 中心帧速度不低于 1.0 m/s；或
- 片段内至少 50% 的有效速度帧不低于 1.0 m/s；

满足任一条件即视为行驶片段。静止片段最多保留配置上限，用于红灯、排队和障碍停车等必要场景。

## 从第一版升级

```bash
python upgrade_config_v2.py --config config.yaml
python prepare_samples.py --config config_v2.yaml
python annotation_app.py --config config_v2.yaml --annotator-id P01
```

升级脚本会使用独立的 `annotation_workspace_v2`，防止第一版试标结果与第二版混合。
