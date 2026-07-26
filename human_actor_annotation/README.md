# SimLingo 关键 Actor 人工盲评标注工具

该项目用于对 `simlingo_liulei` 生成的关键 actor 进行独立人工验证。它直接读取 `data_collection.py` 收集的数据，不读取或展示 LG 选择结果、语言内容和 causal score，从而避免测试人员受到方法输出影响。

## 1. 已实现功能

- 自动递归发现数据集中的路线目录；
- 兼容当前六视角目录名：`rgb_front`、`rgb_front_left`、`rgb_front_right`、`rgb_rear`、`rgb_rear_left`、`rgb_rear_right`；
- 兼容旧命名，例如 `rgb`、`rgb_left_front` 和 `rgb_back`；
- 读取 `.json.gz` 或 `.json` 格式的 `boxes`；
- 将 actor 统一整理为 `actor_id、class、x_m、y_m、distance_m、speed_mps`；
- 自动排除 ego vehicle，按距离和位置筛选候选；
- 候选 `A1、A2……` 按样本稳定随机化，避免 `A1` 总是最近对象；
- 生成约 4 秒的“历史帧 + 当前帧 + 未来帧”六视角同步短片；
- 根据 `measurements` 中的自车速度优先保留行驶片段；
- 在中心帧 `top_rgb` 上绘制匿名 actor 编号，采用锚点、引导线和自动避让布局；
- 支持单一关键对象、多个共同对象、无关键对象、无法判断；
- 支持多人通过浏览器独立标注；
- 自动断点续标，允许回看和修订；
- 每名标注员分别保存 JSONL 和 CSV；
- 统计人工一致率、Fleiss' Kappa、共识结果；
- 可选地将人工共识与 LG actor 预测进行 Top-1 比较；
- 支持按指定冗余度将样本均衡分配给多名测试人员。

## 2. 目录结构

```text
human_actor_annotation/
├── annotation_app.py              # Gradio 标注界面
├── prepare_samples.py             # 生成标注 manifest
├── inspect_dataset.py             # 检查真实数据目录和 boxes 字段
├── assign_samples.py              # 多人均衡分配样本
├── evaluate_annotations.py        # 汇总一致性并与 LG 结果比较
├── config.example.yaml            # 配置模板
├── requirements.txt
├── VERSION
├── .gitignore
├── run_annotation.sh
├── annotation_core/
│   ├── actors.py
│   ├── config.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── render.py
│   └── storage.py
└── tests/
    ├── make_demo_dataset.py
    └── smoke_test.py
```

## 3. 安装

建议单独建立轻量环境，避免改变 SimLingo 训练环境：

```bash
conda create -n actor_annotation python=3.8 -y
conda activate actor_annotation
pip install -r requirements.txt
```

也可以直接在已有环境中安装依赖。

## 4. 配置真实数据集

复制配置模板：

```bash
cp config.example.yaml config.yaml
```

修改：

```yaml
dataset:
  dataset_root: /home/kemove/ll/simlingo_liulei/database/你的数据集/data
```

`dataset_root` 应指向 `data_collection.py` 生成目录中的 `data` 层。程序会继续向下递归寻找包含六视角文件夹的路线目录。

`config.yaml` 包含本机绝对路径，已由 `.gitignore` 排除，不应提交到 Git。仓库只保留可公开复制的 `config.example.yaml`。

## 5. 先检查数据结构

```bash
python inspect_dataset.py --config config.yaml --routes 3
```

该命令会显示：

- 找到的路线数量；
- 六视角目录实际路径；
- `top_rgb、boxes、measurements` 路径；
- 中间帧的 actor 数量；
- 前三个 actor 的标准化结果。

应重点确认：

```text
x_m：自车前向距离
 y_m：自车右向距离
```

当前项目按 CARLA/Unreal 自车坐标约定绘制：`x` 向前、`y` 向右。

## 6. 生成标注样本

复制并修改配置后，直接运行：

```bash
python prepare_samples.py --config config.yaml
```

默认设置为：

- 历史 20 帧、当前 1 帧、未来 20 帧，10 FPS 下首尾时间跨度约 4 秒；
- 每隔 40 帧抽取一次，降低相邻片段的重复；
- 自车中心速度达到 1.0 m/s，或片段中至少 50% 的有效帧达到 1.0 m/s，即视为行驶片段；
- 优先保留行驶片段，静止片段最多占 10%，用于红灯、排队、障碍停车等必要场景；
- 最多生成 1000 条；
- 每条最多展示 20 个候选 actor；默认跳过候选被截断的样本，同时保留零候选帧作为“无关键 actor”负样本。

标注样本数量主要由以下配置控制：

```yaml
sampling:
  sample_stride: 40
  max_samples: 1000
```

`max_samples` 是最终样本数量上限，`sample_stride` 是中心帧抽样间隔。实际数量还会受到 4 秒完整窗口、六视角完整性、boxes、候选数量和行驶状态筛选等条件影响。

`prepare_summary.json` 会记录：

- `samples_written`；
- `moving_samples_written`；
- `stationary_samples_written`；
- `motion_unknown_samples_written`；
- 各类跳过原因。

输出：

```text
annotation_workspace/
├── manifest.jsonl
└── prepare_summary.json
```

预处理阶段只建立 manifest，不会一次性生成全部视频。标注界面打开某个样本时才生成并缓存视频，避免预处理耗时和磁盘占用过大。

## 7. 启动标注界面

单机：

```bash
python annotation_app.py --config config.yaml --annotator-id P01
```

浏览器打开：

```text
http://127.0.0.1:7860
```

局域网多人访问：

```bash
python annotation_app.py \
  --config config.yaml \
  --host 0.0.0.0 \
  --port 7860
```

其他测试人员打开：

```text
http://运行程序的电脑IP:7860
```

每名测试人员必须填写不同的标注员编号，例如 `P01、P02、P03`。Gradio 的页面状态按浏览器会话隔离，标注文件也按编号分别保存。

## 8. 界面中的问题

测试人员依次完成：

1. 判断“自车接下来约 2 秒的安全动作是否受到交通参与者直接约束”；
2. 若存在约束，判断是一个主要对象还是多个对象共同作用，并勾选匿名 actor 编号；
3. 假设所选对象不存在，自车动作是否改变；
4. 该对象主要导致减速、停车、左右避让、低速观察等哪种动作；
5. 给出 1～5 级置信度；
6. 可选备注。

界面给出的关键对象定义是：假设某个交通参与者不存在，如果自车接下来约 2 秒内的减速、停车、避让或保持车距等安全动作会明显变化，则该对象属于关键对象。红绿灯、道路结构和交通规则本身不属于 Actor。

Top RGB 中：小锚点表示 actor 的真实投影位置，圆形编号通过引导线与锚点相连；多个编号会自动避让，边缘对象的编号会被约束在图像内部。

界面不会展示：

- LG 选择的 actor；
- `causal_score`；
- LG 问答；
- 关键 actor 的候选排序分数。

## 9. 标注结果

```text
annotation_workspace/annotations/
├── annotations_P01.jsonl
├── annotations_P01.csv
├── annotations_P02.jsonl
└── annotations_P02.csv
```

JSONL 中同时保存：

- 匿名编号，如 `A2`；
- 真实 actor ID，如 `2145`；
- actor 类别；
- 场景判断；
- 动作影响；
- 置信度；
- 单条耗时；
- 修订版本号。

同一测试人员重新提交同一样本时不会覆盖历史记录，而是追加一个新修订；统计时只使用最新修订。

## 10. 多人样本分配

假设有 5 名测试人员，每条样本希望由 3 人独立标注：

```bash
python assign_samples.py \
  --config config.yaml \
  --annotators P01 P02 P03 P04 P05 \
  --redundancy 3
```

输出：

```text
annotation_workspace/assignments/
├── manifest_P01.jsonl
├── manifest_P02.jsonl
├── manifest_P03.jsonl
├── manifest_P04.jsonl
├── manifest_P05.jsonl
└── assignment_summary.json
```

为每名测试人员复制一份配置，只修改：

```yaml
dataset:
  manifest_path: ./annotation_workspace/assignments/manifest_P01.jsonl
```

所有配置仍应使用同一个 `annotation_dir`，便于最后统一统计。

## 11. 统计人工一致性

```bash
python evaluate_annotations.py --config config.yaml
```

输出：

```text
annotation_workspace/evaluation/
├── summary.json
└── consensus.csv
```

`summary.json` 包含：

- pairwise exact agreement；
- dominant actor agreement；
- Fleiss' Kappa；
- 标注员数量；
- 需要专家复核的歧义样本数量。

`consensus.csv` 给出每条样本的多数共识 actor 和投票比例。

## 12. 与 LG 结果比较

准备一个 JSONL：

```json
{"sample_id": "Town12__route_x__0045", "selected_actor_id": "2145"}
```

然后运行：

```bash
python evaluate_annotations.py \
  --config config.yaml \
  --predictions lg_predictions.jsonl
```

程序也兼容以下常见字段：

```text
critical_actor_id
causal_actor_id
causal_object.id
critical_actor.id
result.causal_object.id
```

输出会增加：

- 人工共识可评样本数；
- LG Top-1 actor accuracy；
- 缺失预测数；
- `prediction_disagreements.csv`。

## 13. Top RGB 编号位置校准

程序优先读取每条路线下的：

```text
surround_camera_config.json
```

并根据 top camera 的高度、FOV 和图像宽度自动计算米/像素关系。

如果编号整体位置正确但缩放不一致，可以在配置中设置：

```yaml
render:
  top_meters_per_pixel: 0.15625
```

如果自车不在图像中心，调整：

```yaml
render:
  top_ego_center_x_ratio: 0.5
  top_ego_center_y_ratio: 0.62
```

如果前后或左右方向相反：

```yaml
render:
  forward_is_up: false
  right_is_right: false
```

## 14. 推荐正式实验设置

第一轮建议：

- 先抽取 50～100 条进行界面和标注规范试运行；
- 由 3 名研究人员独立标注；
- 根据争议样本完善说明，但不要向测试人员展示 LG 结果；
- 正式阶段抽取 600～1000 条；
- 保证每条至少 3 人标注；
- 对无多数共识、低置信度和多因果场景单独专家复核；
- 论文同时报告人工一致性和 LG—人工一致性。

## 15. 自检

项目附带完整合成数据测试：

```bash
python tests/smoke_test.py
```

测试产生的 `config.demo.yaml`、`demo_dataset/` 和 `demo_workspace/` 已由 `.gitignore` 排除。

该测试会验证：

- 六视角目录发现；
- `.json.gz` boxes 解析；
- manifest 生成；
- 六视角 MP4 生成；
- Top RGB 编号绘制；
- JSONL/CSV 保存；
- 共识统计；
- Gradio 界面构建。

## 代理环境报错

如果启动时出现：

```text
ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7897/')
```

这是终端中的代理环境变量使用了 HTTPX 不支持的 `socks://` 协议头。当前版本的
`annotation_app.py` 会在导入 Gradio 前，仅在本 Python 进程中自动忽略这种格式错误的
代理变量，不会修改系统或终端的永久代理设置。

也可以在命令行中临时关闭代理后启动：

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    python annotation_app.py --config config.yaml --annotator-id P01
```

如确实需要让 Python 使用 SOCKS5 代理，应使用 `socks5://`，并安装 HTTPX 的 SOCKS
可选依赖：

```bash
pip install "httpx[socks]"
export ALL_PROXY=socks5://127.0.0.1:7897
```


## 视频区域空白的修复

项目现在会把 OpenCV 中间视频转码为浏览器兼容的 H.264/yuv420p MP4，`mp4v` 格式的不兼容缓存会自动重建。
若终端打印“未找到 FFmpeg”，执行：

```bash
pip install "imageio-ffmpeg>=0.4.9,<0.6"
```

也可以安装系统 FFmpeg。完成后重启 `annotation_app.py`。如需手动清除全部旧视频缓存：

```bash
find "$(python - <<'PY'
from annotation_core.config import load_config
print(load_config('config.yaml').dataset.work_dir / 'cache')
PY
)" -name 'surround_context*' -delete
```
