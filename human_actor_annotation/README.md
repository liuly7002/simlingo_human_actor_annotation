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
- 生成约 4 秒的六视角历史短片，视频最后一帧为当前判断时刻，不展示未来画面；
- 根据 `measurements` 中的自车速度优先保留行驶片段；
- 在当前时刻的 `top_rgb` 上绘制匿名 actor 编号，采用锚点、引导线和自动避让布局；
- 测试人员直接选择一个或多个重点关注 actor，也可选择“无关注对象”或“无法判断”；
- 支持多人通过浏览器独立标注；
- 自动断点续标，允许回看和修订；
- 每名标注员分别保存 JSONL 和 CSV；
- 统计集合完全一致率、平均 Jaccard、集合 F1 和二元 Krippendorff alpha；
- 构造多人共识 actor 集合，并与 LG 最终主要 actor 计算 Hit Rate 和 Strict Top-1；
- LG 提供分析候选列表时，额外计算 Candidate Recall@K；
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

如果此前使用的是“前 20 帧 + 后 20 帧”manifest，必须删除旧 manifest 并按新配置重新生成；标注界面会拒绝包含未来帧的旧 manifest。

默认设置为：

- 历史 40 帧、当前 1 帧、未来 0 帧，10 FPS 下首尾时间跨度约 4 秒；视频最后一帧即当前判断时刻；
- 标注人员只能看到当前时刻之前的历史变化，不能看到当前时刻之后的真实未来画面，避免使用 LG 在该时刻不可获得的信息；
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

`max_samples` 是最终样本数量上限，`sample_stride` 是当前判断帧的抽样间隔。实际数量还会受到 4 秒历史窗口、六视角完整性、boxes、候选数量和行驶状态筛选等条件影响。

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

测试人员只需要完成两个核心问题：

1. 视频结束时刻作为当前判断时刻；根据此前约 4 秒的场景变化，从匿名编号中选择当前时刻之后约 2 秒内会重点关注，并且可能影响驾驶判断或操作的交通参与者；
2. 给出 1～5 级判断置信度。

时间轴定义为：

```text
约 4 秒历史画面 → 视频最后一帧/当前判断时刻 → 判断随后约 2 秒需要关注的对象
```

视频不包含当前判断时刻之后的真实画面，因此测试人员是在当前可用信息下进行判断，而不是观看未来后再进行事后确认。

第一个问题允许：

- 选择一个 actor；
- 选择多个 actor；
- 选择“没有需要重点关注的对象”；
- 选择“画面信息不足，无法判断”。

“无关注对象”和“无法判断”位于同一个选择列表末尾，选择其中任意一个时不能再同时选择 actor 编号。

这里的标注目标是获得**人工驾驶关注对象集合**，用于验证 LG 最终选择的主要 actor 是否落入人工共识集合。界面不再询问“移除对象后动作是否变化”或“对象导致什么驾驶动作”，避免把对象选择、因果验证和动作解释混在同一问卷中。

仅仅出现在画面中、但不会影响驾驶判断的对象不需要选择。红绿灯、道路结构和交通规则本身不属于 Actor。

当前时刻的 Top RGB 中：小锚点表示 actor 的真实投影位置，圆形编号通过引导线与锚点相连；多个编号会自动避让，边缘对象的编号会被约束在图像内部。

界面不会展示：

- LG 最终选择的主要 actor；
- LG 进入反事实分析的候选排序；
- `causal_score`；
- LG 问答或轨迹结果。

## 9. 标注结果

```text
annotation_workspace/annotations/
├── annotations_P01.jsonl
├── annotations_P01.csv
├── annotations_P02.jsonl
└── annotations_P02.csv
```

JSONL 中保存：

- `attention_status`：`actors`、`none`、`uncertain` 或 `skip`；
- 匿名编号，如 `A2`；
- 对应的真实 actor ID 和 actor 类别；
- 判断置信度；
- 可选备注；
- 单条耗时；
- 修订版本号。

同一测试人员重新提交同一样本时不会覆盖历史记录，而是追加一个新修订；统计时只使用最新修订。

当前正式字段与早期试标字段不同。开始正式实验前，建议删除旧的测试标注目录，或在 `config.yaml` 中使用新的 `work_dir`，不要把早期试标记录和正式结果混合。

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

默认采用严格多数规则：某个 actor 的投票比例必须 **大于 0.5**，才进入人工共识集合。阈值可修改：

```bash
python evaluate_annotations.py \
  --config config.yaml \
  --consensus-threshold 0.5
```

输出：

```text
annotation_workspace/evaluation/
├── summary.json
└── consensus.csv
```

`summary.json` 中的人工一致性指标包括：

- `pairwise_status_agreement`：两名测试人员对“有 actor / 无 actor / 无法判断”的状态一致率；
- `pairwise_exact_set_agreement`：两人选择的 actor 集合完全相同的比例；
- `pairwise_mean_jaccard`：两人 actor 集合的平均 Jaccard 相似度；
- `pairwise_mean_set_f1`：两人 actor 集合的平均集合 F1；
- `krippendorff_alpha_binary`：把每个“样本—候选 actor”视为选中/未选中的二元标注单元后计算的一致性。

`consensus.csv` 给出：

- 人工共识 actor 集合 `consensus_actor_ids`；
- 唯一人工主要 actor `consensus_primary_actor_id`（最高票唯一且超过阈值时才存在）；
- 各 actor 的票数和投票比例；
- 无关注对象和无法判断的票数；
- 是否需要专家复核。

## 12. 与 LG 结果比较

生成 LG 预测文件：

```bash
python generate_lg_predictions.py --config config.yaml
```


运行：

```bash
python evaluate_annotations.py \
  --config config.yaml \
  --predictions annotation_workspace/evaluation/lg_predictions.jsonl \
  --candidate-k 3
```

最终主要 actor 字段兼容：

```text
selected_actor_id
critical_actor_id
causal_actor_id
primary_actor_id
causal_object.id
critical_actor.id
result.causal_object.id
```

候选列表字段兼容：

```text
candidate_actor_ids
top_candidate_actor_ids
analysis_candidate_actor_ids
topk_actor_ids
ranked_actor_ids
candidates
candidate_actors
ranked_candidates
```

评价结果包括：

- **Human-consensus Hit Rate**：LG 最终主要 actor 是否落入人工共识 actor 集合；
- **Strict Top-1 Accuracy**：人工存在唯一主要 actor 时，LG 是否与其完全一致；
- **Candidate Recall@K**：人工共识 actor 中有多少进入 LG 前 K 名分析候选；
- 缺失 LG 结果或缺失主要 actor 的样本数；
- `prediction_comparison.csv`：所有可评价样本的逐条结果；
- `prediction_disagreements.csv`：LG 主要 actor 未命中人工共识集合的样本。

需要明确区分：Hit Rate 和 Strict Top-1 验证 LG **最终主要 actor 选择**；Candidate Recall@K 验证 LG **前置候选生成/筛选**。二者不能作为同一个指标解释。

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
- 每条样本至少由 3 名研究人员独立标注；条件允许时建议使用 5 名；
- 根据争议样本完善说明，但不要向测试人员展示 LG 结果；
- 正式阶段抽取 600～1000 条；
- 保证每条至少 3 人标注；
- 对无多数共识、低置信度或人工主要 actor 并列的样本单独专家复核；
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
- manifest 生成，并验证视频最后一帧与当前判断帧一致；
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
