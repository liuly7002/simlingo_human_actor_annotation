# 方案 B：147 + 147 独立人工盲评（新增文件，不覆盖旧算法）

基于原仓库 `simlingo_human_actor_annotation/human_actor_annotation/`。

## 设计

- A 组：在同一 `keyframes.txt` 内，`all_frame_results.jsonl` 的 `has_causal_object=true` 的**全部**147帧。不随机替换、不过滤掉无人工共识的帧。
- B 组：在同一 `keyframes.txt`、LG有输出且 `has_causal_object=false`、CVAA有有效Top-1的帧中，以固定种子随机抽取147帧。
- A/B均使用同一个*与方法输出无关*的候选生成规则。标注者无法在UI看到分组、LG结果或CVAA结果。
- 记录人工多个关键actor、无人需要关注、无法判断。原来的界面与保存格式**不变**。
- 假如CVAA在任一A组帧缺失Top-1，默认直接报错，不能偷偷将“缺失”当成“无关键actor”。需要评估覆盖率而仍要进行标注时可传入 `--allow-missing-cvaa`；B组此时也允许CVAA缺失且由原LG阴性池随机采样。
- 当早期关键帧真实历史不足4秒，视频前部呈现 `HISTORY NOT AVAILABLE` 静态卡，而**不会复制第一帧伪造历史**。`real_history_frames` 记录真实历史帧数。论文应报告此情况；可使用 `--min-real-history 10` 强制至少1秒真实历史，但这可能导致无法保留全部147帧并终止生成。

**重要限制**：这是按 LG 输出分层的条件性人工验证。两组各147帧不能作为原数据集分布下的无偏总体准确率。相同一帧在统计上应当配对，结果分别报告 LG确认/未确认组和人工有/无actor组。

## 需要放置的文件

将 ZIP 中 `human_actor_annotation/` 目录里的所有文件，复制到原仓库 `human_actor_annotation/` 同名目录。**原来的 `annotation_app.py`、`annotation_core/`、`assign_samples.py` 不需要修改**。本包都是新增文件：

- `scheme_b_common.py`: 统一共同关键帧、LG、CVAA主键与加载。
- `prepare_scheme_b.py`: 生成平衡294帧manifest、预测真值连接文件和审计。
- `evaluate_scheme_b.py`: 完整配对人工一致性统计。
- `config.scheme_b.example.yaml`: 独立工作区配置模板。
- `tests/test_scheme_b.py`: 模拟数据基础测试。

## 1. 准备

```bash
cd /root/autodl-tmp/simlingo_human_actor_annotation/human_actor_annotation
conda activate actor_annotation
cp config.scheme_b.example.yaml config.scheme_b.yaml
# 检查 config.scheme_b.yaml 的 dataset_root 与 render.top_ego_center_y_ratio
```

请使用*现有数据*而不是重新计算模型。路径如下：

```
keyframes.txt
  /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/keyframes.txt
LG all_frame_results.jsonl
  <LG 汇总输出目录>/all_frame_results.jsonl
CVAA all_frame_rankings.jsonl
  <CVAA 汇总输出目录>/all_frame_rankings.jsonl
```

LG汇总由 `lg_waypoint_planner_project/tools_bev/aggregate_lg_cvaa_results.py` 输出。CVAA排名由 `CVAA_baseline/cvaa/pipeline_optimized.py` 输出。**不要把 LG 的 `all_actor_scores.jsonl` 误当成 `all_frame_results.jsonl`，也不要把 CVAA 的 `all_actor_scores.jsonl` 当成 Top-1 帧排名。**

## 2. 自动生成 294 个样本

```bash
python prepare_scheme_b.py \
  --config config.scheme_b.yaml \
  --keyframes /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/keyframes.txt \
  --lg-frames /路径/lg_cvaa_results/all_frame_results.jsonl \
  --cvaa-rankings /路径/CVAA/all_frame_rankings.jsonl \
  --target 147 --seed 20261008
```

**重要**：程序要求源文件中 `has_causal_object=true` 的LG帧恰好147个；如果数量不同会**报错**，防止不知不觉混用不同批次数据。A组147帧还必须全部满足当前标注数据可用性；如不满足会输出 `scheme_b_preflight_error.json`，需要检查问题后再决定正式策略，不允许自动拿别的阳性样本替换。

成功后输出：

```
annotation_workspace_scheme_b/
  manifest.jsonl
  scheme_b_unavailable_history.png
  evaluation/
    scheme_b_predictions.jsonl
    scheme_b_prepare_summary.json
    scheme_b_rejected_frames.csv
    scheme_b_actor_mapping_audit.csv
```

**标注前必须检查** `scheme_b_actor_mapping_audit.csv`，理想情况为空。如果某方法选择的actor不在人工候选列表中，应先排查actor ID对齐/候选筛选，**不要**把该问题视作该方法自然预测错误。也要核对 `scheme_b_prepare_summary.json` 中分组数是否147/147。

## 3. 3名研究人员独立标注

```bash
python assign_samples.py \
  --config config.scheme_b.yaml \
  --annotators P01 P02 P03 --redundancy 3

# 三人各自独立的浏览器会话（或分配到单独端口）
python annotation_app.py --config config.scheme_b.yaml --annotator-id P01
```

如果采用多个分配manifest文件，每名标注员单独配置对应 `dataset.manifest_path`，但保留同一 `annotation_dir`。3人同时用同一原 manifest 也可以，三名标注员ID必须不同。尽量不要让标注人员查看生成预测、分组或模型输出。正式标注前可从新工作区清除试验记录，不能合并此前旧工作区的标注。**已有人工标注时准备脚本将拒绝重新生成 manifest**，防止随机样本被悄悄替换。

标注指令：看当前时刻前真实可见的历史+当前画面，选择后2秒内最值得驾驶员注意、可能影响驾驶判断的 **一个或多个** actor。允许“无关注对象”和“无法判断”。不能看到未来真实帧。历史缺失卡代表真实数据未录到这些时刻。

## 4. 评价

```bash
python evaluate_scheme_b.py --config config.scheme_b.yaml
```

输出：

```
annotation_workspace_scheme_b/evaluation/
  scheme_b_evaluation_summary.json
  scheme_b_human_comparison.csv
  scheme_b_disagreements.csv
  scheme_b_non_evaluable.csv
```

主要指标：

- `human_positive_hit_at_1`：人工共识“有actor”的场景中，所选actor落入人工集合的比例；LG无输出计未命中。
- `human_none_specificity`：人工共识“无actor”的场景中，方法是否不选actor。
- `overall_correct_including_human_none`：综合人工正例与负例；只在**双方法结果存在**的共同帧比较，不能解释为总体无偏准确率。
- `human_unique_primary_strict_top1`：人工可给出唯一主要actor时的严格同一ID匹配。
- `groups`：分别输出LG阳性147和LG阴性随机147的配对结果。
- `exact_mcnemar_p`：两方法正确/错误配对后的双侧精确McNemar检验。
- `route_cluster_bootstrap_difference_95ci`：按路线重采样的95%置信区间（解释时优先参考该区间；普通 McNemar p 值未对同一路线的相关性修正）。
- `method_prediction_coverage_on_human_evaluable`：两方法覆盖率，CVAA缺失不会记成“无actor”。

**注意**：每帧至少3名给出明确答案（actors/none）的标注员才进入默认主指标；可以通过 `--min-decisive 2` 做敏感性分析。人工无多数意见及`uncertain`帧需另行披露。样本规模仅294，性能差异置信区间可能较宽。

## 5. 本地自检

```bash
python -m py_compile scheme_b_common.py prepare_scheme_b.py evaluate_scheme_b.py
python tests/test_scheme_b.py
```

验证脚本不调用网络，不要求GPU。*最终真实统计数值必须在你的CARLA数据目录中运行脚本后才能确定*。
