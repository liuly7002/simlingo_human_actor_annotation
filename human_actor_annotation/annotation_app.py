from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _remove_unsupported_socks_proxy_env() -> None:
    """Remove malformed ``socks://`` proxy variables before importing Gradio.

    HTTPX accepts ``socks5://`` when its optional SOCKS dependency is installed,
    but it rejects the non-standard ``socks://`` scheme during import.  This
    annotation application only serves local files, so ignoring only those
    malformed proxy variables is the safest default and does not modify the
    user's shell environment outside this Python process.
    """
    proxy_names = (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    )
    removed = []
    for name in proxy_names:
        value = os.environ.get(name, "").strip()
        if value.lower().startswith("socks://"):
            removed.append(f"{name}={value}")
            os.environ.pop(name, None)

    if removed:
        print(
            "[Proxy] 已忽略 HTTPX 不支持的 socks:// 代理变量："
            + ", ".join(removed)
        )


_remove_unsupported_socks_proxy_env()

import gradio as gr

from annotation_core.config import load_config
from annotation_core.dataset import load_jsonl
from annotation_core.render import candidate_table, render_sample_assets
from annotation_core.storage import (
    append_annotation,
    choose_next_index,
    load_latest_annotations,
    sanitize_annotator_id,
)


SCENE_CHOICES = [
    ("有，一个对象起主要作用", "存在一个主要对象"),
    ("有，多个对象共同约束自车", "多个对象共同影响"),
    ("没有（正常行驶，或主要受道路/信号规则约束）", "无关键对象"),
    ("画面信息不足，无法判断", "无法判断"),
]
SCENE_VALUES = {value for _, value in SCENE_CHOICES}
REMOVAL_CHOICES = ["会明显改变", "可能轻微改变", "不会改变", "不确定"]
ACTION_CHOICES = [
    "减速",
    "停车/等待",
    "向左避让",
    "向右避让",
    "低速观察通过",
    "限制加速/保持车距",
    "保持正常行驶",
    "其他/无法归类",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="关键 Actor 人工盲评标注界面")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--annotator-id", default="", help="可选：预填标注员编号")
    parser.add_argument("--host", default=None, help="覆盖 app.host")
    parser.add_argument("--port", type=int, default=None, help="覆盖 app.port")
    parser.add_argument("--share", action="store_true", help="启用 Gradio share")
    return parser.parse_args()


def build_app(config_path: str, default_annotator_id: str = "") -> gr.Blocks:
    config = load_config(config_path)
    samples = load_jsonl(config.dataset.manifest_path)
    if not samples:
        raise RuntimeError(
            f"manifest 为空或不存在：{config.dataset.manifest_path}\n"
            "请先运行 prepare_samples.py。"
        )
    if any(
        int(sample.get("schema_version", 1)) < 2
        or int(sample.get("history_frames", 0)) < 20
        or int(sample.get("future_frames", 0)) < 20
        for sample in samples
    ):
        raise RuntimeError(
            "检测到第一版或不足 4 秒的 manifest。请先运行：\n"
            "  python upgrade_config_v2.py --config config.yaml\n"
            "  python prepare_samples.py --config config_v2.yaml\n"
            "然后使用 config_v2.yaml 启动界面。"
        )

    def _progress(index: int, annotator_id: str) -> str:
        if not annotator_id.strip():
            return f"样本总数：**{len(samples)}**。请先输入标注员编号并开始。"
        latest = load_latest_annotations(config.dataset.annotation_dir, annotator_id)
        completed = len(latest)
        return (
            f"进度：**{completed}/{len(samples)}**　｜　"
            f"当前序号：**{index + 1}/{len(samples)}**"
        )

    def _sample_metadata(sample: Dict) -> str:
        truncated = int(sample.get("candidate_filter_stats", {}).get("truncated", 0))
        warning = ""
        if truncated:
            warning = f"　｜　⚠ 距离筛选后仍有 {truncated} 个对象因候选上限未展示"
        return (
            f"样本：`{sample['sample_id']}`　｜　路线：`{sample.get('route_name', '')}`"
            f"　｜　中心帧：`{sample.get('center_stem', '')}`{warning}"
        )

    def _form_values(sample: Dict, annotator_id: str) -> Tuple:
        previous = load_latest_annotations(config.dataset.annotation_dir, annotator_id).get(
            sample["sample_id"]
        )
        choices = [actor["display_label"] for actor in sample.get("candidates", [])]
        if previous:
            return (
                gr.update(choices=choices, value=previous.get("selected_display_labels", [])),
                previous.get("scene_judgement"),
                previous.get("removal_changes_action"),
                previous.get("action_effect"),
                int(previous.get("confidence", 3) or 3),
                previous.get("notes", ""),
                "已加载该样本此前的标注；再次提交会保存为新修订。",
            )
        return (
            gr.update(choices=choices, value=[]),
            None,
            None,
            None,
            3,
            "",
            "请先观看短片，再根据中心帧 BEV 中的匿名编号作答。",
        )

    def load_sample(index: int, annotator_id: str):
        try:
            annotator_id = sanitize_annotator_id(annotator_id)
        except ValueError as exc:
            return (
                index,
                time.time(),
                "",
                None,
                None,
                [],
                gr.update(choices=[], value=[]),
                None,
                None,
                None,
                3,
                "",
                f"错误：{exc}",
                f"样本总数：**{len(samples)}**。",
            )

        index = min(max(int(index), 0), len(samples) - 1)
        sample = samples[index]
        assets = render_sample_assets(sample, config)
        form = _form_values(sample, annotator_id)
        return (
            index,
            time.time(),
            _sample_metadata(sample),
            assets["video"] or None,
            assets["current"],
            assets["top"],
            assets["table"],
            *form,
            _progress(index, annotator_id),
        )

    def start_session(annotator_id: str):
        try:
            annotator_id = sanitize_annotator_id(annotator_id)
        except ValueError as exc:
            return (
                0,
                time.time(),
                "",
                None,
                None,
                None,
                [],
                gr.update(choices=[], value=[]),
                None,
                None,
                None,
                3,
                "",
                f"错误：{exc}",
                f"样本总数：**{len(samples)}**。",
            )
        index = choose_next_index(samples, config.dataset.annotation_dir, annotator_id, 0)
        return load_sample(index, annotator_id)

    def validate_answer(
        sample: Dict,
        selected_labels: List[str],
        scene_judgement: str,
        removal_effect: str,
        action_effect: str,
    ) -> str:
        valid_labels = {actor["display_label"] for actor in sample.get("candidates", [])}
        selected_labels = list(selected_labels or [])
        if any(label not in valid_labels for label in selected_labels):
            return "选择中包含当前样本不存在的候选编号。"
        if scene_judgement not in SCENE_VALUES:
            return "请选择当前场景中是否存在直接约束自车动作的关键对象。"
        if scene_judgement == "存在一个主要对象" and len(selected_labels) != 1:
            return "选择“存在一个主要对象”时，必须且只能勾选一个 Actor。"
        if scene_judgement == "多个对象共同影响" and len(selected_labels) < 2:
            return "选择“多个对象共同影响”时，至少勾选两个 Actor。"
        if scene_judgement in {"无关键对象", "无法判断"} and selected_labels:
            return "选择“无关键对象”或“无法判断”时，请取消所有 Actor 勾选。"
        # Negative and ambiguous samples do not have a concrete object to remove.
        # Leaving the two follow-up questions blank reduces unnecessary burden.
        if scene_judgement in {"无关键对象", "无法判断"}:
            return ""
        if removal_effect not in REMOVAL_CHOICES:
            return "请选择“移除对象后动作是否改变”。"
        if action_effect not in ACTION_CHOICES:
            return "请选择主要动作影响。"
        return ""

    def submit_and_next(
        index: int,
        started_at: float,
        annotator_id: str,
        selected_labels: List[str],
        scene_judgement: str,
        removal_effect: str,
        action_effect: str,
        confidence: int,
        notes: str,
    ):
        try:
            annotator_id = sanitize_annotator_id(annotator_id)
        except ValueError as exc:
            result = list(load_sample(index, annotator_id))
            result[-2] = f"错误：{exc}"
            return tuple(result)

        index = min(max(int(index), 0), len(samples) - 1)
        sample = samples[index]
        error = validate_answer(
            sample, selected_labels, scene_judgement, removal_effect, action_effect
        )
        if error:
            result = list(load_sample(index, annotator_id))
            # Preserve user's unsaved input while showing validation error.
            result[7] = gr.update(
                choices=[actor["display_label"] for actor in sample.get("candidates", [])],
                value=selected_labels or [],
            )
            result[8] = scene_judgement
            result[9] = removal_effect
            result[10] = action_effect
            result[11] = confidence
            result[12] = notes
            result[13] = f"未保存：{error}"
            result[1] = started_at
            return tuple(result)

        by_label = {actor["display_label"]: actor for actor in sample.get("candidates", [])}
        selected = [by_label[label] for label in selected_labels or []]
        record = {
            "sample_id": sample["sample_id"],
            "route_name": sample.get("route_name", ""),
            "center_stem": sample.get("center_stem", ""),
            "scene_judgement": scene_judgement,
            "selected_display_labels": list(selected_labels or []),
            "selected_actor_ids": [str(actor["actor_id"]) for actor in selected],
            "selected_actor_classes": [str(actor["class"]) for actor in selected],
            "removal_changes_action": removal_effect,
            "action_effect": action_effect,
            "confidence": int(confidence),
            "notes": notes.strip(),
            "annotation_time_s": round(max(0.0, time.time() - float(started_at)), 3),
            "is_skipped": False,
            "skip_reason": "",
        }
        append_annotation(config.dataset.annotation_dir, annotator_id, record)
        next_index = choose_next_index(
            samples,
            config.dataset.annotation_dir,
            annotator_id,
            start_index=index + 1,
        )
        result = list(load_sample(next_index, annotator_id))
        result[13] = f"已保存 `{sample['sample_id']}`，进入下一条未完成样本。"
        return tuple(result)

    def skip_and_next(index: int, started_at: float, annotator_id: str, notes: str):
        annotator_id = sanitize_annotator_id(annotator_id)
        index = min(max(int(index), 0), len(samples) - 1)
        sample = samples[index]
        append_annotation(
            config.dataset.annotation_dir,
            annotator_id,
            {
                "sample_id": sample["sample_id"],
                "route_name": sample.get("route_name", ""),
                "center_stem": sample.get("center_stem", ""),
                "scene_judgement": "跳过",
                "selected_display_labels": [],
                "selected_actor_ids": [],
                "selected_actor_classes": [],
                "removal_changes_action": "",
                "action_effect": "",
                "confidence": None,
                "notes": notes.strip(),
                "annotation_time_s": round(max(0.0, time.time() - float(started_at)), 3),
                "is_skipped": True,
                "skip_reason": notes.strip() or "人工跳过",
            },
        )
        next_index = choose_next_index(
            samples,
            config.dataset.annotation_dir,
            annotator_id,
            start_index=index + 1,
        )
        result = list(load_sample(next_index, annotator_id))
        result[13] = f"已跳过 `{sample['sample_id']}`。"
        return tuple(result)

    def previous_sample(index: int, annotator_id: str):
        return load_sample(max(0, int(index) - 1), annotator_id)

    def next_sample_without_save(index: int, annotator_id: str):
        return load_sample(min(len(samples) - 1, int(index) + 1), annotator_id)

    with gr.Blocks(title=config.app.title) as demo:
        gr.Markdown(
            "# 关键 Actor 人工盲评标注\n"
            "请根据连续六视角画面和中心帧 BEV 判断。界面不会显示 LG 的选择结果。"
        )
        with gr.Row():
            annotator_id = gr.Textbox(
                label="标注员编号",
                value=default_annotator_id,
                placeholder="例如 P01",
                scale=3,
            )
            start_button = gr.Button("开始 / 继续", variant="primary", scale=1)
        progress = gr.Markdown(f"样本总数：**{len(samples)}**。请先开始。")
        sample_meta = gr.Markdown()

        context_video = gr.Video(
            label="连续六视角场景（约 4 秒：中心帧前约 2 秒 + 后约 2 秒）",
            autoplay=False,
        )
        with gr.Row():
            current_image = gr.Image(label="中心帧六视角", type="filepath")
            top_image = gr.Image(label="中心帧 Top RGB（匿名候选编号）", type="filepath")

        candidate_df = gr.Dataframe(
            headers=["编号", "类别", "前向 x/m", "右向 y/m", "距离/m", "速度/(m/s)"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            visible=config.app.show_candidate_table,
            label="候选对象信息（编号已随机化，不代表距离排序）",
        )

        gr.Markdown(
            "## 标注问题\n"
            "> **关键对象的判断标准：** 假设某个交通参与者不存在，"
            "如果自车接下来约 2 秒内的安全动作（减速、停车、避让或保持车距等）"
            "会发生明显变化，那么该对象可视为关键对象。红绿灯、道路结构和交通规则本身不属于 Actor。"
        )
        scene_judgement = gr.Radio(
            SCENE_CHOICES,
            label="1. 自车接下来约 2 秒的安全动作，是否受到交通参与者的直接约束？",
        )
        selected_labels = gr.CheckboxGroup([], label="2. 请选择关键 Actor 编号")
        removal_effect = gr.Radio(
            REMOVAL_CHOICES,
            label="3. 假设所选对象不存在，自车接下来的安全动作是否会改变？",
        )
        action_effect = gr.Radio(ACTION_CHOICES, label="4. 所选对象主要引起什么动作？")
        confidence = gr.Slider(1, 5, value=3, step=1, label="5. 判断置信度（1 低—5 高）")
        notes = gr.Textbox(
            label="备注（可选；跳过时可填写原因）",
            lines=2,
            visible=config.app.allow_notes,
        )
        status = gr.Markdown("请先观看场景。")

        with gr.Row():
            previous_button = gr.Button("上一条")
            skip_button = gr.Button("跳过")
            next_button = gr.Button("下一条（不保存）")
            submit_button = gr.Button("保存并进入下一条", variant="primary")

        index_state = gr.State(0)
        started_at_state = gr.State(time.time())

        outputs = [
            index_state,
            started_at_state,
            sample_meta,
            context_video,
            current_image,
            top_image,
            candidate_df,
            selected_labels,
            scene_judgement,
            removal_effect,
            action_effect,
            confidence,
            notes,
            status,
            progress,
        ]

        start_button.click(start_session, inputs=[annotator_id], outputs=outputs)
        previous_button.click(
            previous_sample,
            inputs=[index_state, annotator_id],
            outputs=outputs,
        )
        next_button.click(
            next_sample_without_save,
            inputs=[index_state, annotator_id],
            outputs=outputs,
        )
        submit_button.click(
            submit_and_next,
            inputs=[
                index_state,
                started_at_state,
                annotator_id,
                selected_labels,
                scene_judgement,
                removal_effect,
                action_effect,
                confidence,
                notes,
            ],
            outputs=outputs,
        )
        skip_button.click(
            skip_and_next,
            inputs=[index_state, started_at_state, annotator_id, notes],
            outputs=outputs,
        )

        if default_annotator_id.strip():
            demo.load(start_session, inputs=[annotator_id], outputs=outputs)

    return demo


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    demo = build_app(args.config, args.annotator_id)
    host = args.host or config.app.host
    port = args.port or config.app.port
    share = bool(args.share or config.app.share)
    demo.queue(default_concurrency_limit=8).launch(
        server_name=host,
        server_port=port,
        share=share,
        allowed_paths=[str(config.dataset.work_dir), str(config.dataset.dataset_root)],
    )


if __name__ == "__main__":
    main()
