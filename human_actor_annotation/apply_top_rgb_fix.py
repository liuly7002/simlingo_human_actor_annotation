#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仅修复 annotation_core/render.py 的 Top RGB 显示与缓存，不修改标注数据。"""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

REPLACEMENT = '''def annotate_top_rgb(sample: Dict, config: ProjectConfig) -> np.ndarray:
    """#修改20260720：编号中心即 actor 俯视投影中心，不再通过避让算法移动编号。"""
    dataset_root = config.dataset.dataset_root
    center_offset = int(sample["center_frame_offset"])
    frame = sample["frames"][center_offset]
    top_path = resolve_manifest_path(frame.get("top_rgb"), dataset_root)
    top = _read_image(top_path)
    if top is None:
        top = np.zeros((720, 720, 3), dtype=np.uint8)
        cv2.putText(top, "TOP_RGB MISSING", (50, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)

    route_dir = resolve_manifest_path(sample.get("route_dir"), dataset_root)
    route_dir = route_dir if route_dir is not None else dataset_root
    mpp = _derive_meters_per_pixel(top, route_dir, config)
    height, width = top.shape[:2]
    cx = float(config.render.top_ego_center_x_ratio) * width
    cy = float(config.render.top_ego_center_y_ratio) * height
    right_sign = 1.0 if config.render.right_is_right else -1.0
    forward_sign = -1.0 if config.render.forward_is_up else 1.0

    # 可追溯的物理投影：车辆记录中的 x_m/y_m 均为 ego-relative 坐标。
    # 其正方向来自数据采集端：x 向前、y 向右；Top RGB 镜头竖直朝下。
    in_view = []
    outside_count = 0
    for actor in sample.get("candidates", []):
        x = actor.get("x_m")
        y = actor.get("y_m")
        if x is None or y is None:
            outside_count += 1
            continue
        u = cx + right_sign * float(y) / mpp
        v = cy + forward_sign * float(x) / mpp
        if 0 <= u < width and 0 <= v < height:
            in_view.append((actor, int(round(u)), int(round(v))))
        else:
            outside_count += 1

    # 只在真实的投影像素上绘制，不再将视野外对象钉在边缘伪装为可见目标。
    # 标签和 actor 中心重合，可以直接用俯视相机图像校验。
    for actor, u, v in in_view:
        label = str(actor.get("display_label", "?"))
        radius = max(13, min(18, int(config.render.marker_radius_px)))
        cv2.circle(top, (u, v), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(top, (u, v), radius, (0, 230, 255), -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 2)
        cv2.putText(top, label, (u - tw // 2, v + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 2, cv2.LINE_AA)

    # 相机横向/纵向相对自车均为 0，默认标定下 ego 位于 50%/50%。
    ego_u, ego_v = int(round(cx)), int(round(cy))
    cv2.circle(top, (ego_u, ego_v), 11, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.drawMarker(top, (ego_u, ego_v), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=15,
                   thickness=2, line_type=cv2.LINE_AA)
    cv2.putText(top, "EGO", (max(2, ego_u - 17), max(13, ego_v - 17)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2, cv2.LINE_AA)

    legend = f"EGO=camera center | {mpp:.3f} m/px | outside top view: {outside_count}"
    cv2.rectangle(top, (0, 0), (width, 30), (12, 12, 12), -1)
    cv2.putText(top, legend, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (255, 255, 255), 1, cv2.LINE_AA)
    if outside_count:
        cv2.putText(top, "Other IDs may be outside top camera FOV", (8, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    target_w = int(config.render.top_preview_width)
    if target_w > 0 and top.shape[1] != target_w:
        target_h = max(1, int(top.shape[0] * target_w / top.shape[1]))
        top = cv2.resize(top, (target_w, target_h))
    return top
'''


def patch(content: str) -> str:
    if '"""#修改20260720：编号中心即 actor' in content:
        return content
    start = content.find('def annotate_top_rgb(')
    end = content.find('\ndef candidate_table(', start)
    if start < 0 or end < 0:
        raise RuntimeError('原 render.py 结构与预期不符，停止修改')
    result = content[:start] + REPLACEMENT + '\n\n' + content[end+1:]
    sig_start = result.find('def _render_signature(')
    sig_end = result.find('\ndef render_sample_assets(', sig_start)
    if sig_start < 0 or sig_end < 0:
        raise RuntimeError('未找到缓存签名函数，停止修改')
    section = result[sig_start:sig_end]
    old = '"top_meters_per_pixel": config.render.top_meters_per_pixel,'
    if section.count(old) != 1:
        raise RuntimeError('缓存字段布局改变，停止修改')
    section = section.replace('"version": 3,', '"version": 4,', 1)
    section = section.replace(old,
        old + '\n            "top_ego_center_x_ratio": config.render.top_ego_center_x_ratio,'
              '\n            "top_ego_center_y_ratio": config.render.top_ego_center_y_ratio,'
              '\n            "forward_is_up": config.render.forward_is_up,'
              '\n            "right_is_right": config.render.right_is_right,')
    result = result[:sig_start] + section + result[sig_end:]
    ast.parse(result)
    return result


def main():
    ap = argparse.ArgumentParser(description='修正 Top RGB 俯视标注投影展示')
    ap.add_argument('--render', default='annotation_core/render.py')
    args = ap.parse_args()
    target = Path(args.render)
    original = target.read_text(encoding='utf-8')
    changed = patch(original)
    if original == changed:
        print('已经修复过，无需重复修改：', target)
        return
    backup = target.with_suffix('.py.before_top_rgb_fix')
    if backup.exists():
        raise RuntimeError('备份已存在，防止覆盖，请先检查：%s' % backup)
    shutil.copy2(str(target), str(backup))
    target.write_text(changed, encoding='utf-8')
    print('修正成功：', target)
    print('原文件备份：', backup)
    print('需要在 config.scheme_b.yaml 中设置 top_ego_center_y_ratio: 0.5')
    print('重启 Gradio 后新缓存自动生效；manifest 和预测文件未修改')

if __name__ == '__main__':
    main()
