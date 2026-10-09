#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查样本 Top RGB 投影几何和可视范围，不修改任何数据。"""
import argparse
import json
from pathlib import Path

from annotation_core.config import load_config
from annotation_core.dataset import load_jsonl, resolve_manifest_path
from annotation_core.render import _read_image, _derive_meters_per_pixel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.scheme_b.yaml')
    ap.add_argument('--limit', type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    samples = load_jsonl(cfg.dataset.manifest_path)
    totals = {'in_view': 0, 'outside_view': 0, 'missing_xy': 0}
    camera_summaries = []
    for s in samples:
        route = resolve_manifest_path(s.get('route_dir'), cfg.dataset.dataset_root)
        frame = s['frames'][int(s['center_frame_offset'])]
        image_path = resolve_manifest_path(frame.get('top_rgb'), cfg.dataset.dataset_root)
        im = _read_image(image_path)
        if im is None:
            raise RuntimeError('Top RGB not found: %s' % image_path)
        H, W = im.shape[:2]
        mpp = _derive_meters_per_pixel(im, route, cfg)
        meta_path = route / 'surround_camera_config.json'
        camera = {}
        if meta_path.is_file():
            camera = (json.loads(meta_path.read_text(encoding='utf-8')).get('top_rgb') or {})
        cx = cfg.render.top_ego_center_x_ratio * W
        cy = cfg.render.top_ego_center_y_ratio * H
        current = dict.fromkeys(totals, 0)
        shown = []
        for a in s.get('candidates', []):
            x, y = a.get('x_m'), a.get('y_m')
            if x is None or y is None:
                current['missing_xy'] += 1
                continue
            u = cx + (1 if cfg.render.right_is_right else -1) * float(y) / mpp
            v = cy + (-1 if cfg.render.forward_is_up else 1) * float(x) / mpp
            if 0 <= u < W and 0 <= v < H:
                current['in_view'] += 1
                if len(shown) < 5:
                    shown.append('%s:(x=%.1f,y=%.1f)->pixel=(%.0f,%.0f)' % (a['display_label'],x,y,u,v))
            else:
                current['outside_view'] += 1
        for k in totals:
            totals[k] += current[k]
        if len(camera_summaries) < args.limit:
            camera_summaries.append({
                'sample_id': s['sample_id'], 'image_wh': [W,H],
                'camera_pos': camera.get('position'), 'camera_rot': camera.get('rotation'),
                'camera_fov': camera.get('fov'), 'meters_per_pixel': round(mpp, 5),
                'ego_pixel': [round(cx,1),round(cy,1)],
                'actors': current, 'sample_actors': shown
            })
    print(json.dumps({'sample_count':len(samples),'total_actor_instances': totals,
                      'preview': camera_summaries},indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()
