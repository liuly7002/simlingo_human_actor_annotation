#!/usr/bin/env python3
import importlib.util
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

src_path = Path(__file__).resolve().parents[1] / 'apply_top_rgb_fix.py'
spec = importlib.util.spec_from_file_location('top_patch', str(src_path))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

STUB = '''from __future__ import annotations
from typing import Dict
import numpy as np
import cv2
class ProjectConfig: pass
def annotate_top_rgb(sample: Dict, config: ProjectConfig) -> np.ndarray:
    return None

def candidate_table(sample):
    return []

def _render_signature(sample: Dict, config: ProjectConfig) -> str:
    payload = {
        "version": 3,
        "render": {
            "top_meters_per_pixel": config.render.top_meters_per_pixel,
        },
    }
    return str(payload)

def render_sample_assets(sample: Dict, config: ProjectConfig):
    return None
'''
new = p.patch(STUB)
assert new == p.patch(new), 'Patcher must be idempotent'
assert "\"version\": 4" in new
assert 'top_ego_center_y_ratio' in new
assert '编号中心即 actor' in new
compile(new,'patched_render.py','exec')

namespace={}
exec(new,namespace)
raw = np.zeros((512,512,3),dtype=np.uint8)
namespace['_read_image']=lambda _: raw.copy()
namespace['_derive_meters_per_pixel']=lambda *a: 80.0/512.0
namespace['resolve_manifest_path']=lambda s, root: Path(root)/str(s)
R=types.SimpleNamespace
cfg=R(dataset=R(dataset_root=Path('/tmp')), render=R(
    top_ego_center_x_ratio=0.5,top_ego_center_y_ratio=0.5,
    right_is_right=True,forward_is_up=True,marker_radius_px=18,
    top_preview_width=512,top_meters_per_pixel=None))
sample={'center_frame_offset':0,'frames':[{'top_rgb':'top_rgb/0001.jpg'}],
        'route_dir':'sample','candidates':[
            {'display_label':'A1','x_m':10.0,'y_m':0.0},
            {'display_label':'A2','x_m':0.0,'y_m':10.0},
            {'display_label':'A3','x_m':120.0,'y_m':0.0}]}
img=namespace['annotate_top_rgb'](sample,cfg)
assert img.shape==(512,512,3)
# A1 ahead (center x, 64px up), A2 right (64px to right).
assert (img[192,256] != np.array([0,0,0])).any(), 'A1 at true projected pixel'
assert (img[256,320] != np.array([0,0,0])).any(), 'A2 at true projected pixel'
assert (img[256,256] != np.array([0,0,0])).any(), 'EGO at optical center'
# Far A3 must not be clipped onto visible border.
assert (img[60,256] == np.array([0,0,0])).all(), 'outside top camera should not be pinned to edge'
print('PASS: patch idempotent; syntax; exact actor center; ego at 50/50; out-of-FOV not mislabeled; cache signature fields')
