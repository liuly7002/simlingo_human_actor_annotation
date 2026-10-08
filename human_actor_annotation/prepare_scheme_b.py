#!/usr/bin/env python3
"""147 LG causal + 147 preseeded LG no-causal frames, blinded manifest. #修改20260720"""
from __future__ import annotations
import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

from annotation_core.actors import load_normalized_actors, select_candidates
from annotation_core.config import load_config
from annotation_core.dataset import build_route_index, _relative_or_absolute, VIEW_ORDER
from scheme_b_common import (parse_keyframes, load_lg, load_cvaa, key_to_sample_id,
                             write_jsonl, write_json, write_csv)


def _file_index(index, stem):
    return {view: index['view_files'][view].get(stem) for view in VIEW_ORDER}


def prepare_single(key, group, dataset_root, cfg, cache, min_real_history):
    route_rel, frame = key
    route_dir = (dataset_root / route_rel).resolve()
    try:
        route_dir.relative_to(dataset_root)
    except ValueError:
        return None, 'route_outside_data_root'
    if not route_dir.is_dir():
        return None, 'route_missing'
    try:
        if route_rel not in cache:
            cache[route_rel] = build_route_index(route_dir, cfg)
        idx = cache[route_rel]
    except Exception as exc:
        return None, 'route_index_failed:' + type(exc).__name__
    if frame not in idx['stems']:
        return None, 'center_image_or_metadata_missing'
    bpath = idx['boxes_files'].get(frame)
    if bpath is None:
        return None, 'center_boxes_missing'

    try:
        actors = load_normalized_actors(bpath)
        candidates, stats = select_candidates(
            actors, cfg.sampling.include_classes, cfg.sampling.exclude_classes,
            cfg.sampling.max_actor_distance_m, cfg.sampling.min_actor_forward_m,
            cfg.sampling.max_candidates, key_to_sample_id(key), cfg.sampling.random_seed)
    except Exception as exc:
        return None, 'candidate_parse_failed:' + type(exc).__name__
    if stats['truncated'] and cfg.sampling.skip_if_candidates_truncated:
        return None, 'candidate_truncated'
    if not candidates and not cfg.sampling.include_empty_candidate_samples:
        return None, 'no_annotation_candidates'

    # Always expose 40 *time slots* before t0. If frames are absent, make blank
    # slots instead of replaying a fake previous frame or leaking future frames.
    n = cfg.sampling.history_frames
    if n != 40 or cfg.sampling.future_frames != 0:
        raise ValueError('Existing annotation_app.py requires history_frames=40 and future_frames=0')
    stems = []
    if frame.isdigit():
        i = int(frame)
        for k in range(i - n, i + 1):
            stems.append(('%0*d' % (len(frame), k)) if k >= 0 else None)
    else:
        all_stems = sorted(idx['stems'])
        position = all_stems.index(frame)
        history = all_stems[max(0, position - n):position + 1]
        stems = [None] * (n + 1 - len(history)) + history
    frames = []
    actual_history = 0
    for offset, stem in enumerate(stems):
        views = _file_index(idx, stem) if stem else {}
        complete = bool(stem) and all(views.get(v) is not None for v in VIEW_ORDER)
        if offset == n and not complete:
            return None, 'center_six_views_missing'
        if offset != n and complete:
            actual_history += 1
        if not complete:
            placeholder = str(cfg.dataset.work_dir / 'scheme_b_unavailable_history.png')
            frames.append({'stem': '__HISTORY_UNAVAILABLE_%02d__' % offset,
                           'views': {v: placeholder for v in VIEW_ORDER}, 'top_rgb': None})
        else:
            frames.append({'stem': stem,
                           'views': {v: _relative_or_absolute(views[v], dataset_root) for v in VIEW_ORDER},
                           'top_rgb': _relative_or_absolute(idx['top_files'].get(stem), dataset_root)})
    if actual_history < min_real_history:
        return None, 'insufficient_actual_history'
    candidate_clean = []
    for actor in candidates:
        candidate_clean.append({k: actor.get(k) for k in (
            'actor_id', 'class', 'raw_class', 'x_m', 'y_m', 'distance_m',
            'speed_mps', 'role_name', 'extent', 'display_label')})
    # A route without full 4s history shows visually blank unavailable slots.
    # Method group is stored for evaluation only, never rendered in the UI.
    rec = {'schema_version': 1, 'sample_id': key_to_sample_id(key),
           'route_name': route_rel.replace('/', '__'), 'route_dir': route_rel,
           'center_stem': frame, 'center_index': int(frame) if frame.isdigit() else 0,
           'history_frames': 40, 'future_frames': 0, 'frames': frames,
           'center_frame_offset': 40,
           'boxes_path': _relative_or_absolute(bpath, dataset_root),
           'measurement_path': _relative_or_absolute(idx['measurement_files'].get(frame), dataset_root),
           'candidates': candidate_clean, 'candidate_filter_stats': stats,
           'blind_annotation': True, 'evaluation_group': group,
           'real_history_frames': actual_history,
           'unavailable_history_frames': 40 - actual_history}
    return rec, None


def make_scheme_b(cfg, keyframes, lg_frames, cvaa_frames, target=147, seed=20261008,
                  min_real_history=0, require_cvaa=True):
    keys = parse_keyframes(keyframes)
    lg = load_lg(lg_frames, keys)
    cvaa = load_cvaa(cvaa_frames, keys)
    positives = [k for k in keys if k in lg and lg[k]['has_causal_object']]
    negatives = [k for k in keys if k in lg and not lg[k]['has_causal_object']]
    if len(positives) != target:
        raise ValueError('LG confirmed causal frames=%d, expected exactly %d. Check LG file/manifest; do not silently truncate positives.' % (len(positives), target))
    if require_cvaa:
        missing_pos = [k for k in positives if not cvaa.get(k, {}).get('available')]
        if missing_pos:
            raise ValueError('%d LG-positive frames lack CVAA top actor; see CVAA output completeness first. Examples: %s' % (len(missing_pos), missing_pos[:3]))
        negatives = [k for k in negatives if cvaa.get(k, {}).get('available')]

    root = cfg.dataset.dataset_root.resolve()
    # Render an explicit unavailable-history card rather than pretending that
    # the first available camera image was captured before recording began.
    import cv2
    import numpy as np
    blank_path = cfg.dataset.work_dir / 'scheme_b_unavailable_history.png'
    if not blank_path.exists():
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, 'HISTORY NOT AVAILABLE', (65, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
        if not cv2.imwrite(str(blank_path), blank):
            raise IOError('Could not create unavailable history card: %s' % blank_path)
    rng = random.Random(seed)
    rng.shuffle(negatives)
    cache = {}
    samples = []
    rejected = []
    pos_ok = []
    for k in positives:
        sample, why = prepare_single(k, 'lg_positive', root, cfg, cache, min_real_history)
        if sample is None:
            rejected.append({'route_rel': k[0], 'frame': k[1], 'group': 'lg_positive', 'reason': why})
        else:
            pos_ok.append(sample)
    if len(pos_ok) != target:
        summary = {'lg_confirmed_in_source': len(positives), 'lg_annotatable': len(pos_ok),
                   'target': target, 'rejections': rejected}
        write_json(cfg.dataset.work_dir / 'scheme_b_preflight_error.json', summary)
        raise ValueError('Only %d/%d confirmed LG frames annotatable: %s. Do not silently substitute positive frames.' %
                         (len(pos_ok), target, cfg.dataset.work_dir / 'scheme_b_preflight_error.json'))
    samples.extend(pos_ok)
    neg_ok = []
    for k in negatives:
        if len(neg_ok) == target:
            break
        sample, why = prepare_single(k, 'lg_negative_random', root, cfg, cache, min_real_history)
        if sample is None:
            rejected.append({'route_rel': k[0], 'frame': k[1], 'group': 'lg_negative_random', 'reason': why})
        else:
            neg_ok.append(sample)
    if len(neg_ok) < target:
        raise ValueError('Only %d/%d eligible LG-negative frames after exhausting candidates' % (len(neg_ok), target))
    samples.extend(neg_ok)
    rng.shuffle(samples)
    output_preds = []
    mapping_audit = []
    for sample in samples:
        k = (sample['route_dir'], sample['center_stem'])
        cid = set(str(c['actor_id']) for c in sample['candidates'])
        l = lg.get(k, {})
        c = cvaa.get(k, {})
        row = {'sample_id': sample['sample_id'], 'route_rel': k[0], 'frame': k[1],
               'group': sample['evaluation_group'],
               'lg_available': bool(l.get('available')),
               'lg_selected_actor_id': l.get('selected_actor_id'),
               'lg_has_causal_object': bool(l.get('has_causal_object')),
               'cvaa_available': bool(c.get('available')),
               'cvaa_selected_actor_id': c.get('selected_actor_id'),
               'cvaa_num_actors': c.get('num_actors')}
        output_preds.append(row)
        for method in ('lg', 'cvaa'):
            actor = row[method + '_selected_actor_id']
            if actor is not None and actor not in cid:
                mapping_audit.append({'sample_id': sample['sample_id'],
                                      'group': sample['evaluation_group'],
                                      'method': method, 'selected_actor_id': actor,
                                      'reason': 'actor_not_in_method_neutral_annotation_candidates'})
    # Prevent accidental relabeling of an existing blinded annotation set.
    existing = list(cfg.dataset.annotation_dir.glob('annotations_*.jsonl'))
    if existing:
        raise FileExistsError(
            'Existing human annotations found. Do not regenerate/swap the sampled manifest '
            'in-place after annotation has begun: %s' % existing[0])
    write_jsonl(cfg.dataset.manifest_path, samples)
    output_dir = cfg.dataset.work_dir / 'evaluation'
    write_jsonl(output_dir / 'scheme_b_predictions.jsonl', output_preds)
    write_csv(output_dir / 'scheme_b_actor_mapping_audit.csv', mapping_audit)
    write_csv(output_dir / 'scheme_b_rejected_frames.csv', rejected)
    summary = {
        'keyframes_total': len(keys), 'lg_frame_records': len(lg),
        'cvaa_ranked_frames': len(cvaa), 'lg_confirmed_in_source': len(positives),
        'selected_lg_positive': len(pos_ok), 'selected_lg_negative': len(neg_ok),
        'total_samples': len(samples), 'random_seed': seed,
        'min_real_history': min_real_history,
        'require_cvaa': require_cvaa,
        'missing_history_samples': sum(s['unavailable_history_frames'] > 0 for s in samples),
        'actor_mapping_mismatches': len(mapping_audit),
        'source_sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (keyframes, lg_frames, cvaa_frames)},
        'manifest_path': str(cfg.dataset.manifest_path),
        'prediction_path': str(output_dir / 'scheme_b_predictions.jsonl'),
        'rejection_reasons': dict(Counter(r['reason'] for r in rejected))}
    write_json(output_dir / 'scheme_b_prepare_summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description='Scheme B: 147 + 147 blinded annotations')
    p.add_argument('--config', required=True)
    p.add_argument('--keyframes', required=True)
    p.add_argument('--lg-frames', required=True, help='LG all_frame_results.jsonl')
    p.add_argument('--cvaa-rankings', required=True, help='CVAA all_frame_rankings.jsonl')
    p.add_argument('--target', type=int, default=147)
    p.add_argument('--seed', type=int, default=20261008)
    p.add_argument('--min-real-history', type=int, default=0, help='0=retain early frames but mark absent history as blank')
    p.add_argument('--allow-missing-cvaa', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    result = make_scheme_b(cfg, args.keyframes, args.lg_frames, args.cvaa_rankings,
                           args.target, args.seed, args.min_real_history,
                           not args.allow_missing_cvaa)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
