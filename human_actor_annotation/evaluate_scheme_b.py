#!/usr/bin/env python3
"""Paired LG/CVAA human-consensus evaluation on Scheme B. #修改20260720"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from annotation_core.config import load_config
from annotation_core.dataset import load_jsonl
from annotation_core.metrics import (read_annotation_dir, build_consensus,
                                      pairwise_agreement, krippendorff_alpha_binary)
from scheme_b_common import jsonl, write_csv, write_json


def _percentile(v, q):
    if not v:
        return None
    xs = sorted(v)
    index = (len(xs) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (index - lo)


def _mcnemar_exact(b, c):
    # Two-sided binomial exact test of discordant paired outcomes.
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, j) for j in range(k + 1)) / float(2 ** n)
    return min(1.0, p)


def _cluster_ci(rows, seed=20261008, rounds=2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['route_rel']].append(row)
    routes = sorted(grouped)
    if not routes:
        return [None, None]
    rng = random.Random(seed)
    scores = []
    for _ in range(rounds):
        take = [rng.choice(routes) for _ in routes]
        picked = [r for k in take for r in grouped[k]]
        scores.append(sum(int(r['lg_correct']) - int(r['cvaa_correct']) for r in picked) / len(picked))
    return [_percentile(scores, 0.025), _percentile(scores, 0.975)]


def _metrics(rows):
    n = len(rows)
    lg = sum(bool(x['lg_correct']) for x in rows)
    cvaa = sum(bool(x['cvaa_correct']) for x in rows)
    b = sum(bool(x['lg_correct']) and not bool(x['cvaa_correct']) for x in rows)
    c = sum(bool(x['cvaa_correct']) and not bool(x['lg_correct']) for x in rows)
    return {
        'n': n, 'lg_correct': lg, 'cvaa_correct': cvaa,
        'lg_accuracy': lg / n if n else None,
        'cvaa_accuracy': cvaa / n if n else None,
        'paired_difference_lg_minus_cvaa': (lg - cvaa) / n if n else None,
        'lg_only_correct': b, 'cvaa_only_correct': c,
        'exact_mcnemar_p': _mcnemar_exact(b, c) if n else None,
        'route_cluster_bootstrap_difference_95ci': _cluster_ci(rows) if n else [None, None],
        'distinct_routes': len(set(x['route_rel'] for x in rows))
    }


def evaluate(cfg, predictions_path, outdir=None, min_decisive=3):
    manifest = load_jsonl(cfg.dataset.manifest_path)
    ids = [str(x['sample_id']) for x in manifest]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample_id in annotation manifest')
    if not manifest:
        raise ValueError('Empty annotation manifest')
    sample_by_id = {str(x['sample_id']): x for x in manifest}
    preds = {}
    for row in jsonl(predictions_path):
        sid = str(row['sample_id'])
        if sid in preds:
            raise ValueError('Duplicated prediction sample_id: ' + sid)
        preds[sid] = row
    if set(preds) != set(ids):
        raise ValueError('Prediction/manifest sample_id mismatch: only_manifest=%s, only_predictions=%s' %
                         (sorted(set(ids)-set(preds))[:3], sorted(set(preds)-set(ids))[:3]))
    records = [r for r in read_annotation_dir(cfg.dataset.annotation_dir)
               if str(r.get('sample_id')) in sample_by_id]
    consensus = {str(x['sample_id']): x for x in build_consensus(records, threshold=0.5)}
    rows = []
    for sid in ids:
        manifest_row = sample_by_id[sid]
        pred = preds[sid]
        expected_group = ('lg_positive' if pred.get('lg_has_causal_object') else 'lg_negative_random')
        if pred.get('group') != expected_group or manifest_row.get('evaluation_group') != expected_group:
            raise ValueError('Group mismatch at %s' % sid)
        h = consensus.get(sid, {})
        status = str(h.get('consensus_status', 'unannotated'))
        n_decisive = int(h.get('n_decisive_annotators', 0))
        human_set = set(str(x) for x in h.get('consensus_actor_ids', []))
        valid = status in ('actors', 'none') and n_decisive >= min_decisive
        lg_avail = bool(pred.get('lg_available'))
        cvaa_avail = bool(pred.get('cvaa_available'))
        a = pred.get('lg_selected_actor_id')
        b = pred.get('cvaa_selected_actor_id')
        # Null LG output is a valid *decision* to abstain when its record exists;
        # missing CVAA ranking is an unavailable prediction, NOT no-actor.
        def correct(actor, available):
            if not valid or not available:
                return None
            if status == 'none':
                return actor is None
            return actor is not None and str(actor) in human_set
        lg_hit = correct(a, lg_avail)
        cvaa_hit = correct(b, cvaa_avail)
        primary = h.get('consensus_primary_actor_id')
        choices = set(str(x['actor_id']) for x in manifest_row['candidates'])
        rows.append({
            'sample_id': sid, 'route_rel': pred['route_rel'],
            'frame': pred['frame'], 'group': pred['group'],
            'human_consensus_status': status,
            'n_decisive_annotators': n_decisive,
            'n_annotators': int(h.get('n_annotators', 0)),
            'human_actor_ids': sorted(human_set), 'human_primary_actor_id': primary,
            'adjudication_required': bool(h.get('needs_adjudication', True)),
            'eligible_human': valid,
            'lg_available': lg_avail, 'cvaa_available': cvaa_avail,
            'lg_selected_actor_id': a, 'cvaa_selected_actor_id': b,
            'lg_in_annotation_candidates': None if a is None else str(a) in choices,
            'cvaa_in_annotation_candidates': None if b is None else str(b) in choices,
            'lg_correct': lg_hit, 'cvaa_correct': cvaa_hit,
            'strict_human_primary': primary is not None and valid,
            'lg_strict_match': str(a) == str(primary) if valid and primary is not None and lg_avail else None,
            'cvaa_strict_match': str(b) == str(primary) if valid and primary is not None and cvaa_avail else None,
            'real_history_frames': manifest_row.get('real_history_frames', 40),
        })
    paired = [r for r in rows if r['eligible_human'] and r['lg_available'] and r['cvaa_available']]
    human_positive = [r for r in paired if r['human_consensus_status'] == 'actors']
    human_none = [r for r in paired if r['human_consensus_status'] == 'none']
    human_primary = [r for r in paired if r['strict_human_primary']]
    strict = []
    for r in human_primary:
        strict.append(dict(r, lg_correct=r['lg_strict_match'], cvaa_correct=r['cvaa_strict_match']))

    summary = {
        'experimental_design': 'Scheme B, method-blind 147 confirmed LG + 147 seeded nonconfirmed LG',
        'manifest_samples': len(rows),
        'manifest_by_group': dict(Counter(r['group'] for r in rows)),
        'annotated_samples': sum(r['n_annotators'] > 0 for r in rows),
        'human_evaluable_samples': sum(r['eligible_human'] for r in rows),
        'human_actors': sum(r['human_consensus_status'] == 'actors' and r['eligible_human'] for r in rows),
        'human_none': sum(r['human_consensus_status'] == 'none' and r['eligible_human'] for r in rows),
        'method_prediction_coverage_on_human_evaluable': {
            m: {'available': sum(r[m + '_available'] for r in rows if r['eligible_human']),
                'total': sum(r['eligible_human'] for r in rows)} for m in ('lg', 'cvaa')},
        'common_paired_evaluable': len(paired),
        'overall_correct_including_human_none': _metrics(paired),
        'human_positive_hit_at_1': _metrics(human_positive),
        'human_none_specificity': _metrics(human_none),
        'human_unique_primary_strict_top1': _metrics(strict),
        'groups': {g: _metrics([r for r in paired if r['group'] == g])
                   for g in ('lg_positive', 'lg_negative_random')},
        'unresolved_actor_ids_in_evaluable': {
            m: sum(r[m + '_available'] and r[m + '_selected_actor_id'] is not None
                   and r[m + '_in_annotation_candidates'] is False for r in rows if r['eligible_human'])
            for m in ('lg', 'cvaa')},
        'short_history_evaluable_samples': sum(r['eligible_human'] and r['real_history_frames'] < 40 for r in rows),
        'annotation_rule': {'min_decisive_annotators': min_decisive, 'consensus_threshold_strictly_greater_than': 0.5},
        'caveat': 'Balanced conditional sample (LG positive/negative), not population-weighted accuracy.'
    }
    if records:
        human_agreement = pairwise_agreement(records)
        human_agreement['krippendorff_alpha_binary'] = krippendorff_alpha_binary(records, manifest)
        summary['inter_annotator_agreement'] = human_agreement
    outdir = Path(outdir) if outdir else cfg.dataset.work_dir / 'evaluation'
    write_csv(outdir / 'scheme_b_human_comparison.csv', rows)
    write_csv(outdir / 'scheme_b_disagreements.csv',
              [r for r in paired if bool(r['lg_correct']) != bool(r['cvaa_correct'])])
    write_csv(outdir / 'scheme_b_non_evaluable.csv', [r for r in rows if r not in paired])
    write_json(outdir / 'scheme_b_evaluation_summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description='Scheme B paired human comparison')
    p.add_argument('--config', required=True)
    p.add_argument('--predictions', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--min-decisive', type=int, default=3)
    args = p.parse_args()
    cfg = load_config(args.config)
    pred = args.predictions or cfg.dataset.work_dir / 'evaluation' / 'scheme_b_predictions.jsonl'
    result = evaluate(cfg, pred, args.output_dir, args.min_decisive)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
