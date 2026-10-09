"""Standalone synthetic smoke tests for Scheme B without CARLA data."""
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Emulate only the interface of the pre-existing annotation_core; the
# integration uses the repository's *actual* core modules at runtime.
core = types.ModuleType('annotation_core')
core.__path__ = []
sys.modules['annotation_core'] = core
actors = types.ModuleType('annotation_core.actors')
actors.load_normalized_actors = lambda p: [{'actor_id': '11', 'class': 'vehicle', 'display_label': 'A1', 'x_m': 6, 'y_m': 0},
                                           {'actor_id': '22', 'class': 'vehicle', 'display_label': 'A2', 'x_m': 9, 'y_m': 1}]
actors.select_candidates = lambda a, *args: (a, {'truncated': 0})
sys.modules['annotation_core.actors'] = actors
config = types.ModuleType('annotation_core.config')
config.load_config = lambda path: None
sys.modules['annotation_core.config'] = config
dataset = types.ModuleType('annotation_core.dataset')
dataset.VIEW_ORDER = ('front_left', 'front', 'front_right', 'rear_left', 'rear', 'rear_right')

def make_index(route, cfg):
    files = {('%04d' % i): (route / ('%04d.png' % i)) for i in range(41)}
    bfiles = {('%04d' % i): (route / ('%04d.json' % i)) for i in range(41)}
    return {'stems': list(files), 'view_files': {v: files for v in dataset.VIEW_ORDER},
            'boxes_files': bfiles, 'top_files': files, 'measurement_files': {}}

dataset.build_route_index = make_index
dataset._relative_or_absolute = lambda p, root: str(p) if p else None
dataset.load_jsonl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
sys.modules['annotation_core.dataset'] = dataset
metric = types.ModuleType('annotation_core.metrics')
metric._records = []
metric.read_annotation_dir = lambda path: metric._records

def consensus(records, threshold=0.5):
    from collections import defaultdict, Counter
    groups = defaultdict(list)
    for x in records:
        groups[x['sample_id']].append(x)
    output = []
    for sid, items in groups.items():
        vc = Counter(a for i in items for a in i.get('selected_actor_ids', []))
        aset = sorted(a for a, c in vc.items() if c / len(items) > threshold)
        status = 'actors' if aset else 'none'
        output.append(dict(sample_id=sid, consensus_status=status,
                           consensus_actor_ids=aset,
                           consensus_primary_actor_id=aset[0] if len(aset) == 1 else None,
                           n_decisive_annotators=len(items), n_annotators=len(items),
                           needs_adjudication=False))
    return output

metric.build_consensus = consensus
metric.pairwise_agreement = lambda records: {'pairwise_exact_set_agreement': 1.0}
metric.krippendorff_alpha_binary = lambda records, samples: 1.0
sys.modules['annotation_core.metrics'] = metric

from scheme_b_common import parse_keyframes, load_lg, load_cvaa, jsonl, write_jsonl
from prepare_scheme_b import make_scheme_b, recover_lg_id, history_stratum, sample_matched_negatives
from evaluate_scheme_b import evaluate


def test_scheme_b():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = tmp / 'data'
        for r in ('route001', 'route002'):
            (root / r).mkdir(parents=True)
        keys = tmp / 'keyframes.txt'
        lines = ['route001/0010', 'route001/0012', 'route002/0014', 'route002/0016']
        keys.write_text('\n'.join(lines) + '\n')
        assert len(parse_keyframes(keys)) == 4
        lgfile, cfile = tmp / 'lg.jsonl', tmp / 'cvaa.jsonl'
        write_jsonl(lgfile, [{'route_rel': s.split('/')[0], 'frame': s.split('/')[1],
                              'has_causal_object': i < 2,
                              'causal_actor_id': '11' if i == 0 else '22' if i == 1 else None}
                             for i, s in enumerate(lines)])
        write_jsonl(cfile, [{'route_id': s.split('/')[0], 'frame': s.split('/')[1],
                             'top_actor_id': '22', 'num_actors': 2}
                            for s in lines])
        # Actual LG full-output schema uses lg_result.causal_analysis.*.
        full_lg = tmp / 'all_frame_results_full.jsonl'
        write_jsonl(full_lg, [
            {'route_rel': line.split('/')[0], 'frame': line.split('/')[1],
             'lg_result': {'causal_analysis': {
                 'has_causal_object': i < 2,
                 'causal_object': {'id': '11' if i == 0 else '22'} if i < 2 else None}}}
            for i, line in enumerate(lines)
        ])
        parsed_full = load_lg(full_lg, parse_keyframes(keys))
        assert sum(v['has_causal_object'] for v in parsed_full.values()) == 2
        assert parsed_full[('route001', '0010')]['selected_actor_id'] == '11'
        assert parsed_full[('route002', '0014')]['selected_actor_id'] is None
        # A real LG causal actor may be accepted despite missing CARLA actor ID.
        # The LG planning code falls back to the actor's class/ego-relative xy.
        no_id_lg = tmp / 'lg_null_actor_id.jsonl'
        missing_records = []
        for i, line in enumerate(lines):
            route, frame = line.split('/')
            causal = {'has_causal_object': i < 2,
                      'causal_object': ({'id': None if i == 0 else '22',
                                         'class': 'vehicle', 'x_m': 6 if i == 0 else 9,
                                         'y_m': 0 if i == 0 else 1}
                                        if i < 2 else {'exists': False})}
            missing_records.append({'route_rel': route, 'frame': frame,
                                    'lg_result': {'causal_analysis': causal}})
        write_jsonl(no_id_lg, missing_records)
        parsed_missing = load_lg(no_id_lg, parse_keyframes(keys))
        assert sum(v['has_causal_object'] for v in parsed_missing.values()) == 2
        one = parsed_missing[('route001', '0010')]
        assert one['actor_id_missing'] and one['selected_actor_id'] is None
        match, reason = recover_lg_id(one, actors.load_normalized_actors(None))
        assert (match, reason) == ('11', 'unique_class_pose_match'), (match, reason)
        no_pose = dict(one, actor_signature={'class': 'vehicle', 'x_m': 100, 'y_m': 100})
        assert recover_lg_id(no_pose, actors.load_normalized_actors(None))[0] is None
        # The user's mistaken cvaa argument is an LG full output; fail clearly.
        try:
            load_cvaa(full_lg, parse_keyframes(keys))
            raise AssertionError('LG full output mistaken for CVAA unexpectedly parsed')
        except ValueError as exc:
            assert 'Wrong --cvaa-rankings input' in str(exc), exc
        work = tmp / 'work'
        work.mkdir()
        fakecfg = SimpleNamespace(
            dataset=SimpleNamespace(dataset_root=root, work_dir=work,
                                    manifest_path=work / 'manifest.jsonl',
                                    annotation_dir=work / 'annotations'),
            sampling=SimpleNamespace(history_frames=40, future_frames=0,
                                     include_classes=['vehicle'], exclude_classes=['ego'],
                                     max_actor_distance_m=100, min_actor_forward_m=-50,
                                     max_candidates=80, random_seed=1,
                                     skip_if_candidates_truncated=True,
                                     include_empty_candidate_samples=True))
        # End-to-end missing-ID recovery must preserve LG true-positive status.
        alternate_work = tmp / 'missing_id_work'
        alternate_work.mkdir()
        alternate_cfg = SimpleNamespace(dataset=SimpleNamespace(
            dataset_root=root, work_dir=alternate_work,
            manifest_path=alternate_work / 'manifest.jsonl',
            annotation_dir=alternate_work / 'annotations'), sampling=fakecfg.sampling)
        recovered_summary = make_scheme_b(alternate_cfg, keys, no_id_lg, cfile, target=2, seed=1234)
        assert recovered_summary['lg_confirmed_missing_id_in_source'] == 1
        assert recovered_summary['lg_recovered_actor_id_in_samples'] == 1
        assert recovered_summary['lg_positive_unresolvable_in_samples'] == 0
        recovered_rows = list(jsonl(alternate_work / 'evaluation/scheme_b_predictions.jsonl'))
        row = next(r for r in recovered_rows if r['frame'] == '0010')
        assert row['lg_has_causal_object'] is True and row['lg_selected_actor_id'] == '11'
        assert row['lg_available'] is True and row['lg_actor_id_missing_in_source'] is True
        summary = make_scheme_b(fakecfg, keys, lgfile, cfile, target=2, seed=1234)
        assert summary['selected_lg_positive'] == 2 and summary['selected_lg_negative'] == 2
        assert summary['actor_mapping_mismatches'] == 0
        samples = list(jsonl(work / 'manifest.jsonl'))
        assert all(len(x['frames']) == 41 and x['center_frame_offset'] == 40 for x in samples)
        assert any(x['unavailable_history_frames'] > 0 for x in samples)
        assert all(x['frames'][-1]['stem'] == x['center_stem'] for x in samples)
        answers = {'0010': ['11'], '0012': ['22'], '0014': [], '0016': ['11']}
        metric._records = []
        for x in samples:
            for person in ('P01', 'P02', 'P03'):
                a = answers[x['center_stem']]
                metric._records.append({'sample_id': x['sample_id'],
                                        'annotator_id': person,
                                        'attention_status': 'actors' if a else 'none',
                                        'selected_actor_ids': a})
        summary = evaluate(fakecfg, work / 'evaluation' / 'scheme_b_predictions.jsonl')
        s = summary['overall_correct_including_human_none']
        assert s['n'] == 4, s
        assert s['lg_correct'] == 3 and s['cvaa_correct'] == 1, s
        assert s['paired_difference_lg_minus_cvaa'] == 0.5, s
        assert summary['human_none_specificity']['n'] == 1
        assert summary['human_positive_hit_at_1']['n'] == 3
        assert summary['method_prediction_coverage_on_human_evaluable']['cvaa']['available'] == 4
        # Negative testing: total positive count must not be silently changed.
        try:
            make_scheme_b(fakecfg, keys, lgfile, cfile, target=1)
            raise AssertionError('positive cohort truncation unexpectedly allowed')
        except ValueError as exc:
            assert 'expected exactly' in str(exc)
        # CVAA missing ranking is *not* a true prediction of no key actor.
        missing = tmp / 'cvaa_missing.jsonl'
        write_jsonl(missing, [{'route_id': 'route001', 'frame': '0010', 'top_actor_id': '22'}])
        try:
            make_scheme_b(fakecfg, keys, lgfile, missing, target=2)
            raise AssertionError('missing CVAA result unexpectedly admitted')
        except ValueError as exc:
            assert 'lack CVAA top actor' in str(exc)
        # Regression: matched negative sampling must follow the positive real-history
        # distribution, not simply the first N negative frames after shuffling.
        assert [history_stratum(x) for x in (40, 21, 1, 0)] == [
            'complete_40', 'partial_20_39', 'short_1_19', 'none_0']
        toy_keys = [('route', '%04d' % i) for i in range(10)]
        toy_history = [1, 1, 1, 40, 1, 25, 40, 40, 1, 40]
        def toy_preparer(k, group, root, cfg, cache, min_real_history):
            i = int(k[1]); return {'real_history_frames': toy_history[i], 'center_stem': k[1]}, None
        negs, have, rejected = sample_matched_negatives(
            toy_keys, None, None, {}, 0,
            {'complete_40': 3, 'partial_20_39': 1, 'short_1_19': 1},
            preparer=toy_preparer)
        assert len(negs) == 5 and not rejected
        from collections import Counter
        assert Counter(history_stratum(s['real_history_frames']) for s in negs) == {
            'complete_40': 3, 'partial_20_39': 1, 'short_1_19': 1}
        # Ensure a quota cannot be silently filled from the wrong history stratum.
        _negs, _have, _ = sample_matched_negatives(
            toy_keys, None, None, {}, 0,
            {'complete_40': 5}, preparer=toy_preparer)
        assert len(_negs) == 4 and _have['complete_40'] == 4
        print('PASS: 2+2; LG null-ID; class/pose recovery; paired evaluation; stratified negative sampling')

if __name__ == '__main__':
    test_scheme_b()
