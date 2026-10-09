"""Scheme B shared, method-neutral keyframes/prediction IO. #修改20260720"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open('r', encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                except ValueError as exc:
                    raise ValueError('%s:%s JSON error: %s' % (path, n, exc))
                if not isinstance(obj, dict):
                    raise ValueError('%s:%s must be JSON object' % (path, n))
                yield obj


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    names = sorted(set().union(*(row.keys() for row in rows)))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def parse_keyframes(path):
    result = []
    used = set()
    for n, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip().replace('\\', '/').rstrip('/')
        if not line or line.startswith('#'):
            continue
        parts = [x for x in line.split('/') if x]
        if len(parts) < 2 or '..' in parts or line.startswith('/'):
            raise ValueError('Invalid keyframes.txt entry line %d: %s' % (n, line))
        route_rel = '/'.join(parts[:-1])
        frame = parts[-1]
        key = (route_rel, frame)
        if key in used:
            continue
        used.add(key)
        result.append(key)
    if not result:
        raise ValueError('No keyframes found in %s' % path)
    return result


def _flag(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ('true', '1', 'yes')
    return bool(x)


def _resolve_key(row, by_basename, kind):
    """Resolve method output to (relative route path, original frame stem)."""
    frame = row.get('frame')
    if frame is None:
        raise ValueError('%s record has no frame field' % kind)
    frame = str(frame)
    route = row.get('route_rel')
    if route:
        route = str(route).strip('/').replace('\\', '/')
    else:
        basename = str(row.get('route_id') or '')
        matches = by_basename.get(basename, set())
        if len(matches) > 1:
            raise ValueError('%s route_id collision for %r; supply route_rel' % (kind, basename))
        if not matches:
            return None
        route = next(iter(matches))
    return (route, frame)


def _actor_id(value):
    if isinstance(value, dict):
        for field in ('id', 'actor_id', 'track_id', 'instance_id'):
            if value.get(field) is not None:
                return str(value[field])
        return None
    if value is None or isinstance(value, bool):
        return None
    return str(value) if str(value).strip() else None


def load_lg(path, keys):
    """Accept either LG all_frame_results.jsonl or all_frame_results_full.jsonl.

    The latter has its original per-frame LG JSON nested under lg_result,
    while the former exposes has_causal_object and causal_actor_id at top level.
    A schema mismatch raises an informative error rather than returning zero.
    """
    allowed = set(keys)
    by_basename = {}
    for route, _ in keys:
        by_basename.setdefault(Path(route).name, set()).add(route)
    rows = {}
    recognized = 0
    matched = 0
    for row in jsonl(path):
        key = _resolve_key(row, by_basename, 'LG')
        if key is None or key not in allowed:
            continue
        matched += 1
        if key in rows:
            raise ValueError('Duplicate LG record %r' % (key,))

        if 'has_causal_object' in row:
            # Flattened summary format: all_frame_results.jsonl.
            recognized += 1
            flag = _flag(row['has_causal_object'])
            selected = _actor_id(row.get('causal_actor_id'))
        else:
            # Full format: all_frame_results_full.jsonl -> lg_result.
            raw = row.get('lg_result')
            if not isinstance(raw, dict):
                raise ValueError('Unrecognized LG schema at %r: expected has_causal_object '
                                 'or lg_result.causal_analysis; file=%s' % (key, path))
            causal = raw.get('causal_analysis')
            if not isinstance(causal, dict) or 'has_causal_object' not in causal:
                raise ValueError('Missing lg_result.causal_analysis.has_causal_object at %r' % (key,))
            recognized += 1
            flag = _flag(causal['has_causal_object'])
            selected = _actor_id(causal.get('causal_object'))

        # A confirmed LG actor can legitimately have id=None: actor_loader.py
        # retains actors without CARLA IDs and causal_response.py tracks them
        # using (class, x_m, y_m). Never turn this into a no-actor prediction.
        signature = None
        if flag:
            actor = (row.get('lg_result', {}).get('causal_analysis', {}).get('causal_object')
                     if 'lg_result' in row else None)
            if isinstance(actor, dict):
                signature = {
                    'class': actor.get('class'),
                    'x_m': actor.get('x_m'),
                    'y_m': actor.get('y_m'),
                    'distance_m': actor.get('distance_m'),
                }
            # Preserve genuine source anomalies for auditing. With a flat
            # aggregate, the original causal actor position may be unavailable.
        else:
            selected = None
        rows[key] = {'available': True, 'selected_actor_id': selected,
                     'has_causal_object': flag,
                     'actor_id_missing': bool(flag and selected is None),
                     'actor_signature': signature}

    if not matched or not recognized:
        raise ValueError('No keyframes matched LG input %s. Check route_rel, frame, and input format.' % path)
    return rows


def load_cvaa(path, keys):
    """Read official CVAA all_frame_rankings.jsonl (not LG aggregation)."""
    allowed = set(keys)
    by_basename = {}
    for route, _ in keys:
        by_basename.setdefault(Path(route).name, set()).add(route)
    rows = {}
    matched = 0
    recognized = 0
    for row in jsonl(path):
        key = _resolve_key(row, by_basename, 'CVAA')
        if key is None or key not in allowed:
            continue
        matched += 1
        if key in rows:
            raise ValueError('Duplicate CVAA record %r' % (key,))
        if 'top_actor_id' not in row and 'ranking' not in row:
            raise ValueError('Wrong --cvaa-rankings input: file %s contains no top_actor_id '
                             'or ranking at %r. This looks like an LG aggregation, '
                             'not CVAA all_frame_rankings.jsonl.' % (path, key))
        recognized += 1
        rank = row.get('ranking') or []
        actor = _actor_id(row.get('top_actor_id'))
        if actor is None and isinstance(rank, list) and rank:
            actor = _actor_id(rank[0].get('actor_id')) if isinstance(rank[0], dict) else None
        rows[key] = {'available': bool(actor), 'selected_actor_id': actor,
                     'num_actors': row.get('num_actors', len(rank)),
                     'ranked_actor_ids': [_actor_id(x.get('actor_id')) for x in rank
                                          if isinstance(x, dict) and _actor_id(x.get('actor_id')) is not None]}
    if matched and not recognized:
        raise ValueError('No CVAA ranking fields found in %s' % path)
    return rows


def key_to_sample_id(key):
    return key[0].replace('/', '__') + '__' + key[1]
