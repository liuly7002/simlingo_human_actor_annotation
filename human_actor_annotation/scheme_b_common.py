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


def load_lg(path, keys):
    """LG all_frame_results.jsonl -> exact route relative + frame key."""
    allowed = set(keys)
    by_basename = {}
    for route, frame in keys:
        by_basename.setdefault(Path(route).name, set()).add(route)
    rows = {}
    for row in jsonl(path):
        frame = str(row.get('frame', ''))
        route = str(row.get('route_rel') or '').strip('/').replace('\\', '/')
        if not route:
            matches = by_basename.get(str(row.get('route_id', '')), set())
            if len(matches) != 1:
                raise ValueError('LG ambiguous or absent route_rel: %r frame=%s' % (row.get('route_id'), frame))
            route = next(iter(matches))
        key = (route, frame)
        if key not in allowed:
            continue
        if key in rows:
            raise ValueError('Duplicate LG record %r' % (key,))
        selected = row.get('causal_actor_id')
        flag = _flag(row.get('has_causal_object'))
        selected = str(selected) if selected is not None and str(selected).strip() else None
        if flag and selected is None:
            raise ValueError('LG has_causal_object=true, missing causal_actor_id: %r' % (key,))
        if not flag:
            selected = None
        rows[key] = {'available': True, 'selected_actor_id': selected,
                     'has_causal_object': flag}
    return rows


def load_cvaa(path, keys):
    """CVAA all_frame_rankings.jsonl -> key, no inference from missing output."""
    allowed = set(keys)
    by_basename = {}
    for route, frame in keys:
        by_basename.setdefault(Path(route).name, set()).add(route)
    rows = {}
    for row in jsonl(path):
        frame = str(row.get('frame', ''))
        route = row.get('route_rel')
        if route:
            route = str(route).replace('\\', '/').strip('/')
        else:
            basename = str(row.get('route_id', ''))
            matches = by_basename.get(basename, set())
            if not matches:
                continue
            if len(matches) > 1:
                raise ValueError('CVAA route_id collision; need route_rel: %s' % basename)
            route = next(iter(matches))
        key = (route, frame)
        if key not in allowed:
            continue
        if key in rows:
            raise ValueError('Duplicate CVAA record %r' % (key,))
        actor = row.get('top_actor_id')
        rank = row.get('ranking') or []
        if actor is None and isinstance(rank, list) and rank:
            actor = rank[0].get('actor_id')
        actor = str(actor) if actor is not None and str(actor).strip() else None
        rows[key] = {'available': bool(actor), 'selected_actor_id': actor,
                     'num_actors': row.get('num_actors', len(rank)),
                     'ranked_actor_ids': [str(x['actor_id']) for x in rank if isinstance(x, dict) and x.get('actor_id') is not None]}
    return rows


def key_to_sample_id(key):
    return key[0].replace('/', '__') + '__' + key[1]
