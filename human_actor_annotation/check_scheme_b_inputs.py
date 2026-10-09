#!/usr/bin/env python3
"""Quick input audit for Scheme B; reads no CARLA images. #修改20260720"""
import argparse
from pathlib import Path
from scheme_b_common import parse_keyframes, load_lg, load_cvaa


def main():
    p = argparse.ArgumentParser(description='Check LG causal counts and CVAA ranking coverage')
    p.add_argument('--keyframes', required=True)
    p.add_argument('--lg-frames', required=True)
    p.add_argument('--cvaa-rankings', default=None)
    p.add_argument('--target', type=int, default=147)
    a = p.parse_args()

    keys = parse_keyframes(a.keyframes)
    lg = load_lg(a.lg_frames, keys)
    pos = [k for k in keys if lg.get(k, {}).get('has_causal_object')]
    neg = [k for k in keys if k in lg and not lg[k]['has_causal_object']]
    print('Manifest keyframes:', len(keys))
    print('LG frame records matched:', len(lg))
    print('LG confirmed actor frames:', len(pos))
    print('LG non-confirmed actor frames:', len(neg))
    print('LG missing output frames:', len(keys) - len(lg))
    unresolved = [k for k in pos if lg[k].get('actor_id_missing')]
    print('LG confirmed frames missing actor ID:', len(unresolved))
    for k in unresolved[:10]:
        print('  missing-ID example:', k, 'actor:', lg[k].get('actor_signature'))
    if unresolved:
        print('NOTICE: Missing actor IDs require unique, class+geometry-matched '
              'annotation candidates, or they remain non-comparable. '
              'They MUST NOT be counted as no-actor predictions.')
    print('Positive target:', a.target)
    if len(pos) != a.target:
        print('NOTICE: confirmed count differs from target; verify tau/config, do not truncate silently.')
    if a.cvaa_rankings:
        cvaa = load_cvaa(a.cvaa_rankings, keys)
        available = sum(bool(v.get('available')) for v in cvaa.values())
        no_cvaa_pos = [k for k in pos if not cvaa.get(k, {}).get('available')]
        print('CVAA ranking records matched:', len(cvaa))
        print('CVAA valid top actor records:', available)
        print('LG-positive frames without valid CVAA top actor:', len(no_cvaa_pos))
        if no_cvaa_pos:
            print('Examples:', no_cvaa_pos[:5])


if __name__ == '__main__':
    main()
