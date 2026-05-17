#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..config import DEFAULT_REMOTE_HOST, RUN_ROOT


def _now():
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def _run(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def _stable_int(s):
    h = hashlib.md5(str(s).encode('utf-8')).hexdigest()[:8]
    return int(h, 16)


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def _write_raw(path, arr):
    _ensure_dir(Path(path).parent)
    np.savetxt(str(path), arr)


def _copy_if_exists(src, dst):
    sp = Path(src)
    if sp.exists():
        _ensure_dir(Path(dst).parent)
        shutil.copy2(str(sp), str(dst))


def _split_indices(n, ratio, seed):
    if n <= 1:
        return np.arange(n, dtype=int), np.array([], dtype=int)

    n_test = int(round(n * ratio))
    if n_test < 1:
        n_test = 1
    if n_test >= n:
        n_test = n - 1

    idx = np.arange(n, dtype=int)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return train_idx, test_idx


def _save_group_split(src_group_root, src_set_dir, train_group_root, test_group_root, test_ratio, base_seed):
    energy = np.load(str(src_set_dir / 'energy.npy'))
    box = np.load(str(src_set_dir / 'box.npy'))
    coord = np.load(str(src_set_dir / 'coord.npy'))
    force = np.load(str(src_set_dir / 'force.npy'))
    virial_path = src_set_dir / 'virial.npy'
    virial = np.load(str(virial_path)) if virial_path.exists() else None

    n = int(energy.shape[0])
    group_key = str(src_group_root)
    seed = int(base_seed) + _stable_int(group_key)
    train_idx, test_idx = _split_indices(n, float(test_ratio), seed)

    train_set = Path(train_group_root) / src_set_dir.name
    test_set = Path(test_group_root) / src_set_dir.name
    _ensure_dir(train_set)
    _ensure_dir(test_set)

    # Write npy subsets
    np.save(str(train_set / 'energy.npy'), energy[train_idx])
    np.save(str(train_set / 'box.npy'), box[train_idx])
    np.save(str(train_set / 'coord.npy'), coord[train_idx])
    np.save(str(train_set / 'force.npy'), force[train_idx])
    if virial is not None:
        np.save(str(train_set / 'virial.npy'), virial[train_idx])

    np.save(str(test_set / 'energy.npy'), energy[test_idx])
    np.save(str(test_set / 'box.npy'), box[test_idx])
    np.save(str(test_set / 'coord.npy'), coord[test_idx])
    np.save(str(test_set / 'force.npy'), force[test_idx])
    if virial is not None:
        np.save(str(test_set / 'virial.npy'), virial[test_idx])

    # Write raw subsets for compatibility
    _write_raw(Path(train_group_root) / 'energy.raw', np.reshape(energy[train_idx], (-1, 1)))
    _write_raw(Path(train_group_root) / 'box.raw', np.reshape(box[train_idx], (len(train_idx), -1)))
    _write_raw(Path(train_group_root) / 'coord.raw', np.reshape(coord[train_idx], (len(train_idx), -1)))
    _write_raw(Path(train_group_root) / 'force.raw', np.reshape(force[train_idx], (len(train_idx), -1)))
    if virial is not None:
        _write_raw(Path(train_group_root) / 'virial.raw', np.reshape(virial[train_idx], (len(train_idx), -1)))

    _write_raw(Path(test_group_root) / 'energy.raw', np.reshape(energy[test_idx], (-1, 1)))
    _write_raw(Path(test_group_root) / 'box.raw', np.reshape(box[test_idx], (len(test_idx), -1)))
    _write_raw(Path(test_group_root) / 'coord.raw', np.reshape(coord[test_idx], (len(test_idx), -1)))
    _write_raw(Path(test_group_root) / 'force.raw', np.reshape(force[test_idx], (len(test_idx), -1)))
    if virial is not None:
        _write_raw(Path(test_group_root) / 'virial.raw', np.reshape(virial[test_idx], (len(test_idx), -1)))

    # Copy atom type metadata
    _copy_if_exists(Path(src_group_root) / 'type.raw', Path(train_group_root) / 'type.raw')
    _copy_if_exists(Path(src_group_root) / 'type_map.raw', Path(train_group_root) / 'type_map.raw')
    _copy_if_exists(Path(src_group_root) / 'type.raw', Path(test_group_root) / 'type.raw')
    _copy_if_exists(Path(src_group_root) / 'type_map.raw', Path(test_group_root) / 'type_map.raw')

    return {
        'group': str(src_group_root),
        'set_dir': str(src_set_dir),
        'frames_total': n,
        'frames_train': int(len(train_idx)),
        'frames_test': int(len(test_idx)),
        'seed': int(seed),
    }


def _list_dataset_dirs(local_source_root):
    root = Path(local_source_root)
    ds = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith('_'):
            continue
        ds.append(p)
    return ds


def _list_set_dirs(dataset_dir):
    return sorted(dataset_dir.glob('**/set.*'))


def main():
    ap = argparse.ArgumentParser(description='Split step3_final_seq_dp datasets into train/test on server 11')
    ap.add_argument('--remote-host', default=DEFAULT_REMOTE_HOST)
    ap.add_argument('--remote-root', required=True)
    ap.add_argument('--local-source-root', default=str(RUN_ROOT / 'dp_source_cache'))
    ap.add_argument('--train-root', default=str(RUN_ROOT / 'dp_train_data'))
    ap.add_argument('--test-root', default=str(RUN_ROOT / 'dp_test_data'))
    ap.add_argument('--test-ratio', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=20260513)
    ap.add_argument('--sync-source', action='store_true', default=True)
    ap.add_argument('--no-sync-source', dest='sync_source', action='store_false')
    ap.add_argument('--clean-target', action='store_true', default=True)
    ap.add_argument('--no-clean-target', dest='clean_target', action='store_false')
    ap.add_argument('--summary-path', default=str(RUN_ROOT / 'dp_train_test_split_summary.json'))
    args = ap.parse_args()

    if args.test_ratio <= 0 or args.test_ratio >= 1:
        raise ValueError('test-ratio must be between 0 and 1')

    local_source_root = Path(args.local_source_root)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    if args.sync_source:
        _ensure_dir(local_source_root)
        rsync_cmd = [
            'rsync', '-az', '--delete', '--exclude', '_nnpgen_tools',
            '{0}:{1}/'.format(args.remote_host, args.remote_root),
            str(local_source_root) + '/'
        ]
        _run(rsync_cmd)

    datasets = _list_dataset_dirs(local_source_root)
    if not datasets:
        raise RuntimeError('No dataset directories found in local source root: {0}'.format(local_source_root))

    if args.clean_target:
        if train_root.exists():
            shutil.rmtree(str(train_root))
        if test_root.exists():
            shutil.rmtree(str(test_root))
    _ensure_dir(train_root)
    _ensure_dir(test_root)

    split_rows = []
    dataset_rows = []

    for ds in datasets:
        set_dirs = _list_set_dirs(ds)
        ds_train = train_root / ds.name
        ds_test = test_root / ds.name

        for src_set in set_dirs:
            src_group_root = src_set.parent
            rel_group = src_group_root.relative_to(ds)
            train_group_root = ds_train / rel_group
            test_group_root = ds_test / rel_group
            row = _save_group_split(
                src_group_root=src_group_root,
                src_set_dir=src_set,
                train_group_root=train_group_root,
                test_group_root=test_group_root,
                test_ratio=args.test_ratio,
                base_seed=args.seed,
            )
            row['dataset'] = ds.name
            row['relative_group'] = str(rel_group)
            split_rows.append(row)

        dataset_rows.append({
            'dataset': ds.name,
            'groups': len(set_dirs),
            'frames_total': int(sum(r['frames_total'] for r in split_rows if r['dataset'] == ds.name)),
            'frames_train': int(sum(r['frames_train'] for r in split_rows if r['dataset'] == ds.name)),
            'frames_test': int(sum(r['frames_test'] for r in split_rows if r['dataset'] == ds.name)),
        })

    summary = {
        'created_at': _now(),
        'remote_host': args.remote_host,
        'remote_root': args.remote_root,
        'local_source_root': str(local_source_root),
        'train_root': str(train_root),
        'test_root': str(test_root),
        'test_ratio': float(args.test_ratio),
        'seed': int(args.seed),
        'datasets': dataset_rows,
        'totals': {
            'datasets': len(dataset_rows),
            'groups': int(sum(d['groups'] for d in dataset_rows)),
            'frames_total': int(sum(d['frames_total'] for d in dataset_rows)),
            'frames_train': int(sum(d['frames_train'] for d in dataset_rows)),
            'frames_test': int(sum(d['frames_test'] for d in dataset_rows)),
        },
        'groups': split_rows,
    }

    sp = Path(args.summary_path)
    _ensure_dir(sp.parent)
    sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')

    print(json.dumps(summary['totals'], indent=2, sort_keys=True))
    for d in dataset_rows:
        print('{dataset}: groups={groups} total={frames_total} train={frames_train} test={frames_test}'.format(**d))
    print('[Saved] {0}'.format(sp))


if __name__ == '__main__':
    main()
