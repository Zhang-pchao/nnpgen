#!/usr/bin/env python3
import argparse
import datetime
import json
from pathlib import Path


def _collect_group_dirs(root: Path, require_set: bool):
    groups = []
    for p in sorted(root.glob('**/group_*')):
        if not p.is_dir():
            continue
        if require_set and not any((p / x).is_dir() for x in ['set.000', 'set.001', 'set.002']):
            # generic fallback: any set.* dir
            has_set = any(sp.is_dir() and sp.name.startswith('set.') for sp in p.iterdir()) if p.exists() else False
            if not has_set:
                continue
        groups.append(str(p.resolve()))
    return groups


def main():
    ap = argparse.ArgumentParser(description='Fill finetune JSON systems arrays from split train/test roots')
    ap.add_argument('--input-json', required=True)
    ap.add_argument('--train-root', required=True)
    ap.add_argument('--test-root', required=True)
    ap.add_argument('--backup', action='store_true', default=True)
    ap.add_argument('--no-backup', dest='backup', action='store_false')
    ap.add_argument('--require-set', action='store_true', default=True)
    ap.add_argument('--no-require-set', dest='require_set', action='store_false')
    args = ap.parse_args()

    input_json = Path(args.input_json)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    if not input_json.is_file():
        raise FileNotFoundError('input-json not found: {0}'.format(input_json))
    if not train_root.is_dir():
        raise FileNotFoundError('train-root not found: {0}'.format(train_root))
    if not test_root.is_dir():
        raise FileNotFoundError('test-root not found: {0}'.format(test_root))

    train_systems = _collect_group_dirs(train_root, args.require_set)
    test_systems = _collect_group_dirs(test_root, args.require_set)

    if not train_systems:
        raise RuntimeError('No train systems found under: {0}'.format(train_root))
    if not test_systems:
        raise RuntimeError('No test systems found under: {0}'.format(test_root))

    obj = json.loads(input_json.read_text())

    training = obj.setdefault('training', {})
    tr_data = training.setdefault('training_data', {})
    va_data = training.setdefault('validation_data', {})

    tr_data['systems'] = train_systems
    va_data['systems'] = test_systems

    if args.backup:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = input_json.with_name(input_json.name + '.bak_' + ts)
        backup_path.write_text(input_json.read_text())

    input_json.write_text(json.dumps(obj, indent=2) + '\n')

    print('updated_json', str(input_json))
    print('train_systems', len(train_systems))
    print('test_systems', len(test_systems))
    print('train_first', train_systems[0])
    print('test_first', test_systems[0])


if __name__ == '__main__':
    main()
