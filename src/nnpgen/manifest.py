import re
from pathlib import Path
from typing import Dict, List

from .config import DEFAULT_POSCAR_POOL_ROOT
from .utils import read_json


_FRAME_RE = re.compile(r'(frame_\d{6})')


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _extract_frame_name(record_json_path: Path, info: Dict) -> str:
    paths = info.get('paths', {})
    frame_dir = paths.get('frame_dir', '')
    if frame_dir:
        return Path(frame_dir).name

    if record_json_path.name == 'frame_info.json':
        return record_json_path.parent.name

    m = _FRAME_RE.search(record_json_path.stem)
    if m:
        return m.group(1)

    return 'frame_unknown'


def _extract_job_name(info: Dict, record_json_path: Path) -> str:
    job_name = info.get('job_name')
    if job_name:
        return str(job_name)

    # run/<job_name>/system_xxx/frame_xxx/frame_info.json
    # run/<job_name>/system_xxx/frame_records/frame_000001.json
    parts = record_json_path.parts
    for i, part in enumerate(parts):
        if part == 'run' and i + 1 < len(parts):
            return parts[i + 1]
    return ''


def _pick_poscar_path(info: Dict) -> str:
    paths = info.get('paths', {})
    candidates = [
        paths.get('pooled_poscar', ''),
        paths.get('final_poscar', ''),
    ]

    for c in candidates:
        if c:
            p = Path(c)
            if p.exists():
                return str(p.resolve())

    for c in candidates:
        if c:
            return str(Path(c).resolve())

    return ''


def _normalize_record(record_json_path: Path, info: Dict) -> Dict:
    frame_name = _extract_frame_name(record_json_path, info)
    system_name = str(info.get('system_name', 'system_unknown'))

    return {
        'job_name': _extract_job_name(info, record_json_path),
        'system_name': system_name,
        'frame_name': frame_name,
        'selected_frame_index_1based': _safe_int(info.get('selected_frame_index_1based')),
        'source_xyz_path': str(info.get('source_input', '')),
        'source_frame_index_0based': _safe_int(info.get('source_frame_index_0based')),
        'source_frame_index_1based': _safe_int(info.get('source_frame_index_1based')),
        'poscar_path': _pick_poscar_path(info),
        'record_json_path': str(record_json_path.resolve()),
        'status': str(info.get('status', 'planned')),
        'paths': info.get('paths', {}),
    }


def frame_key(system_name: str, frame_name: str) -> str:
    return '{0}::{1}'.format(system_name, frame_name)


def scan_run_records(run_root: Path) -> List[Dict]:
    run_root = Path(run_root).resolve()
    if not run_root.exists():
        return []

    records = []
    json_paths = sorted(run_root.glob('system_*/frame_*/frame_info.json'))
    json_paths += sorted(run_root.glob('system_*/frame_records/frame_*.json'))

    for p in json_paths:
        try:
            info = read_json(p)
            records.append(_normalize_record(p, info))
        except Exception:
            continue

    records.sort(key=lambda r: (r.get('system_name', ''), r.get('frame_name', '')))
    return records


def scan_poscar_pool(job_name: str, poscar_pool_root: Path = DEFAULT_POSCAR_POOL_ROOT) -> List[Dict]:
    pool_job_dir = Path(poscar_pool_root).resolve() / job_name
    if not pool_job_dir.exists():
        return []

    records = []
    for poscar in sorted(pool_job_dir.glob('system_*/*.vasp')):
        system_name = poscar.parent.name
        m = _FRAME_RE.search(poscar.name)
        frame_name = m.group(1) if m else 'frame_unknown'
        records.append(
            {
                'job_name': job_name,
                'system_name': system_name,
                'frame_name': frame_name,
                'selected_frame_index_1based': None,
                'source_xyz_path': '',
                'source_frame_index_0based': None,
                'source_frame_index_1based': None,
                'poscar_path': str(poscar.resolve()),
                'record_json_path': '',
                'status': 'planned',
                'paths': {'pooled_poscar': str(poscar.resolve())},
            }
        )

    return records


def resolve_records(job_name: str, run_root: Path, poscar_pool_root: Path = DEFAULT_POSCAR_POOL_ROOT) -> List[Dict]:
    run_records = scan_run_records(run_root)
    pool_records = scan_poscar_pool(job_name=job_name, poscar_pool_root=poscar_pool_root)

    by_key = {}
    for rec in run_records:
        by_key[frame_key(rec['system_name'], rec['frame_name'])] = rec

    for pool_rec in pool_records:
        key = frame_key(pool_rec['system_name'], pool_rec['frame_name'])
        if key in by_key:
            by_key[key]['poscar_path'] = pool_rec['poscar_path']
            by_key[key].setdefault('paths', {})['pooled_poscar'] = pool_rec['poscar_path']
        else:
            by_key[key] = pool_rec

    records = list(by_key.values())
    records.sort(key=lambda r: (r.get('system_name', ''), r.get('frame_name', '')))
    return records
