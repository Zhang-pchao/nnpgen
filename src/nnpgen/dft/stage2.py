import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import (
    DEFAULT_PLAN_ROOT,
    DEFAULT_REMOTE_HOST,
    ELEMENT_ORDER,
    PROJECT_ROOT,
    RUN_ROOT,
    VASP15_INCAR_TEMPLATE,
    VASP15_POTCAR_ORG_DIR,
    VASP15_ROOT,
    VASP15_RUN_ROOT,
)
from ..utils import abs_path, ensure_dir, read_json, write_json


ALLOWED_11_ROOT = PROJECT_ROOT
ALLOWED_15_ROOT = VASP15_ROOT
TEMPLATE_POTCAR_ORG_15 = VASP15_POTCAR_ORG_DIR
TEMPLATE_INCAR_15 = VASP15_INCAR_TEMPLATE

STAGE1_DEFAULT_MANIFEST = DEFAULT_PLAN_ROOT / 'stage1_manifest.json'
STAGE2_DEFAULT_MANIFEST = DEFAULT_PLAN_ROOT / 'stage2_dft_manifest.json'
STAGE2_DEFAULT_RUN_ROOT_15 = VASP15_RUN_ROOT / 'stage2_dft'

SUPPORTED_ELEMENTS = set(ELEMENT_ORDER)


def _now() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def _run(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def _ssh(host: str, command: str) -> str:
    return _run(['ssh', host, command]).stdout


def _scp_to_remote(local_path: Path, host: str, remote_path: str) -> None:
    _run(['scp', str(local_path), '{0}:{1}'.format(host, remote_path)])


def _scp_from_remote(host: str, remote_path: str, local_path: Path) -> None:
    _run(['scp', '{0}:{1}'.format(host, remote_path), str(local_path)])


def _assert_under_11(path: Path) -> None:
    p = path.resolve()
    root = ALLOWED_11_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError('Path outside allowed 11 root: {0}'.format(p))


def _assert_under_15(path_15: str) -> None:
    p = Path(path_15).resolve()
    root = ALLOWED_15_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError('Path outside allowed 15 root: {0}'.format(p))


def _tmp_text_file(text: str) -> Path:
    fd, path = tempfile.mkstemp(prefix='nnpgen_stage2_', suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    return Path(path)


def _remote_exists(host: str, path_remote: str) -> bool:
    out = _ssh(host, '[ -e {0} ] && echo 1 || echo 0'.format(shlex.quote(path_remote))).strip()
    return out == '1'


def _remote_read_json(host: str, path_remote: str) -> Dict:
    fd, path = tempfile.mkstemp(prefix='nnpgen_stage2_json_', suffix='.json')
    os.close(fd)
    p = Path(path)
    try:
        _scp_from_remote(host, path_remote, p)
        with p.open('r') as f:
            return json.load(f)
    finally:
        if p.exists():
            p.unlink()


def _remote_write_json(host: str, path_remote: str, data: Dict) -> None:
    tmp = _tmp_text_file(json.dumps(data, indent=2, sort_keys=True) + '\n')
    try:
        _scp_to_remote(tmp, host, path_remote)
    finally:
        if tmp.exists():
            tmp.unlink()


def _remote_write_text(host: str, path_remote: str, text: str) -> None:
    tmp = _tmp_text_file(text)
    try:
        _scp_to_remote(tmp, host, path_remote)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_incar_template(host_15: str, path_15: str) -> str:
    fd, path = tempfile.mkstemp(prefix='nnpgen_incar_', suffix='.txt')
    os.close(fd)
    p = Path(path)
    try:
        _scp_from_remote(host_15, path_15, p)
        return p.read_text()
    finally:
        if p.exists():
            p.unlink()


def _parse_poscar_local(poscar_path: Path) -> Dict:
    lines = poscar_path.read_text().splitlines()
    if len(lines) < 7:
        raise ValueError('Invalid POSCAR: {0}'.format(poscar_path))
    elems = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(elems) != len(counts):
        raise ValueError('POSCAR element/count mismatch: {0}'.format(poscar_path))
    for e in elems:
        if e not in SUPPORTED_ELEMENTS:
            raise ValueError('Unsupported element {0} in {1}'.format(e, poscar_path))
    comp = {e: c for e, c in zip(elems, counts)}
    return {'elements': elems, 'counts': counts, 'atom_count': sum(counts), 'composition': comp}


def _patch_incar_ncore(incar_text: str, ncore: int) -> str:
    lines = incar_text.splitlines()
    out = []
    found = False
    for line in lines:
        if re.match(r'^\s*NCORE\s*=', line):
            out.append('NCORE    = {0}'.format(ncore))
            found = True
        else:
            out.append(line)
    if not found:
        out.append('NCORE    = {0}'.format(ncore))
    return '\n'.join(out) + '\n'


def _job_state_from_qstat(host: str, job_id: str) -> str:
    if not job_id:
        return ''
    cmd = 'qstat -f {0} 2>/dev/null | awk -F= \'/job_state/ {{gsub(/ /,\"\",$2); print $2; exit}}\''.format(shlex.quote(job_id))
    return _ssh(host, cmd).strip()


def _map_qstat_to_status(job_state: str) -> str:
    js = job_state.strip().upper()
    if js in {'Q', 'H', 'W', 'T'}:
        return 'submitted'
    if js in {'R', 'E'}:
        return 'running'
    if js in {'C'}:
        return 'finished'
    return ''


def _find_jobs_by_name(host: str, task_name: str) -> List[Dict[str, str]]:
    if not task_name:
        return []

    user = _ssh(host, 'whoami').strip()
    if user:
        out = _ssh(host, 'qstat -u {0} 2>/dev/null || true'.format(shlex.quote(user)))
    else:
        out = _ssh(host, 'qstat 2>/dev/null || true')

    jobs = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('master:') or s.startswith('Job ID') or s.startswith('Req') or s.startswith('---'):
            continue

        parts = s.split()
        job_id = ''
        job_name = ''
        state = ''

        if len(parts) >= 10:
            job_id = parts[0]
            job_name = parts[3]
            state = parts[-2].upper()
        elif len(parts) >= 6:
            job_id = parts[0]
            job_name = parts[1]
            state = parts[4].upper()
        else:
            continue

        if job_name == task_name:
            jobs.append({'job_id': job_id, 'state': state})

    return jobs


def _pick_preferred_job(jobs: List[Dict[str, str]]) -> Dict[str, str]:
    if not jobs:
        return {}
    order = ['R', 'E', 'Q', 'H', 'W', 'T', 'C']
    for st in order:
        for j in jobs:
            if str(j.get('state', '')).upper() == st:
                return j
    return jobs[0]


def _refresh_task_status_remote(host: str, task_dir_15: str, info: Dict) -> Dict:
    status = str(info.get('status', 'planned'))
    job_id = str(info.get('job_id') or '')
    task_name = str(info.get('task_name') or '')

    qstate = _job_state_from_qstat(host, job_id) if job_id else ''
    mapped = _map_qstat_to_status(qstate)

    if not mapped and task_name:
        jobs = _find_jobs_by_name(host, task_name)
        preferred = _pick_preferred_job(jobs)
        if preferred:
            mapped = _map_qstat_to_status(preferred.get('state', ''))
            if preferred.get('job_id'):
                info['job_id'] = preferred['job_id']

    tag_finished = _remote_exists(host, task_dir_15 + '/tag_finished')
    tag_failed = _remote_exists(host, task_dir_15 + '/tag_failed')
    outcar_exists = _remote_exists(host, task_dir_15 + '/OUTCAR')
    oszicar_exists = _remote_exists(host, task_dir_15 + '/OSZICAR')

    if mapped:
        status = mapped
    else:
        if tag_finished:
            status = 'finished'
        elif tag_failed:
            status = 'failed'
        elif outcar_exists or oszicar_exists:
            status = 'running'

    info['status'] = status
    info['updated_at'] = _now()
    return info


def _load_stage2_manifest(manifest_path: Path) -> Dict:
    data = read_json(manifest_path)
    if 'entries' not in data or not isinstance(data['entries'], list):
        raise ValueError('Invalid stage2 manifest, entries list missing: {0}'.format(manifest_path))
    return data


def add_stage2_select_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--stage1-manifest', default=str(STAGE1_DEFAULT_MANIFEST))
    parser.add_argument('--stage2-manifest', default=str(STAGE2_DEFAULT_MANIFEST))
    parser.add_argument('--target-root-15', default=str(STAGE2_DEFAULT_RUN_ROOT_15))
    parser.add_argument('--n-select', type=int, default=4)


def run_stage2_select(args: argparse.Namespace) -> Path:
    stage1_manifest = abs_path(args.stage1_manifest)
    stage2_manifest = abs_path(args.stage2_manifest)
    _assert_under_11(stage1_manifest)
    _assert_under_11(stage2_manifest)

    target_root_15 = str(args.target_root_15).rstrip('/')
    _assert_under_15(target_root_15)

    data = read_json(stage1_manifest)
    entries = data.get('entries', [])

    finished = [e for e in entries if e.get('status') == 'finished' and e.get('pooled_poscar_path')]
    selected = finished[: max(1, int(args.n_select))]

    out_entries = []
    for i, e in enumerate(selected, start=1):
        poscar_11 = str(e.get('pooled_poscar_path'))
        pinfo = _parse_poscar_local(Path(poscar_11))
        task_name = 'task_{0:03d}'.format(i)
        out_entries.append(
            {
                'task_name': task_name,
                'source_poscar_path_11': poscar_11,
                'source_xyz_path': e.get('source_xyz_path', ''),
                'frame_index': e.get('source_frame_index_0based', None),
                'system_name': e.get('system_name', ''),
                'composition': pinfo['composition'],
                'atom_count': pinfo['atom_count'],
                'elements_order': pinfo['elements'],
                'target_task_dir_15': target_root_15 + '/' + task_name,
                'status': 'planned',
                'job_id': None,
            }
        )

    out = {
        'schema_version': 'nnpgen.stage2.firsttype.v1',
        'created_at': _now(),
        'updated_at': _now(),
        'source_stage1_manifest': str(stage1_manifest),
        'selection_rule': 'status=finished,first_n',
        'requested_n': int(args.n_select),
        'selected_n': len(out_entries),
        'target_root_15': target_root_15,
        'entries': out_entries,
    }

    ensure_dir(stage2_manifest.parent)
    write_json(stage2_manifest, out)
    print('Stage2 manifest written: {0} selected={1}'.format(stage2_manifest, len(out_entries)))
    return stage2_manifest


def add_stage2_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--stage2-manifest', default=str(STAGE2_DEFAULT_MANIFEST))
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)
    parser.add_argument('--queue', default='normal4')
    parser.add_argument('--node-list', default='node41,node42,node43')
    parser.add_argument('--ppn', type=int, default=28)
    parser.add_argument('--ncore', type=int, default=28)
    parser.add_argument('--walltime', default='1000:00:00')


def _build_run_pbs(task_name: str, task_dir: str, queue: str, node: str, ppn: int, walltime: str) -> str:
    return (
        '#!/bin/bash -l\n'
        '#PBS -q {queue}\n'
        '#PBS -N {job}\n'
        '#PBS -l nodes={node}:ppn={ppn}\n'
        '#PBS -l walltime={walltime}\n\n'
        'module load vasp/vasp.5.4.4.pl2\n\n'
        'cd {task_dir}\n'
        'ulimit -s unlimited\n'
        'export OMP_NUM_THREADS=1\n\n'
        'if [ -f tag_finished ]; then\n'
        '  exit 0\n'
        'fi\n'
        'rm -f tag_failed\n'
        'mpirun -np {ppn} vasp_std\n'
        'rc=$?\n'
        'if [ "$rc" -eq 0 ]; then\n'
        '  touch tag_finished\n'
        'else\n'
        '  touch tag_failed\n'
        'fi\n'
        'exit "$rc"\n'
    ).format(queue=queue, job=task_name, node=node, ppn=ppn, walltime=walltime, task_dir=task_dir)


def run_stage2_prepare(args: argparse.Namespace) -> Path:
    manifest_path = abs_path(args.stage2_manifest)
    _assert_under_11(manifest_path)
    data = _load_stage2_manifest(manifest_path)

    node_list = [x.strip() for x in str(args.node_list).split(',') if x.strip()]
    if not node_list:
        node_list = ['node41', 'node42', 'node43']

    incar_template_text = _load_incar_template(args.host_15, str(TEMPLATE_INCAR_15))

    processed = 0

    for idx, e in enumerate(data['entries']):
        task_name = str(e.get('task_name'))
        task_dir = str(e.get('target_task_dir_15'))
        poscar_11 = Path(str(e.get('source_poscar_path_11'))).resolve()
        _assert_under_15(task_dir)

        _ssh(args.host_15, 'mkdir -p {0}'.format(shlex.quote(task_dir)))

        poscar_remote = task_dir + '/POSCAR'
        _scp_to_remote(poscar_11, args.host_15, poscar_remote)

        elems = e.get('elements_order') or _parse_poscar_local(poscar_11)['elements']
        pot_sources = [str(TEMPLATE_POTCAR_ORG_15 / ('POTCAR_' + el)) for el in elems]
        pot_src_expr = ' '.join(shlex.quote(x) for x in pot_sources)
        potcar_remote = task_dir + '/POTCAR'
        pot_cmd = (
            'set -e; '
            'for f in {src}; do [ -f "$f" ] || {{ echo "Missing POTCAR source: $f" >&2; exit 1; }}; done; '
            'cat {src} > {out}'
        ).format(src=pot_src_expr, out=shlex.quote(potcar_remote))
        _ssh(args.host_15, pot_cmd)

        incar_text = _patch_incar_ncore(incar_template_text, int(args.ncore))
        incar_remote = task_dir + '/INCAR'
        _remote_write_text(args.host_15, incar_remote, incar_text)

        node = node_list[idx % len(node_list)]
        run_pbs_text = _build_run_pbs(task_name, task_dir, args.queue, node, int(args.ppn), args.walltime)
        run_pbs_remote = task_dir + '/run.pbs'
        _remote_write_text(args.host_15, run_pbs_remote, run_pbs_text)
        _ssh(args.host_15, 'chmod 755 {0}'.format(shlex.quote(run_pbs_remote)))

        ti = {
            'task_name': task_name,
            'job_id': None,
            'status': 'prepared',
            'source_poscar_path_11': str(poscar_11),
            'source_xyz_path': e.get('source_xyz_path', ''),
            'frame_index': e.get('frame_index', None),
            'system_name': e.get('system_name', ''),
            'composition': e.get('composition', {}),
            'elements_order': elems,
            'paths': {
                'task_dir_15': task_dir,
                'poscar': poscar_remote,
                'potcar': potcar_remote,
                'incar': incar_remote,
                'run_pbs': run_pbs_remote,
            },
            'resources': {
                'queue': args.queue,
                'node': node,
                'ppn': int(args.ppn),
                'ncore': int(args.ncore),
                'walltime': args.walltime,
            },
            'updated_at': _now(),
        }
        ti_remote = task_dir + '/task_info.json'
        _remote_write_json(args.host_15, ti_remote, ti)

        e['status'] = 'prepared'
        e['job_id'] = None
        e['prepared'] = {
            'task_info_path_15': ti_remote,
            'poscar_path_15': poscar_remote,
            'potcar_path_15': potcar_remote,
            'incar_path_15': incar_remote,
            'run_pbs_path_15': run_pbs_remote,
            'updated_at': _now(),
        }
        processed += 1

    data['updated_at'] = _now()
    data['prepared_n'] = processed
    write_json(manifest_path, data)
    print('Stage2 prepared tasks: {0}'.format(processed))
    return manifest_path


def add_stage2_submit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--stage2-manifest', default=str(STAGE2_DEFAULT_MANIFEST))
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)
    parser.add_argument('--retry-failed', action='store_true')


def run_stage2_submit(args: argparse.Namespace) -> List[str]:
    manifest_path = abs_path(args.stage2_manifest)
    _assert_under_11(manifest_path)
    data = _load_stage2_manifest(manifest_path)

    submitted = []

    for e in data['entries']:
        task_dir = str(e.get('target_task_dir_15'))
        task_name = str(e.get('task_name'))
        info_remote = task_dir + '/task_info.json'
        run_pbs_remote = task_dir + '/run.pbs'

        if not _remote_exists(args.host_15, info_remote):
            continue

        info = _remote_read_json(args.host_15, info_remote)
        info = _refresh_task_status_remote(args.host_15, task_dir, info)
        cur = str(info.get('status', 'planned'))

        if cur == 'finished':
            _remote_write_json(args.host_15, info_remote, info)
            e['status'] = 'finished'
            e['job_id'] = info.get('job_id')
            continue

        if cur == 'failed' and not args.retry_failed:
            _remote_write_json(args.host_15, info_remote, info)
            e['status'] = 'failed'
            e['job_id'] = info.get('job_id')
            continue

        if cur in {'submitted', 'running'}:
            _remote_write_json(args.host_15, info_remote, info)
            e['status'] = cur
            e['job_id'] = info.get('job_id')
            continue

        existing_jobs = _find_jobs_by_name(args.host_15, task_name)
        preferred = _pick_preferred_job(existing_jobs)
        if preferred:
            st = _map_qstat_to_status(preferred.get('state', '')) or 'submitted'
            info['job_id'] = preferred.get('job_id')
            info['status'] = st
            info['updated_at'] = _now()
            _remote_write_json(args.host_15, info_remote, info)
            e['status'] = st
            e['job_id'] = info.get('job_id')
            continue

        if not _remote_exists(args.host_15, run_pbs_remote):
            continue

        out = _ssh(args.host_15, 'cd {0} && qsub run.pbs'.format(shlex.quote(task_dir))).strip()
        job_id = out.split()[0]

        info['job_id'] = job_id
        info['status'] = 'submitted'
        info['updated_at'] = _now()
        _remote_write_json(args.host_15, info_remote, info)

        e['status'] = 'submitted'
        e['job_id'] = job_id
        e['submitted_at'] = _now()
        submitted.append(job_id)

        print('Submitted {0}: job_id={1}'.format(task_name, job_id))

    data['updated_at'] = _now()
    data['submitted_n'] = len(submitted)
    write_json(manifest_path, data)
    print('Stage2 submitted jobs: {0}'.format(len(submitted)))
    return submitted


def add_stage2_status_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--run-root-15', required=True)
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)


def run_stage2_status(args: argparse.Namespace) -> Dict[str, int]:
    run_root = str(args.run_root_15).rstrip('/')
    _assert_under_15(run_root)

    out = _ssh(args.host_15, 'find {0} -type f -name task_info.json | sort'.format(shlex.quote(run_root)))
    paths = [x.strip() for x in out.splitlines() if x.strip()]

    counts = {
        'total': 0,
        'planned': 0,
        'prepared': 0,
        'submitted': 0,
        'running': 0,
        'finished': 0,
        'failed': 0,
    }

    for p in paths:
        task_dir = str(Path(p).parent)
        info = _remote_read_json(args.host_15, p)
        info = _refresh_task_status_remote(args.host_15, task_dir, info)
        _remote_write_json(args.host_15, p, info)

        s = str(info.get('status', 'planned'))
        if s not in counts:
            s = 'planned'
        counts['total'] += 1
        counts[s] += 1

    print(
        'run_root_15={0} total={1} planned={2} prepared={3} submitted={4} running={5} finished={6} failed={7}'.format(
            run_root,
            counts['total'],
            counts['planned'],
            counts['prepared'],
            counts['submitted'],
            counts['running'],
            counts['finished'],
            counts['failed'],
        )
    )
    return counts


def is_15_root(path_str: str) -> bool:
    p = Path(path_str).resolve()
    root = ALLOWED_15_ROOT.resolve()
    return p == root or root in p.parents
