import argparse
import json
import os
import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..config import (
    DEFAULT_REMOTE_HOST,
    ELEMENT_ORDER,
    PROJECT_ROOT,
    VASP15_INCAR_TEMPLATE,
    VASP15_POTCAR_ORG_DIR,
    VASP15_ROOT,
    VASP15_RUN_ROOT,
    VASP15_SMOKE_ROOT,
)
from ..utils import abs_path, read_json, write_json


ALLOWED_15_ROOT = VASP15_ROOT
ALLOWED_15_SMOKE_ROOT = VASP15_SMOKE_ROOT
ALLOWED_15_RUN_ROOT = VASP15_RUN_ROOT
SUPPORTED_ELEMENTS = set(ELEMENT_ORDER)


def _now() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def _assert_under_project(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    p = path.resolve()
    if p != root and root not in p.parents:
        raise ValueError('Path is outside project root: {0}'.format(p))


def _assert_target_root_15(target_root_15: Path) -> None:
    target = Path(str(target_root_15)).resolve()
    smoke = ALLOWED_15_SMOKE_ROOT.resolve()
    run_root = ALLOWED_15_RUN_ROOT.resolve()

    in_smoke = target == smoke or smoke in target.parents
    in_run = target == run_root or run_root in target.parents
    if not in_smoke and not in_run:
        raise ValueError('target-root-15 must be under {0} or {1}'.format(smoke, run_root))


def _run(cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def _ssh(host: str, command: str) -> str:
    result = _run(['ssh', host, command])
    return result.stdout


def _scp_to_remote(src: Path, host: str, dst: str) -> None:
    _run(['scp', str(src), '{0}:{1}'.format(host, dst)])


def _sha256_local(path: Path) -> str:
    out = _run(['sha256sum', str(path)]).stdout.strip()
    return out.split()[0]


def _sha256_remote(host: str, path_remote: str) -> str:
    out = _ssh(host, 'sha256sum {0}'.format(shlex.quote(path_remote))).strip()
    return out.split()[0]


def _parse_poscar_elements_remote(host: str, poscar_path_remote: str) -> List[str]:
    cmd = "awk 'NR==6{{print;exit}}' {0}".format(shlex.quote(poscar_path_remote))
    line = _ssh(host, cmd).strip()
    elems = [x for x in line.split() if x]
    if not elems:
        raise ValueError('Failed to parse POSCAR elements on server 15: {0}'.format(poscar_path_remote))

    for elem in elems:
        if elem not in SUPPORTED_ELEMENTS:
            raise ValueError('Unsupported element in POSCAR: {0}'.format(elem))
    return elems


def _build_potcar_remote(host: str, task_dir_15: str, elements_order: List[str], potcar_org_dir_15: str) -> str:
    source_files = []
    for elem in elements_order:
        source_files.append('{0}/POTCAR_{1}'.format(potcar_org_dir_15.rstrip('/'), elem))

    source_expr = ' '.join(shlex.quote(p) for p in source_files)
    potcar_out = '{0}/POTCAR'.format(task_dir_15.rstrip('/'))

    cmd = (
        'set -euo pipefail; '
        'for f in {src}; do [ -f "$f" ] || {{ echo "Missing POTCAR source: $f" >&2; exit 1; }}; done; '
        'cat {src} > {out}'
    ).format(src=source_expr, out=shlex.quote(potcar_out))

    _ssh(host, cmd)
    return potcar_out


def _patch_incar_remote(host: str, task_dir_15: str, incar_template_15: str, ncore: int) -> str:
    incar_out = '{0}/INCAR'.format(task_dir_15.rstrip('/'))
    cmd = (
        'set -euo pipefail; '
        'cp {template} {out}; '
        'if grep -Eq "^[[:space:]]*NCORE[[:space:]]*=" {out}; then '
        'sed -i -E "s/^[[:space:]]*NCORE[[:space:]]*=.*/NCORE    = {ncore}/" {out}; '
        'else '
        'echo "NCORE    = {ncore}" >> {out}; '
        'fi'
    ).format(template=shlex.quote(incar_template_15), out=shlex.quote(incar_out), ncore=ncore)
    _ssh(host, cmd)
    return incar_out


def _write_remote_text_file(host: str, text: str, path_remote: str) -> None:
    cmd = "cat > {path} <<'EOF'\n{text}\nEOF".format(path=shlex.quote(path_remote), text=text)
    _ssh(host, cmd)


def _remote_exists(host: str, path_remote: str) -> bool:
    cmd = '[ -e {0} ] && echo 1 || echo 0'.format(shlex.quote(path_remote))
    return _ssh(host, cmd).strip() == '1'


def _remote_json_load(host: str, path_remote: str) -> Dict:
    fd, tmp_path = tempfile.mkstemp(prefix='nnpgen_remote_json_', suffix='.json')
    os.close(fd)
    local_path = Path(tmp_path)
    try:
        _run(['scp', '{0}:{1}'.format(host, path_remote), str(local_path)])
        with local_path.open('r') as f:
            return json.load(f)
    finally:
        if local_path.exists():
            local_path.unlink()


def _build_run_pbs_text(task_name: str, task_dir_15: str, queue: str, nodes_expr: str, walltime: str, mpirun_np: int) -> str:
    return (
        '#!/bin/bash -l\n'
        '#PBS -q {queue}\n'
        '#PBS -N {task_name}\n'
        '#PBS -l nodes={nodes_expr}\n'
        '#PBS -l walltime={walltime}\n\n'
        'module load vasp/vasp.5.4.4.pl2\n\n'
        'cd {task_dir}\n'
        'ulimit -s unlimited\n'
        'export OMP_NUM_THREADS=1\n\n'
        'mpirun -np {np} vasp_std\n'
    ).format(
        queue=queue,
        task_name=task_name,
        nodes_expr=nodes_expr,
        walltime=walltime,
        task_dir=task_dir_15,
        np=mpirun_np,
    )


def add_prepare_vasp_from_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--manifest', required=True, help='Path to manifest json on server 11')
    parser.add_argument('--job-name', default='', help='Override job name; default from manifest')
    parser.add_argument('--target-root-15', default=str(ALLOWED_15_RUN_ROOT), help='Must stay under .../SiO2/run or .../SiO2/smoke')
    parser.add_argument('--ncore', type=int, default=24)
    parser.add_argument('--queue', default='normal3')
    parser.add_argument('--nodes-expr', default='node39:ppn=24')
    parser.add_argument('--walltime', default='1000:00:00')
    parser.add_argument('--mpirun-np', type=int, default=24)
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)


def run_prepare_vasp_from_manifest(args: argparse.Namespace) -> Path:
    manifest_path = abs_path(args.manifest)
    _assert_under_project(manifest_path)

    data = read_json(manifest_path)
    if 'tasks' not in data or not isinstance(data['tasks'], list):
        raise ValueError('Invalid manifest: missing tasks list')

    job_name = args.job_name.strip() if args.job_name else str(data.get('job_name', '')).strip()
    if not job_name:
        raise ValueError('job_name is empty')

    target_root_15 = Path(args.target_root_15)
    _assert_target_root_15(target_root_15)

    job_root_15 = '{0}/{1}'.format(str(target_root_15).rstrip('/'), job_name)
    _ssh(args.host_15, 'mkdir -p {0}'.format(shlex.quote(job_root_15)))

    processed = []
    reused_existing = 0
    target_resolved = Path(str(target_root_15)).resolve()
    target_mode = 'run' if target_resolved == ALLOWED_15_RUN_ROOT.resolve() or ALLOWED_15_RUN_ROOT.resolve() in target_resolved.parents else 'smoke'

    for task in data['tasks']:
        task_name = str(task.get('task_name', '')).strip()
        poscar_path_11 = str(task.get('poscar_path_11', '')).strip()
        source_xyz_path = str(task.get('source_xyz_path', '')).strip()
        source_frame_index = task.get('source_frame_index_0based', None)

        if not task_name:
            raise ValueError('Manifest task missing task_name')
        if not poscar_path_11:
            raise ValueError('Manifest task missing poscar_path_11: {0}'.format(task_name))

        poscar_local = Path(poscar_path_11).resolve()
        if not poscar_local.exists():
            raise FileNotFoundError('Missing POSCAR on 11: {0}'.format(poscar_local))

        task_dir_15 = '{0}/{1}'.format(job_root_15, task_name)
        _ssh(args.host_15, 'mkdir -p {0}'.format(shlex.quote(task_dir_15)))
        task_info_remote = '{0}/task_info.json'.format(task_dir_15)

        if _remote_exists(args.host_15, task_info_remote):
            old_task_info = _remote_json_load(args.host_15, task_info_remote)
            old_status = str(old_task_info.get('status', 'prepared'))
            if old_status in ['finished', 'submitted', 'running']:
                reused_existing += 1
                task['status'] = old_status
                task['target_work_dir_15'] = task_dir_15
                task['prepared'] = {
                    'poscar_path_15': str(old_task_info.get('paths', {}).get('poscar', task_dir_15 + '/POSCAR')),
                    'potcar_path_15': str(old_task_info.get('paths', {}).get('potcar', task_dir_15 + '/POTCAR')),
                    'incar_path_15': str(old_task_info.get('paths', {}).get('incar', task_dir_15 + '/INCAR')),
                    'run_pbs_path_15': str(old_task_info.get('paths', {}).get('run_pbs', task_dir_15 + '/run.pbs')),
                    'task_info_path_15': task_info_remote,
                    'potcar_elements_order': old_task_info.get('potcar_elements_order', []),
                    'ncore': args.ncore,
                    'prepared_at': _now(),
                    'reused_existing': True,
                }
                processed.append(
                    {
                        'task_name': task_name,
                        'poscar_11': str(poscar_local),
                        'poscar_15': task['prepared']['poscar_path_15'],
                        'task_dir_15': task_dir_15,
                        'elements_order': old_task_info.get('potcar_elements_order', []),
                        'status': old_status,
                        'reused_existing': True,
                    }
                )
                continue

        poscar_remote = '{0}/POSCAR'.format(task_dir_15)
        _scp_to_remote(poscar_local, args.host_15, poscar_remote)

        sha_local = _sha256_local(poscar_local)
        sha_remote = _sha256_remote(args.host_15, poscar_remote)
        if sha_local != sha_remote:
            raise ValueError('POSCAR checksum mismatch for task {0}'.format(task_name))

        elements_order = _parse_poscar_elements_remote(args.host_15, poscar_remote)
        potcar_remote = _build_potcar_remote(
            host=args.host_15,
            task_dir_15=task_dir_15,
            elements_order=elements_order,
            potcar_org_dir_15=str(VASP15_POTCAR_ORG_DIR),
        )
        incar_remote = _patch_incar_remote(
            host=args.host_15,
            task_dir_15=task_dir_15,
            incar_template_15=str(VASP15_INCAR_TEMPLATE),
            ncore=args.ncore,
        )

        run_pbs_remote = '{0}/run.pbs'.format(task_dir_15)
        run_pbs_text = _build_run_pbs_text(
            task_name=task_name,
            task_dir_15=task_dir_15,
            queue=args.queue,
            nodes_expr=args.nodes_expr,
            walltime=args.walltime,
            mpirun_np=args.mpirun_np,
        )
        _write_remote_text_file(args.host_15, run_pbs_text, run_pbs_remote)
        _ssh(args.host_15, 'chmod 755 {0}'.format(shlex.quote(run_pbs_remote)))

        task_info = {
            'task_name': task_name,
            'job_id': None,
            'status': 'prepared',
            'poscar_path_11': str(poscar_local),
            'source_xyz_path': source_xyz_path,
            'source_frame_index': source_frame_index,
            'paths': {
                'task_dir_15': task_dir_15,
                'poscar': poscar_remote,
                'potcar': potcar_remote,
                'incar': incar_remote,
                'run_pbs': run_pbs_remote,
            },
            'potcar_elements_order': elements_order,
            'updated_at': _now(),
        }

        _write_remote_text_file(args.host_15, json.dumps(task_info, indent=2, sort_keys=True) + '\n', task_info_remote)

        task['status'] = 'prepared'
        task['target_work_dir_15'] = task_dir_15
        task['prepared'] = {
            'poscar_path_15': poscar_remote,
            'potcar_path_15': potcar_remote,
            'incar_path_15': incar_remote,
            'run_pbs_path_15': run_pbs_remote,
            'task_info_path_15': task_info_remote,
            'potcar_elements_order': elements_order,
            'ncore': args.ncore,
            'prepared_at': _now(),
        }

        processed.append(
            {
                'task_name': task_name,
                'poscar_11': str(poscar_local),
                'poscar_15': poscar_remote,
                'task_dir_15': task_dir_15,
                'elements_order': elements_order,
                'status': 'prepared',
                'reused_existing': False,
            }
        )

    data['job_name'] = job_name
    data['target_mode'] = target_mode
    data['target_base_15'] = job_root_15
    data['step2_prepare'] = {
        'processed_tasks': len(processed),
        'reused_existing_tasks': reused_existing,
        'ncore': args.ncore,
        'queue': args.queue,
        'nodes_expr': args.nodes_expr,
        'walltime': args.walltime,
        'mpirun_np': args.mpirun_np,
        'host_15': args.host_15,
        'prepared_at': _now(),
    }
    data['updated_at'] = _now()

    write_json(manifest_path, data)

    print('Prepared tasks: {0}'.format(len(processed)))
    for item in processed:
        print('task={0} poscar11={1} poscar15={2}'.format(item['task_name'], item['poscar_11'], item['poscar_15']))
    print('Updated manifest: {0}'.format(manifest_path))

    return manifest_path
