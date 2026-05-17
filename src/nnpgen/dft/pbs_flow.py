import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from ..utils import abs_path, ensure_dir, read_json, slugify, write_json


ALLOWED_15_ROOT = VASP15_ROOT
ALLOWED_15_SMOKE = VASP15_SMOKE_ROOT
ALLOWED_15_RUN = VASP15_RUN_ROOT
TEMPLATE_INCAR_15 = VASP15_INCAR_TEMPLATE
TEMPLATE_POTCAR_ORG_15 = VASP15_POTCAR_ORG_DIR
TEMPLATE_VASP4_PBS_15 = VASP15_ROOT / 'template' / 'script' / 'vasp.pbs'
TEMPLATE_VASPSP2DP_15 = VASP15_ROOT / 'template' / 'script' / 'vasp_sp2dpdata.py'

SUPPORTED_ELEMENTS = list(ELEMENT_ORDER)
LARGE_VASP_FILES = ['WAVECAR', 'CHGCAR', 'CHG', 'DOSCAR', 'PROCAR', 'EIGENVAL', 'REPORT', 'PCDAT', 'IBZKPT', 'XDATCAR']


def _now() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


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


def _scp_to_remote(local_path: Path, host: str, remote_path: str) -> None:
    _run(['scp', str(local_path), '{0}:{1}'.format(host, remote_path)])


def _scp_from_remote(host: str, remote_path: str, local_path: Path) -> None:
    _run(['scp', '{0}:{1}'.format(host, remote_path), str(local_path)])


def _assert_under_project(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    p = path.resolve()
    if p != root and root not in p.parents:
        raise ValueError('Path outside project root: {0}'.format(p))


def _assert_target_15(path_15: Path, allow_smoke: bool, allow_run: bool) -> None:
    p = Path(str(path_15)).resolve()
    if p != ALLOWED_15_ROOT.resolve() and ALLOWED_15_ROOT.resolve() not in p.parents:
        raise ValueError('Path outside allowed 15 root: {0}'.format(p))

    in_smoke = (p == ALLOWED_15_SMOKE.resolve()) or (ALLOWED_15_SMOKE.resolve() in p.parents)
    in_run = (p == ALLOWED_15_RUN.resolve()) or (ALLOWED_15_RUN.resolve() in p.parents)

    if not allow_smoke and in_smoke:
        raise ValueError('smoke path not allowed for this command: {0}'.format(p))
    if not allow_run and in_run:
        raise ValueError('run path not allowed for this command: {0}'.format(p))


def _read_poscar_info_local(poscar_path: Path) -> Dict:
    lines = poscar_path.read_text().splitlines()
    if len(lines) < 7:
        raise ValueError('Invalid POSCAR: {0}'.format(poscar_path))
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(elements) != len(counts):
        raise ValueError('POSCAR element/count mismatch: {0}'.format(poscar_path))

    for elem in elements:
        if elem not in SUPPORTED_ELEMENTS:
            raise ValueError('Unsupported element in POSCAR: {0} ({1})'.format(elem, poscar_path))

    natoms = sum(counts)
    return {'elements': elements, 'counts': counts, 'natoms': natoms}


def _group_key(elements: List[str], natoms: int, group_mode: str) -> str:
    if group_mode == 'single':
        return 'group_all'
    key = 'comp_{0}_n{1}'.format('_'.join(elements), natoms)
    return slugify(key).lower()


def _temp_file_with_text(text: str) -> Path:
    fd, path = tempfile.mkstemp(prefix='nnpgen_', suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    return Path(path)


def _write_remote_text(host: str, remote_path: str, text: str) -> None:
    tmp = _temp_file_with_text(text)
    try:
        _scp_to_remote(tmp, host, remote_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _remote_json_load(host: str, remote_path: str) -> Dict:
    fd, path = tempfile.mkstemp(prefix='nnpgen_json_', suffix='.json')
    os.close(fd)
    local = Path(path)
    try:
        _scp_from_remote(host, remote_path, local)
        with local.open('r') as f:
            return json.load(f)
    finally:
        if local.exists():
            local.unlink()


def _remote_json_save(host: str, remote_path: str, data: Dict) -> None:
    tmp = _temp_file_with_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    try:
        _scp_to_remote(tmp, host, remote_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _remote_find_files(host: str, root_path: str, filename: str) -> List[str]:
    cmd = 'find {0} -type f -name {1} | sort'.format(shlex.quote(root_path), shlex.quote(filename))
    out = _ssh(host, cmd)
    return [line.strip() for line in out.splitlines() if line.strip().startswith('/')]


def _remote_list_group_dirs(host: str, run_root_15: str) -> List[str]:
    cmd = 'find {0} -mindepth 1 -maxdepth 1 -type d | sort'.format(shlex.quote(run_root_15))
    out = _ssh(host, cmd)
    return [line.strip() for line in out.splitlines() if line.strip().startswith('/')]


def _remote_exists(host: str, path_remote: str) -> bool:
    cmd = '[ -e {0} ] && echo 1 || echo 0'.format(shlex.quote(path_remote))
    out = _ssh(host, cmd).strip()
    return out == '1'


def _parse_job_id(qsub_stdout: str) -> str:
    lines = [line.strip() for line in qsub_stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError('Empty qsub output')
    for line in reversed(lines):
        m = re.search(r'(\d+\.[A-Za-z0-9_.-]+)', line)
        if m:
            return m.group(1)
    for line in reversed(lines):
        tok = line.split()[0]
        if re.match(r'^\d+$', tok):
            return tok
    raise ValueError('Could not parse qsub job id from output: {0}'.format(repr(qsub_stdout)))


def _parse_qstat_detail(text: str) -> Dict:
    info = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('job_state ='):
            info['job_state'] = s.split('=', 1)[1].strip().upper()
        elif s.startswith('exit_status ='):
            raw = s.split('=', 1)[1].strip()
            try:
                info['exit_status'] = int(raw)
            except Exception:
                pass
    return info


def _query_qstat_info(host: str, job_id: str) -> Dict:
    cmd = 'qstat -f {0} 2>/dev/null || true'.format(shlex.quote(str(job_id)))
    text = _ssh(host, cmd)
    return _parse_qstat_detail(text)


def _map_qstat_to_status(job_state: str, exit_status: Optional[int]) -> str:
    s = str(job_state or '').strip().upper()
    if s in ['R', 'E']:
        return 'running'
    if s in ['Q', 'H', 'W', 'T', 'S']:
        return 'submitted'
    if s == 'C':
        if exit_status is None:
            return 'finished'
        return 'finished' if int(exit_status) == 0 else 'failed'
    return 'submitted'


def _job_status_from_qstat(host: str, job_id: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    jid = str(job_id or '').strip()
    if not jid:
        return None
    if jid in cache:
        return cache[jid]

    info = _query_qstat_info(host, jid)
    if not info.get('job_state'):
        cache[jid] = None
        return None

    status = _map_qstat_to_status(info.get('job_state', ''), info.get('exit_status'))
    cache[jid] = status
    return status


def _unit_status_from_tasks(statuses: List[str]) -> str:
    if not statuses:
        return 'prepared'
    if any(s == 'failed' for s in statuses):
        return 'failed'
    if any(s == 'running' for s in statuses):
        return 'running'
    if any(s == 'submitted' for s in statuses):
        return 'submitted'
    if all(s == 'finished' for s in statuses):
        return 'finished'
    return 'prepared'


def _build_run_task_script(execution_mode: str, mpirun_np: int) -> str:
    return (
        '#!/bin/bash\n'
        'set -euo pipefail\n'
        'MODE="${1:-' + execution_mode + '}"\n'
        'cd "$(dirname "$0")"\n'
        'if [ "$MODE" = "mock" ]; then\n'
        '  natoms=$(awk "NR==7{sum=0; for(i=1;i<=NF;i++) sum+=\$i; print sum; exit}" POSCAR)\n'
        '  cp POSCAR CONTCAR\n'
        '  {\n'
        '    echo "Mock OUTCAR"\n'
        '    echo "free  energy   TOTEN  =      -1.000000 eV"\n'
        '    echo "VOLUME and BASIS"\n'
        '    echo "d1"\n'
        '    echo "d2"\n'
        '    echo "d3"\n'
        '    echo "d4"\n'
        '    echo "10.0 0.0 0.0"\n'
        '    echo "0.0 10.0 0.0"\n'
        '    echo "0.0 0.0 10.0"\n'
        '    echo "FORCE on cell =-STRESS in cart"\n'
        '    echo "x1"\n'
        '    echo "x2"\n'
        '    echo "x3"\n'
        '    echo "x4"\n'
        '    echo "x5"\n'
        '    echo "x6"\n'
        '    echo "x7"\n'
        '    echo "x8"\n'
        '    echo "x9"\n'
        '    echo "x10"\n'
        '    echo "x11"\n'
        '    echo "x12"\n'
        '    echo "Total 1 2 3 4 5 6"\n'
        '    echo "TOTAL-FORCE"\n'
        '    echo "header1"\n'
        '    for ((i=1;i<=natoms;i++)); do\n'
        '      echo "0.000000 0.000000 0.000000 0.000000 0.000000 0.000000"\n'
        '    done\n'
        '  } > OUTCAR\n'
        '  echo "<vasprun>mock</vasprun>" > vasprun.xml\n'
        '  echo "mock finished" > OSZICAR\n'
        '  exit 0\n'
        'fi\n'
        'module load vasp/vasp.5.4.4.pl2\n'
        'export OMP_NUM_THREADS=1\n'
        'mpirun -np ${MPIRUN_NP:-' + str(mpirun_np) + '} vasp_std\n'
    )


def _build_group_pbs_script(
    queue: str,
    pbs_name: str,
    nodes_expr: str,
    walltime: str,
    group_dir: str,
    run_mode: str,
    mpirun_np: int,
) -> str:
    return (
        '#!/bin/bash -l\n'
        '#PBS -q ' + queue + '\n'
        '#PBS -N ' + pbs_name + '\n'
        '#PBS -l nodes=' + nodes_expr + '\n'
        '#PBS -l walltime=' + walltime + '\n\n'
        'set -euo pipefail\n'
        'cd ' + group_dir + '\n'
        'export MPIRUN_NP=' + str(mpirun_np) + '\n'
        'RUN_MODE="' + run_mode + '"\n\n'
        'py_update() {\n'
        'python - "$1" "$2" "$3" <<\'PY\'\n'
        'import json,sys,time\n'
        'path,status,jobid = sys.argv[1], sys.argv[2], sys.argv[3]\n'
        'with open(path) as f:\n'
        '    d=json.load(f)\n'
        'd[\'status\']=status\n'
        'if jobid and jobid != \"-\":\n'
        '    d[\'job_id\']=jobid\n'
        'd[\'updated_at\']=time.strftime(\"%Y-%m-%dT%H:%M:%SZ\", time.gmtime())\n'
        'with open(path,\'w\') as f:\n'
        '    json.dump(d,f,indent=2,sort_keys=True)\n'
        '    f.write(\"\\n\")\n'
        'PY\n'
        '}\n\n'
        'py_get_status() {\n'
        'python - "$1" <<\'PY\'\n'
        'import json,sys\n'
        'with open(sys.argv[1]) as f:\n'
        '    d=json.load(f)\n'
        'print(d.get(\'status\',\'prepared\'))\n'
        'PY\n'
        '}\n\n'
        'if [ -f group_info.json ]; then\n'
        '  py_update group_info.json running "${PBS_JOBID:-}"\n'
        'fi\n\n'
        'for t in $(ls -d task_* 2>/dev/null | sort); do\n'
        '  info="$t/task_info.json"\n'
        '  [ -f "$info" ] || continue\n'
        '  s=$(py_get_status "$info")\n'
        '  if [ "$s" = "finished" ]; then\n'
        '    continue\n'
        '  fi\n'
        '  py_update "$info" running "${PBS_JOBID:-}"\n'
        '  set +e\n'
        '  bash "$t/run_task.sh" "$RUN_MODE"\n'
        '  rc=$?\n'
        '  set -e\n'
        '  if [ "$rc" -eq 0 ]; then\n'
        '    py_update "$info" finished "${PBS_JOBID:-}"\n'
        '  else\n'
        '    py_update "$info" failed "${PBS_JOBID:-}"\n'
        '    if [ -f group_info.json ]; then\n'
        '      py_update group_info.json failed "${PBS_JOBID:-}"\n'
        '    fi\n'
        '    exit "$rc"\n'
        '  fi\n'
        'done\n\n'
        'if [ -f group_info.json ]; then\n'
        '  py_update group_info.json finished "${PBS_JOBID:-}"\n'
        'fi\n'
    )


def _patch_incar_text(template_text: str, ncore: int) -> str:
    lines = template_text.splitlines()
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


def add_prepare_seq_pbs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--job-name', default='')
    parser.add_argument('--target-root-15', default=str(ALLOWED_15_RUN))
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)
    parser.add_argument('--group-mode', choices=['composition', 'single'], default='composition')
    parser.add_argument('--execution-mode', choices=['vasp', 'mock'], default='vasp')
    parser.add_argument('--ncore', type=int, default=24)
    parser.add_argument('--queue', default='normal3')
    parser.add_argument('--nodes-expr', default='node39:ppn=24')
    parser.add_argument('--walltime', default='1000:00:00')
    parser.add_argument('--mpirun-np', type=int, default=24)
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--clean-target', action='store_true')


def run_prepare_seq_pbs_from_manifest(args: argparse.Namespace) -> Path:
    manifest_path = abs_path(args.manifest)
    _assert_under_project(manifest_path)
    manifest = read_json(manifest_path)

    tasks = list(manifest.get('tasks', []))
    if args.max_tasks is not None and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise ValueError('No tasks found in manifest')

    job_name = args.job_name.strip() if args.job_name else str(manifest.get('job_name', '')).strip()
    if not job_name:
        raise ValueError('job_name is empty')

    target_root_15 = Path(args.target_root_15)
    _assert_target_15(target_root_15, allow_smoke=True, allow_run=True)

    run_root_15 = str(target_root_15).rstrip('/') + '/' + job_name
    if args.clean_target:
        _ssh(args.host_15, 'rm -rf {0}'.format(shlex.quote(run_root_15)))
    _ssh(args.host_15, 'mkdir -p {0}'.format(shlex.quote(run_root_15)))

    incar_template_local = _temp_file_with_text('')
    try:
        _scp_from_remote(args.host_15, str(TEMPLATE_INCAR_15), incar_template_local)
        incar_template_text = incar_template_local.read_text()
    finally:
        if incar_template_local.exists():
            incar_template_local.unlink()

    grouped = OrderedDict()
    task_meta = []
    for i, task in enumerate(tasks, start=1):
        poscar_path_11 = Path(str(task.get('poscar_path_11', ''))).resolve()
        if not poscar_path_11.exists():
            raise FileNotFoundError('Missing POSCAR on 11: {0}'.format(poscar_path_11))
        pinfo = _read_poscar_info_local(poscar_path_11)
        gk = _group_key(pinfo['elements'], pinfo['natoms'], args.group_mode)
        grouped.setdefault(gk, []).append((task, pinfo, poscar_path_11, i))

    group_records = []
    group_index = 0

    for group_key, entries in grouped.items():
        group_index += 1
        group_name = 'group_{0:03d}_{1}'.format(group_index, group_key)
        group_name = slugify(group_name).lower()
        group_dir_15 = run_root_15 + '/' + group_name
        _ssh(args.host_15, 'mkdir -p {0}'.format(shlex.quote(group_dir_15)))

        first_info = entries[0][1]
        elems = first_info['elements']
        natoms = first_info['natoms']

        sub_idx = 0
        for task, pinfo, poscar_path_11, original_idx in entries:
            sub_idx += 1
            sub_name = 'task_{0:03d}'.format(sub_idx)
            sub_dir_15 = group_dir_15 + '/' + sub_name
            _ssh(args.host_15, 'mkdir -p {0}'.format(shlex.quote(sub_dir_15)))

            poscar_remote = sub_dir_15 + '/POSCAR'
            _scp_to_remote(poscar_path_11, args.host_15, poscar_remote)

            pot_sources = [str(TEMPLATE_POTCAR_ORG_15 / ('POTCAR_' + e)) for e in pinfo['elements']]
            pot_source_expr = ' '.join(shlex.quote(x) for x in pot_sources)
            potcar_remote = sub_dir_15 + '/POTCAR'
            pot_cmd = (
                'set -e; '
                'for f in {src}; do [ -f "$f" ] || {{ echo "Missing POTCAR source: $f" >&2; exit 1; }}; done; '
                'cat {src} > {out}'
            ).format(src=pot_source_expr, out=shlex.quote(potcar_remote))
            _ssh(args.host_15, pot_cmd)

            incar_text = _patch_incar_text(incar_template_text, args.ncore)
            incar_remote = sub_dir_15 + '/INCAR'
            _write_remote_text(args.host_15, incar_remote, incar_text)

            run_task_text = _build_run_task_script(args.execution_mode, args.mpirun_np)
            run_task_remote = sub_dir_15 + '/run_task.sh'
            _write_remote_text(args.host_15, run_task_remote, run_task_text)
            _ssh(args.host_15, 'chmod 755 {0}'.format(shlex.quote(run_task_remote)))

            task_info = {
                'task_name': sub_name,
                'status': 'prepared',
                'job_id': None,
                'group_name': group_name,
                'group_dir_15': group_dir_15,
                'task_dir_15': sub_dir_15,
                'source_manifest_task_name': task.get('task_name', ''),
                'poscar_path_11': str(poscar_path_11),
                'source_xyz_path': task.get('source_xyz_path', ''),
                'source_frame_index': task.get('source_frame_index_0based', None),
                'elements_order': pinfo['elements'],
                'atom_counts': pinfo['counts'],
                'natoms': pinfo['natoms'],
                'paths': {
                    'poscar': poscar_remote,
                    'potcar': potcar_remote,
                    'incar': incar_remote,
                    'run_task': run_task_remote,
                },
                'updated_at': _now(),
            }
            task_info_remote = sub_dir_15 + '/task_info.json'
            _write_remote_text(args.host_15, task_info_remote, json.dumps(task_info, indent=2, sort_keys=True) + '\n')

            task['status'] = 'prepared'
            task['pbs_group_name'] = group_name
            task['pbs_subtask_name'] = sub_name
            task['target_work_dir_15'] = sub_dir_15
            task['prepared_step3'] = {
                'group_dir_15': group_dir_15,
                'task_dir_15': sub_dir_15,
                'poscar_path_15': poscar_remote,
                'potcar_path_15': potcar_remote,
                'incar_path_15': incar_remote,
                'elements_order': pinfo['elements'],
                'natoms': pinfo['natoms'],
                'prepared_at': _now(),
            }

            task_meta.append({'group_name': group_name, 'task_name': sub_name, 'task_dir_15': sub_dir_15})

        pbs_name = slugify(group_name)[:60]
        run_pbs_text = _build_group_pbs_script(
            queue=args.queue,
            pbs_name=pbs_name,
            nodes_expr=args.nodes_expr,
            walltime=args.walltime,
            group_dir=group_dir_15,
            run_mode=args.execution_mode,
            mpirun_np=args.mpirun_np,
        )
        run_pbs_remote = group_dir_15 + '/run.pbs'
        _write_remote_text(args.host_15, run_pbs_remote, run_pbs_text)
        _ssh(args.host_15, 'chmod 755 {0}'.format(shlex.quote(run_pbs_remote)))

        group_info = {
            'group_name': group_name,
            'status': 'prepared',
            'job_id': None,
            'run_pbs': run_pbs_remote,
            'group_dir_15': group_dir_15,
            'execution_mode': args.execution_mode,
            'queue': args.queue,
            'nodes_expr': args.nodes_expr,
            'walltime': args.walltime,
            'mpirun_np': args.mpirun_np,
            'ncore': args.ncore,
            'elements_order': elems,
            'natoms': natoms,
            'task_count': len(entries),
            'updated_at': _now(),
        }
        _write_remote_text(args.host_15, group_dir_15 + '/group_info.json', json.dumps(group_info, indent=2, sort_keys=True) + '\n')

        group_records.append(group_info)

    manifest['step3_prepare'] = {
        'prepared_at': _now(),
        'job_name': job_name,
        'run_root_15': run_root_15,
        'group_mode': args.group_mode,
        'execution_mode': args.execution_mode,
        'group_count': len(group_records),
        'task_count': len(tasks),
    }
    manifest['updated_at'] = _now()
    write_json(manifest_path, manifest)

    print('Prepared sequential PBS layout: groups={0} tasks={1} root15={2}'.format(len(group_records), len(tasks), run_root_15))
    return manifest_path


def add_submit_pbs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--run-root-15', required=True)
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)
    parser.add_argument('--max-groups', type=int, default=None)


def run_submit_pbs(args: argparse.Namespace) -> List[str]:
    _assert_target_15(Path(args.run_root_15), allow_smoke=True, allow_run=True)

    unit_dirs = _remote_list_group_dirs(args.host_15, args.run_root_15)
    if args.max_groups is not None and args.max_groups > 0:
        unit_dirs = unit_dirs[: args.max_groups]

    job_ids = []
    qstat_cache = {}

    for unit_dir in unit_dirs:
        run_pbs = unit_dir + '/run.pbs'
        if not _remote_exists(args.host_15, run_pbs):
            continue

        ti_path = unit_dir + '/task_info.json'
        gi_path = unit_dir + '/group_info.json'
        is_task_unit = _remote_exists(args.host_15, ti_path)
        is_group_unit = _remote_exists(args.host_15, gi_path)

        if not is_task_unit and not is_group_unit:
            continue

        info_path = ti_path if is_task_unit else gi_path
        info = _remote_json_load(args.host_15, info_path)
        status = str(info.get('status', 'prepared'))
        existing_job_id = str(info.get('job_id') or '').strip()

        if status == 'finished':
            print('Skip finished unit: {0}'.format(unit_dir))
            continue

        qstat_status = None
        if existing_job_id:
            qstat_status = _job_status_from_qstat(args.host_15, existing_job_id, qstat_cache)

        if qstat_status in ['submitted', 'running']:
            if status != qstat_status:
                info['status'] = qstat_status
                info['updated_at'] = _now()
                _remote_json_save(args.host_15, info_path, info)
            print('Skip active unit: {0} job_id={1} status={2}'.format(unit_dir, existing_job_id, qstat_status))
            continue

        if qstat_status == 'finished':
            info['status'] = 'finished'
            info['updated_at'] = _now()
            _remote_json_save(args.host_15, info_path, info)
            print('Skip already finished unit: {0} job_id={1}'.format(unit_dir, existing_job_id))
            continue

        if status in ['submitted', 'running'] and existing_job_id and qstat_status is None:
            print('Skip uncertain active unit (no qstat record): {0} job_id={1}'.format(unit_dir, existing_job_id))
            continue

        if qstat_status == 'failed' and status != 'failed':
            info['status'] = 'failed'
            info['updated_at'] = _now()
            _remote_json_save(args.host_15, info_path, info)

        out = _ssh(args.host_15, 'cd {0} && qsub run.pbs'.format(shlex.quote(unit_dir)))
        job_id = _parse_job_id(out)
        job_ids.append(job_id)

        info = _remote_json_load(args.host_15, info_path)
        info['status'] = 'submitted'
        info['job_id'] = job_id
        info['updated_at'] = _now()
        _remote_json_save(args.host_15, info_path, info)

        task_infos = _remote_find_files(args.host_15, unit_dir, 'task_info.json')
        for tip in task_infos:
            ti = _remote_json_load(args.host_15, tip)
            if str(ti.get('status', 'prepared')) != 'finished':
                ti['status'] = 'submitted'
                ti['job_id'] = job_id
                ti['updated_at'] = _now()
                _remote_json_save(args.host_15, tip, ti)

        label = 'task' if is_task_unit else 'group'
        print('Submitted {0}: {1} job_id={2}'.format(label, unit_dir, job_id))

    print('Total units submitted: {0}'.format(len(job_ids)))
    return job_ids


def _collect_task_infos_remote(host: str, run_root_15: str) -> List[Tuple[str, Dict]]:
    task_paths = _remote_find_files(host, run_root_15, 'task_info.json')
    rows = []
    for p in task_paths:
        try:
            rows.append((p, _remote_json_load(host, p)))
        except Exception:
            continue
    return rows


def run_status_pbs(run_root_15: str, host_15: str = '15') -> Dict[str, int]:
    _assert_target_15(Path(run_root_15), allow_smoke=True, allow_run=True)

    rows = _collect_task_infos_remote(host_15, run_root_15)
    qstat_cache = {}
    counts = {
        'total': 0,
        'prepared': 0,
        'submitted': 0,
        'running': 0,
        'finished': 0,
        'failed': 0,
    }
    unit_statuses = {}

    for ti_path, ti in rows:
        s = str(ti.get('status', 'prepared'))
        job_id = str(ti.get('job_id') or '').strip()

        if job_id:
            qstatus = _job_status_from_qstat(host_15, job_id, qstat_cache)
            if qstatus in counts:
                s = qstatus

        task_dir_15 = str(ti.get('task_dir_15') or ti.get('paths', {}).get('task_dir_15', ''))
        if not job_id and s != 'failed' and task_dir_15:
            if _remote_exists(host_15, task_dir_15 + '/OUTCAR') and _remote_exists(host_15, task_dir_15 + '/OSZICAR'):
                s = 'finished'

        if s not in counts:
            s = 'prepared'

        if str(ti.get('status', 'prepared')) != s:
            ti['status'] = s
            ti['updated_at'] = _now()
            _remote_json_save(host_15, ti_path, ti)

        unit_key = str(ti.get('group_dir_15') or task_dir_15 or '')
        if unit_key:
            unit_statuses.setdefault(unit_key, []).append(s)

        counts['total'] += 1
        counts[s] += 1

    for unit_dir, statuses in unit_statuses.items():
        gi_path = unit_dir.rstrip('/') + '/group_info.json'
        if not _remote_exists(host_15, gi_path):
            continue
        gi = _remote_json_load(host_15, gi_path)
        g_status = _unit_status_from_tasks(statuses)
        if str(gi.get('status', 'prepared')) != g_status:
            gi['status'] = g_status
            gi['updated_at'] = _now()
            _remote_json_save(host_15, gi_path, gi)

    print(
        'run_root_15={0} total={1} submitted={2} running={3} finished={4} failed={5}'.format(
            run_root_15,
            counts['total'],
            counts['submitted'],
            counts['running'],
            counts['finished'],
            counts['failed'],
        )
    )
    return counts


def add_collect_results_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--run-root-15', required=True)
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)
    parser.add_argument('--output-root-11', required=True)


def run_collect_results(args: argparse.Namespace) -> Path:
    _assert_target_15(Path(args.run_root_15), allow_smoke=True, allow_run=True)

    output_root = abs_path(args.output_root_11)
    _assert_under_project(output_root)
    ensure_dir(output_root)

    rows = _collect_task_infos_remote(args.host_15, args.run_root_15)
    collected = []

    for _, ti in rows:
        if ti.get('status') != 'finished':
            continue

        group_name = str(ti.get('group_name', 'group_unknown'))
        task_name = str(ti.get('task_name', 'task_unknown'))
        task_dir_15 = str(ti.get('task_dir_15', ''))
        if not task_dir_15:
            continue

        local_task_dir = ensure_dir(output_root / group_name / task_name)

        copied = []
        for fn in ['POSCAR', 'CONTCAR', 'OUTCAR', 'vasprun.xml', 'OSZICAR', 'task_info.json']:
            remote_file = task_dir_15 + '/' + fn
            local_file = local_task_dir / fn
            try:
                _scp_from_remote(args.host_15, remote_file, local_file)
                copied.append(fn)
            except Exception:
                pass

        collected.append(
            {
                'group_name': group_name,
                'task_name': task_name,
                'task_dir_15': task_dir_15,
                'local_task_dir': str(local_task_dir),
                'copied_files': copied,
            }
        )

    collect_manifest = {
        'schema_version': 'nnpgen.dft.collect.v1',
        'created_at': _now(),
        'run_root_15': args.run_root_15,
        'output_root_11': str(output_root),
        'finished_tasks_collected': len(collected),
        'items': collected,
    }

    out_json = output_root / 'collect_manifest.json'
    write_json(out_json, collect_manifest)
    print('Collected finished tasks: {0} output={1}'.format(len(collected), output_root))
    return out_json


def _build_typemap(elements: List[str]) -> Dict[str, int]:
    order = [e for e in SUPPORTED_ELEMENTS if e in elements]
    return {e: i for i, e in enumerate(order)}


def _patch_vaspsp2dp_template(template_text: str, data_path: str, atom_num: int, typemap: Dict[str, int]) -> str:
    text = template_text
    text = re.sub(r'^path\s*=\s*.*$', 'path={0}'.format(repr(data_path)), text, flags=re.M)
    text = re.sub(r'^atom_num\s*=\s*\d+\s*$', 'atom_num={0}'.format(int(atom_num)), text, flags=re.M)
    text = re.sub(r'^tyep_map\s*=\s*\{.*\}\s*$', 'tyep_map={0}'.format(repr(typemap)), text, flags=re.M)
    text = text.replace('os.makedirs(npy_fold,exist_ok=True)', 'if not os.path.exists(npy_fold):\n    os.makedirs(npy_fold)')
    return text


def add_convert_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--run-root-15', required=True)
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)
    parser.add_argument('--dataset-root-15', required=True)
    parser.add_argument('--dataset-index-11', required=True)
    parser.add_argument('--cleanup-large', action='store_true')


def run_convert_dataset(args: argparse.Namespace) -> Path:
    _assert_target_15(Path(args.run_root_15), allow_smoke=True, allow_run=True)
    _assert_target_15(Path(args.dataset_root_15), allow_smoke=True, allow_run=True)

    dataset_index_11 = abs_path(args.dataset_index_11)
    _assert_under_project(dataset_index_11)
    ensure_dir(dataset_index_11.parent)

    rows = _collect_task_infos_remote(args.host_15, args.run_root_15)
    by_unit = OrderedDict()
    for _, ti in rows:
        if ti.get('status') != 'finished':
            continue
        g = str(ti.get('group_name', 'group_unknown'))
        elems = [str(x) for x in ti.get('elements_order', [])]
        natoms = int(ti.get('natoms', 0))
        unit = slugify('{0}_n{1}_{2}'.format(g, natoms, '_'.join(elems))).lower()
        by_unit.setdefault(unit, []).append(ti)

    if not by_unit:
        manifest = {
            'schema_version': 'nnpgen.dft.dataset_index.v1',
            'created_at': _now(),
            'run_root_15': args.run_root_15,
            'dataset_root_15': args.dataset_root_15,
            'converted_groups': 0,
            'items': [],
        }
        write_json(dataset_index_11, manifest)
        print('No finished tasks for conversion')
        return dataset_index_11

    template_local = _temp_file_with_text('')
    try:
        _scp_from_remote(args.host_15, str(TEMPLATE_VASPSP2DP_15), template_local)
        template_text = template_local.read_text()
    finally:
        if template_local.exists():
            template_local.unlink()

    items = []

    for unit_name, tasks in by_unit.items():
        group_dir = str(tasks[0].get('group_dir_15'))
        conv_dir = group_dir + '/_dp_convert_' + unit_name
        out_unit_dir = str(Path(args.dataset_root_15) / unit_name)

        _ssh(args.host_15, 'rm -rf {0} && mkdir -p {0}'.format(shlex.quote(conv_dir)))
        _ssh(args.host_15, 'mkdir -p {0}'.format(shlex.quote(out_unit_dir)))

        copied_count = 0
        for idx, ti in enumerate(tasks, start=1):
            src_task_dir = str(ti.get('task_dir_15'))
            outcar_remote = src_task_dir + '/OUTCAR'
            poscar_remote = src_task_dir + '/POSCAR'
            if not _remote_exists(args.host_15, outcar_remote):
                continue
            if not _remote_exists(args.host_15, poscar_remote):
                continue

            copied_count += 1
            dst_idx_dir = conv_dir + '/{0}'.format(copied_count)
            _ssh(args.host_15, 'mkdir -p {0}'.format(shlex.quote(dst_idx_dir)))
            _ssh(
                args.host_15,
                'cp {0} {1}/POSCAR && cp {2} {1}/OUTCAR'.format(
                    shlex.quote(poscar_remote),
                    shlex.quote(dst_idx_dir),
                    shlex.quote(outcar_remote),
                ),
            )

        if copied_count == 0:
            continue

        first_poscar = Path(str(tasks[0].get('poscar_path_11', '')))
        pinfo = _read_poscar_info_local(first_poscar)
        typemap = _build_typemap(pinfo['elements'])
        patched_script = _patch_vaspsp2dp_template(template_text, conv_dir, pinfo['natoms'], typemap)

        patched_local = _temp_file_with_text(patched_script)
        try:
            _scp_to_remote(patched_local, args.host_15, conv_dir + '/VASPsp2dpdata.py')
        finally:
            if patched_local.exists():
                patched_local.unlink()

        _ssh(args.host_15, 'cd {0} && python VASPsp2dpdata.py'.format(shlex.quote(conv_dir)))

        _ssh(
            args.host_15,
            'cp {0}/energy.raw {1}/energy.raw; '
            'cp {0}/box.raw {1}/box.raw; '
            'cp {0}/coord.raw {1}/coord.raw; '
            'cp {0}/force.raw {1}/force.raw; '
            'cp {0}/virial.raw {1}/virial.raw; '
            'cp {0}/type.raw {1}/type.raw; '
            'cp {0}/type_map.raw {1}/type_map.raw; '
            'mkdir -p {1}/set.000; '
            'cp {0}/set.000/*.npy {1}/set.000/'.format(shlex.quote(conv_dir), shlex.quote(out_unit_dir)),
        )

        if args.cleanup_large:
            for ti in tasks:
                td = str(ti.get('task_dir_15'))
                rm_expr = ' '.join(shlex.quote(td + '/' + fn) for fn in LARGE_VASP_FILES)
                _ssh(args.host_15, 'rm -f {0}'.format(rm_expr))

        items.append(
            {
                'group_name': str(tasks[0].get('group_name', 'group_unknown')),
                'unit_name': unit_name,
                'task_count': copied_count,
                'dataset_group_dir_15': out_unit_dir,
                'example_files': [
                    out_unit_dir + '/energy.raw',
                    out_unit_dir + '/coord.raw',
                    out_unit_dir + '/set.000/energy.npy',
                ],
            }
        )

    manifest = {
        'schema_version': 'nnpgen.dft.dataset_index.v1',
        'created_at': _now(),
        'run_root_15': args.run_root_15,
        'dataset_root_15': args.dataset_root_15,
        'converted_groups': len(items),
        'items': items,
    }
    write_json(dataset_index_11, manifest)
    print('Converted dataset groups: {0} dataset_root_15={1}'.format(len(items), args.dataset_root_15))
    return dataset_index_11


def run_status_router(run_root: str, host_15: str, legacy_status_func, legacy_args) -> Dict[str, int]:
    if run_root.startswith(str(ALLOWED_15_ROOT)):
        return run_status_pbs(run_root_15=run_root, host_15=host_15)
    return legacy_status_func(legacy_args)
