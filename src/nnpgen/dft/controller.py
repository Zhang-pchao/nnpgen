import argparse
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

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
ALLOWED_27_ROOT = Path(os.environ.get('NNPGEN_SLURM_ROOT', str(Path.cwd() / 'remote_slurm')))
ALLOWED_27_USERS_ROOT = Path(os.environ.get('NNPGEN_SLURM_ALT_ROOT', str(Path.cwd() / 'remote_slurm_alt')))
ALLOWED_SLURM_ROOTS = (ALLOWED_27_ROOT, ALLOWED_27_USERS_ROOT)
TEMPLATE_POTCAR_ORG_15 = VASP15_POTCAR_ORG_DIR
TEMPLATE_INCAR_15 = VASP15_INCAR_TEMPLATE
DEFAULT_TEMPLATE_INCAR_27 = os.environ.get('NNPGEN_SLURM_INCAR_TEMPLATE', str(ALLOWED_27_ROOT / 'template' / 'INCAR'))
DEFAULT_TEMPLATE_POTCAR_27 = os.environ.get('NNPGEN_SLURM_POTCAR_SOURCE', str(ALLOWED_27_ROOT / 'template' / 'POTCAR'))
DEFAULT_VASP_ENV_SCRIPT_27 = os.environ.get('NNPGEN_SLURM_VASP_ENV_SCRIPT', '')
DEFAULT_TEMPLATE_INCAR_28 = os.environ.get('NNPGEN_SLURM_ALT_INCAR_TEMPLATE', str(ALLOWED_27_USERS_ROOT / 'template' / 'INCAR'))
DEFAULT_TEMPLATE_POTCAR_28 = os.environ.get('NNPGEN_SLURM_ALT_POTCAR_SOURCE', str(ALLOWED_27_USERS_ROOT / 'template' / 'POTCAR'))
DEFAULT_VASP_ENV_SCRIPT_28 = os.environ.get('NNPGEN_SLURM_ALT_VASP_ENV_SCRIPT', '')

DEFAULT_JOB_NAME = os.environ.get('NNPGEN_STAGE2_JOB_NAME', 'stage2_dft')
DEFAULT_STAGE1_MANIFEST = DEFAULT_PLAN_ROOT / 'stage1_manifest.json'
DEFAULT_STAGE1_RUN_ROOT = RUN_ROOT / 'stage1_md'
DEFAULT_STAGE2_MANIFEST = DEFAULT_PLAN_ROOT / (DEFAULT_JOB_NAME + '_manifest.json')
DEFAULT_STAGE2_RECORD = DEFAULT_PLAN_ROOT / (DEFAULT_JOB_NAME + '_record.md')
DEFAULT_STAGE2_CONTROLLER_DIR = RUN_ROOT / DEFAULT_JOB_NAME / 'controller'
DEFAULT_STAGE2_CONTROLLER_STATE = DEFAULT_STAGE2_CONTROLLER_DIR / 'controller_state.json'
DEFAULT_STAGE2_CONTROLLER_LOG = DEFAULT_STAGE2_CONTROLLER_DIR / 'controller.log'
DEFAULT_STAGE2_PID_FILE = DEFAULT_STAGE2_CONTROLLER_DIR / 'controller.pid'
DEFAULT_RUN_ROOT_15 = VASP15_RUN_ROOT / DEFAULT_JOB_NAME
DEFAULT_RUN_ROOT_27 = ALLOWED_27_ROOT / 'run' / DEFAULT_JOB_NAME
DEFAULT_RUN_ROOT_28 = ALLOWED_27_USERS_ROOT / 'run' / DEFAULT_JOB_NAME

SUPPORTED_ELEMENTS = set(ELEMENT_ORDER)
AVOID_NODES = {'node01', 'node02', 'node33', 'node34', 'node36'}
DEFAULT_SLURM_PARTITIONS = os.environ.get('NNPGEN_SLURM_PARTITIONS', 'standard')
DEFAULT_SLURM_PARTITION_CORES = os.environ.get('NNPGEN_SLURM_PARTITION_CORES', 'standard=24')
SLURM_IDLE_STATES = {'idle'}

_SSH_EXTRA_ARGS = []


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _run(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def _set_ssh_extra_args(args):
    global _SSH_EXTRA_ARGS
    _SSH_EXTRA_ARGS = list(args)


def _ssh(host, command):
    return _run(['ssh'] + _SSH_EXTRA_ARGS + [host, command]).stdout


def _tmp_text_file(text, suffix='.tmp'):
    fd, path = tempfile.mkstemp(prefix='nnpgen_stage2_step20_', suffix=suffix)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    return Path(path)


def _scp_to_remote_atomic(local_path, host, remote_path):
    last_err = None
    for _ in range(3):
        token = '{0}_{1}'.format(os.getpid(), random.randint(1000, 999999))
        remote_tmp = remote_path + '.tmp_' + token
        try:
            _run(['scp'] + _SSH_EXTRA_ARGS + [str(local_path), '{0}:{1}'.format(host, remote_tmp)])
            _ssh(host, 'mv {0} {1}'.format(shlex.quote(remote_tmp), shlex.quote(remote_path)))
            return
        except subprocess.CalledProcessError as ex:
            last_err = ex
            time.sleep(1)
    try:
        _run(['scp'] + _SSH_EXTRA_ARGS + [str(local_path), '{0}:{1}'.format(host, remote_path)])
        return
    except subprocess.CalledProcessError:
        if last_err is not None:
            raise last_err
        raise


def _scp_from_remote(host, remote_path, local_path):
    _run(['scp'] + _SSH_EXTRA_ARGS + ['{0}:{1}'.format(host, remote_path), str(local_path)])


def _remote_exists(host, path_remote):
    out = _ssh(host, '[ -e {0} ] && echo 1 || echo 0'.format(shlex.quote(path_remote))).strip()
    return out == '1'


def _remote_mkdir(host, path_remote):
    _ssh(host, 'mkdir -p {0}'.format(shlex.quote(path_remote)))


def _remote_read_json(host, path_remote):
    out = _ssh(host, 'cat {0}'.format(shlex.quote(path_remote))).strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # Tolerate transient/partial task_info writes on remote hosts.
        # Caller will refresh status and rewrite a clean JSON file.
        return {}


def _remote_write_json(host, path_remote, data):
    tmp = _tmp_text_file(json.dumps(data, indent=2, sort_keys=True) + '\n', suffix='.json')
    try:
        _scp_to_remote_atomic(tmp, host, path_remote)
    finally:
        if tmp.exists():
            tmp.unlink()


def _remote_write_text(host, path_remote, text):
    tmp = _tmp_text_file(text, suffix='.txt')
    try:
        _scp_to_remote_atomic(tmp, host, path_remote)
    finally:
        if tmp.exists():
            tmp.unlink()


def _assert_under_11(path):
    p = path.resolve()
    root = ALLOWED_11_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError('Path outside allowed 11 root: {0}'.format(p))


def _assert_under_15(path_15):
    p = Path(path_15).resolve()
    root = ALLOWED_15_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError('Path outside allowed 15 root: {0}'.format(p))


def _assert_under_27(path_27):
    p = Path(path_27).resolve()
    for root in ALLOWED_SLURM_ROOTS:
        r = root.resolve()
        if p == r or r in p.parents:
            return
    allowed = ', '.join(str(x.resolve()) for x in ALLOWED_SLURM_ROOTS)
    raise ValueError('Path outside allowed slurm roots ({0}): {1}'.format(allowed, p))


def _parse_system_index(system_name):
    m = re.search(r'^system_(\d+)', str(system_name or ''))
    if not m:
        return None
    return int(m.group(1))


def _slurm_route_host(system_name, host_27, host_28):
    idx = _parse_system_index(system_name)
    if idx is None:
        return str(host_27)
    return str(host_27) if (idx % 2 == 1) else str(host_28)


def _parse_frame_index_1based(frame_name_or_entry):
    if isinstance(frame_name_or_entry, dict):
        frame_name = str(frame_name_or_entry.get('frame_name') or '')
        m = re.search(r'(\d+)$', frame_name)
        if m:
            return int(m.group(1))
        idx0 = frame_name_or_entry.get('source_frame_index_0based')
        if idx0 is None:
            raise ValueError('Cannot derive frame index')
        return int(idx0) + 1
    m = re.search(r'(\d+)$', str(frame_name_or_entry))
    if not m:
        raise ValueError('Cannot parse frame index from {0}'.format(frame_name_or_entry))
    return int(m.group(1))


def _parse_poscar_local(poscar_path):
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
    return {'elements': elems, 'counts': counts, 'atom_count': sum(counts)}


def _patch_incar_ncore(incar_text, ncore):
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


def _node_queue_cores(node):
    m = re.match(r'^node(\d+)$', node)
    if not m:
        return '', 0
    n = int(m.group(1))
    if node in {'node41', 'node42'}:
        return 'normal4', 28
    if node == 'node43':
        return 'normal4', 32
    if 33 <= n <= 40:
        return 'normal3', 24
    if 17 <= n <= 32:
        return 'normal2', 20
    if 1 <= n <= 16:
        return 'normal1', 16
    return '', 0


def _node_number(node):
    m = re.match(r'^node(\d+)$', node)
    return int(m.group(1)) if m else 999


def _parse_pestat_lines(text):
    rows = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith('Netload file') or s.startswith('Node'):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        node = parts[0]
        if not re.match(r'^node\d+$', node):
            continue
        state = parts[1].replace('*', '').lower()
        tasks = None
        if len(parts) > 8:
            t = parts[8].replace('*', '')
            if t.isdigit():
                tasks = int(t)
        queue, cores = _node_queue_cores(node)
        if not queue:
            continue
        rows.append({'node': node, 'state': state, 'tasks': tasks, 'queue': queue, 'cores': cores})
    return rows


def _collect_submit_slots(host_15):
    rows = _parse_pestat_lines(_ssh(host_15, 'pestat'))
    free = []
    for r in rows:
        if r['node'] in AVOID_NODES:
            continue
        if r['state'] != 'free':
            continue
        if r['tasks'] is not None and r['tasks'] != 0:
            continue
        free.append(r)

    groups = {'normal1': [], 'normal2': [], 'normal3': [], 'normal4': []}
    for r in free:
        groups[r['queue']].append(r)

    for q in groups:
        groups[q].sort(key=lambda x: _node_number(x['node']))

    if len(groups['normal1']) > 4:
        groups['normal1'] = groups['normal1'][4:]
    else:
        groups['normal1'] = []

    ordered = []
    for q in ['normal3', 'normal4', 'normal2', 'normal1']:
        ordered.extend(groups[q])

    return ordered


def _parse_qstat_jobs(text):
    jobs = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith('master:') or s.startswith('Job ID') or s.startswith('Req') or s.startswith('---'):
            continue
        parts = s.split()
        if len(parts) >= 10:
            jobs.append({'job_id': parts[0], 'job_name': parts[3], 'state': parts[-2].upper()})
        elif len(parts) >= 6:
            jobs.append({'job_id': parts[0], 'job_name': parts[1], 'state': parts[4].upper()})
    return jobs


def _qstat_user_jobs(host_15):
    user = _ssh(host_15, 'whoami').strip()
    out = _ssh(host_15, 'qstat -u {0} 2>/dev/null || true'.format(shlex.quote(user)))
    return _parse_qstat_jobs(out)


def _jobs_by_name(host_15, job_name):
    return [j for j in _qstat_user_jobs(host_15) if j.get('job_name') == job_name]


def _qstat_state_by_jobid(host_15, job_id):
    if not job_id:
        return ''
    out = _ssh(host_15, 'qstat -f {0} 2>/dev/null || true'.format(shlex.quote(job_id)))
    for line in out.splitlines():
        s = line.strip()
        if s.startswith('job_state') and '=' in s:
            return s.split('=', 1)[1].strip().upper()
    return ''


def _map_qstate_to_status(qstate):
    s = str(qstate or '').strip().upper()
    if s in {'Q', 'H', 'W', 'T'}:
        return 'submitted'
    if s in {'R', 'E'}:
        return 'running'
    if s in {'C'}:
        return 'finished'
    return ''


def _preferred_job(jobs):
    if not jobs:
        return {}
    for st in ['R', 'E', 'Q', 'H', 'W', 'T', 'C']:
        for j in jobs:
            if j.get('state', '').upper() == st:
                return j
    return jobs[0]


def _preferred_slurm_job(jobs):
    if not jobs:
        return {}
    for st in ['RUNNING', 'COMPLETING', 'PENDING', 'CONFIGURING', 'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
        for j in jobs:
            if j.get('state', '').upper() == st:
                return j
    return jobs[0]


def _parse_partition_cores(spec):
    out = {}
    for raw in str(spec or '').split(','):
        s = raw.strip()
        if not s or '=' not in s:
            continue
        k, v = s.split('=', 1)
        k = k.strip()
        try:
            out[k] = int(v.strip())
        except Exception:
            continue
    return out


def _parse_sinfo_node_lines(text):
    rows = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        partition = parts[0].replace('*', '')
        node = parts[1]
        state = parts[2].replace('*', '').lower()
        rows.append({'partition': partition, 'node': node, 'state': state})
    return rows


def _collect_submit_slots_slurm(host_27, partitions, partition_cores):
    plist = [p.strip() for p in str(partitions or '').split(',') if p.strip()]
    if not plist:
        return []
    cmd = 'sinfo -N -h -p {0} -o \"%P %N %t\"'.format(shlex.quote(','.join(plist)))
    rows = _parse_sinfo_node_lines(_ssh(host_27, cmd))

    by_partition = {}
    for p in plist:
        by_partition[p] = []
    for r in rows:
        part = r['partition']
        if part not in by_partition:
            continue
        if r['state'] not in SLURM_IDLE_STATES:
            continue
        by_partition[part].append(r['node'])

    used_nodes = set()
    slots = []
    for part in plist:
        cores = int(partition_cores.get(part, 0))
        if cores <= 0:
            continue
        for node in sorted(set(by_partition.get(part, []))):
            if node in used_nodes:
                continue
            slots.append({'partition': part, 'node': node, 'cores': cores})
            used_nodes.add(node)
    return slots


def _parse_squeue_jobs(text):
    jobs = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        jobs.append({'job_id': parts[0], 'state': parts[1].upper()})
    return jobs


def _slurm_jobs_by_name(host_27, job_name):
    user = _ssh(host_27, 'whoami').strip()
    cmd = 'squeue -h -u {0} -n {1} -o \"%A %T\" 2>/dev/null || true'.format(
        shlex.quote(user),
        shlex.quote(job_name),
    )
    return _parse_squeue_jobs(_ssh(host_27, cmd))


def _slurm_state_by_jobid(host_27, job_id):
    if not job_id:
        return ''
    sq = _ssh(host_27, 'squeue -h -j {0} -o \"%T\" 2>/dev/null || true'.format(shlex.quote(job_id))).strip()
    if sq:
        return sq.splitlines()[0].strip().upper()
    sa = _ssh(host_27, 'sacct -n -X -j {0} --format=State 2>/dev/null || true'.format(shlex.quote(job_id)))
    for raw in sa.splitlines():
        s = raw.strip().upper()
        if not s:
            continue
        return s.split()[0]
    return ''


def _map_slurm_state_to_status(state):
    s = str(state or '').strip().upper()
    if not s:
        return ''
    if s in {'PENDING', 'CONFIGURING'}:
        return 'submitted'
    if s in {'RUNNING', 'COMPLETING', 'SUSPENDED'}:
        return 'running'
    if s in {'COMPLETED'}:
        return 'finished'
    if s in {'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL'}:
        return 'failed'
    return ''


def _slurm_job_name(job_name, system_name, frame_index_1based):
    m = re.search(r'^system_(\d+)', system_name)
    sys_id = m.group(1) if m else '0000'
    token = _job_family_token(job_name)
    return 's2s{0}_{1}_{2:06d}'.format(token, sys_id, frame_index_1based)


def _build_run_slurm(task_dir, partition, cores, walltime, job_name, env_script, node):
    node_line = '#SBATCH --nodelist={0}\n'.format(node) if node else ''
    return (
        '#!/usr/bin/env bash\n'
        '#SBATCH --partition={partition}\n'
        '#SBATCH --nodes=1\n'
        '#SBATCH --ntasks-per-node={cores}\n'
        '#SBATCH --cpus-per-task=1\n'
        '{node_line}'
        '#SBATCH --job-name={job}\n'
        '#SBATCH --output=%x.o%j\n'
        '#SBATCH --error=%x.e%j\n'
        '#SBATCH --time={walltime}\n\n'
        'set -euo pipefail\n'
        'export OMP_NUM_THREADS=1\n'
        'ulimit -s unlimited\n'
        'ENV_SCRIPT={env_script}\n'
        '# shellcheck source=/dev/null\n'
        'source \"$ENV_SCRIPT\"\n\n'
        'cd {task_dir}\n'
        'if [ -f tag_finished ]; then\n'
        '  exit 0\n'
        'fi\n'
        'rm -f tag_failed\n'
        'NP=${{SLURM_NTASKS:-{cores}}}\n'
        'set +e\n'
        'mpirun -np \"$NP\" \"$VASP_STD_BIN\"\n'
        'rc=$?\n'
        'set -e\n'
        'if [ $rc -eq 0 ]; then\n'
        '  touch tag_finished\n'
        'else\n'
        '  touch tag_failed\n'
        'fi\n'
        'exit $rc\n'
    ).format(
        partition=partition,
        cores=cores,
        node_line=node_line,
        job=job_name,
        walltime=walltime,
        env_script=shlex.quote(env_script),
        task_dir=task_dir,
    )


def _assemble_potcar_from_piece_dir(host_27, potcar_dir_27, elements, target_potcar_27):
    srcs = ['{0}/POTCAR_{1}'.format(str(potcar_dir_27).rstrip('/'), el) for el in elements]
    src_expr = ' '.join(shlex.quote(s) for s in srcs)
    cmd = (
        'set -e; '
        'for f in {srcs}; do [ -f \"$f\" ] || {{ echo \"Missing POTCAR piece: $f\" >&2; exit 2; }}; done; '
        'cat {srcs} > {target}'
    ).format(srcs=src_expr, target=shlex.quote(target_potcar_27))
    _ssh(host_27, cmd)


def _load_or_init_manifest(
    manifest_path,
    stage1_manifest_path,
    stage1_run_root,
    run_root_remote,
    run_root_remote_28,
    job_name,
    poll_seconds,
    inactivity_stop_hours,
    stride,
    backend,
    remote_host,
    host_27,
    host_28,
):
    if manifest_path.exists():
        data = read_json(manifest_path)
        if 'entries' not in data or not isinstance(data['entries'], list):
            raise ValueError('Invalid Stage-2 manifest format: {0}'.format(manifest_path))
    else:
        data = {
            'schema_version': 'nnpgen.stage2.controller.v1',
            'job_name': job_name,
            'created_at': _now(),
            'updated_at': _now(),
            'entries': [],
        }

    data['stage1_manifest_path'] = str(stage1_manifest_path)
    data['stage1_run_root_11'] = str(stage1_run_root)
    data['backend'] = str(backend)
    data['remote_host'] = str(remote_host)
    data['run_root_remote'] = str(run_root_remote)
    data['host_27'] = str(host_27 or '')
    data['host_28'] = str(host_28 or '')
    if str(backend) == 'pbs':
        data['run_root_15'] = str(run_root_remote)
    elif str(backend) == 'slurm':
        data['run_root_27'] = str(run_root_remote)
    elif str(backend) == 'slurm_dual':
        data['run_root_27'] = str(run_root_remote)
        data['run_root_28'] = str(run_root_remote_28 or run_root_remote)
    data['selection_rule'] = {
        'name': 'per-system-step',
        'frame_index_base': '1-based',
        'criterion': '(frame_index_1based - 1) % {0} == 0'.format(int(stride)),
        'stride': int(stride),
    }
    data['poll_seconds'] = int(poll_seconds)
    data['inactivity_stop_hours'] = int(inactivity_stop_hours)
    data.setdefault('last_new_eligible_at', None)
    data.setdefault('consecutive_no_new_seconds', 0)
    data.setdefault('controller_status', 'initialized')
    data['updated_at'] = _now()
    write_json(manifest_path, data)
    return data


def _entry_key(system_name, frame_name):
    return '{0}/{1}'.format(system_name, frame_name)


def _job_family_token(job_name):
    s = str(job_name or 'stage2')
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:4]


def _pbs_job_name(job_name, system_name, frame_index_1based):
    m = re.search(r'^system_(\d+)', system_name)
    sys_id = m.group(1) if m else '0000'
    token = _job_family_token(job_name)
    return 's2{0}_{1}_{2:06d}'.format(token, sys_id, frame_index_1based)


def _build_run_pbs(task_dir, queue, node, cores, walltime, job_name):
    return (
        '#!/bin/bash -l\n'
        '#PBS -q {queue}\n'
        '#PBS -N {job}\n'
        '#PBS -l nodes={node}:ppn={cores}\n'
        '#PBS -l walltime={walltime}\n\n'
        'module load vasp/vasp.5.4.4.pl2\n\n'
        'cd {task_dir}\n'
        'ulimit -s unlimited\n'
        'export OMP_NUM_THREADS=1\n\n'
        'if [ -f tag_finished ]; then\n'
        '  exit 0\n'
        'fi\n'
        'rm -f tag_failed\n'
        'mpirun -np {cores} vasp_std\n'
        'rc=$?\n'
        'if [ $rc -eq 0 ]; then\n'
        '  touch tag_finished\n'
        'else\n'
        '  touch tag_failed\n'
        'fi\n'
        'exit $rc\n'
    ).format(queue=queue, job=job_name, node=node, cores=cores, walltime=walltime, task_dir=task_dir)


def _count_statuses(entries):
    counts = {'total': 0, 'planned': 0, 'selected': 0, 'transferred': 0, 'prepared': 0, 'submitted': 0, 'running': 0, 'finished': 0, 'failed': 0}
    for e in entries:
        s = str(e.get('status', 'planned'))
        if s not in counts:
            s = 'planned'
        counts['total'] += 1
        counts[s] += 1
    return counts


def _entry_task_dir(entry):
    return str(
        entry.get('target_task_dir_remote')
        or entry.get('target_task_dir_28')
        or entry.get('target_task_dir_27')
        or entry.get('target_task_dir_15')
        or ''
    )


def _refresh_entry_status_pbs(host_15, entry):
    task_dir = _entry_task_dir(entry)
    task_info_remote = task_dir + '/task_info.json'
    if not task_dir or not _remote_exists(host_15, task_info_remote):
        return entry

    info = _remote_read_json(host_15, task_info_remote)
    job_id = str(info.get('job_id') or entry.get('job_id') or '')
    job_name = str(info.get('pbs_job_name') or entry.get('pbs_job_name') or '')

    qstate = _qstat_state_by_jobid(host_15, job_id) if job_id else ''
    mapped = _map_qstate_to_status(qstate)
    if not mapped and job_name:
        picked = _preferred_job(_jobs_by_name(host_15, job_name))
        if picked:
            mapped = _map_qstate_to_status(picked.get('state', ''))
            if picked.get('job_id'):
                job_id = picked['job_id']

    tag_finished = _remote_exists(host_15, task_dir + '/tag_finished')
    tag_failed = _remote_exists(host_15, task_dir + '/tag_failed')
    outcar_exists = _remote_exists(host_15, task_dir + '/OUTCAR')
    oszicar_exists = _remote_exists(host_15, task_dir + '/OSZICAR')

    status = str(info.get('status') or entry.get('status') or 'planned')
    if tag_finished:
        status = 'finished'
    elif tag_failed:
        status = 'failed'
    elif mapped in {'submitted', 'running', 'finished'}:
        status = mapped
    elif outcar_exists or oszicar_exists:
        status = 'running'

    info['status'] = status
    info['job_id'] = job_id if job_id else None
    info['updated_at'] = _now()
    _remote_write_json(host_15, task_info_remote, info)

    entry['status'] = status
    entry['job_id'] = info['job_id']
    entry['updated_at'] = info['updated_at']
    if info.get('resource'):
        entry['resource'] = info.get('resource')
    return entry


def _refresh_entry_status_slurm(host_27, entry):
    task_dir = _entry_task_dir(entry)
    task_info_remote = task_dir + '/task_info.json'
    if not task_dir or not _remote_exists(host_27, task_info_remote):
        return entry

    info = _remote_read_json(host_27, task_info_remote)
    job_id = str(info.get('job_id') or entry.get('job_id') or '')
    job_name = str(info.get('slurm_job_name') or entry.get('slurm_job_name') or '')

    sstate = _slurm_state_by_jobid(host_27, job_id) if job_id else ''
    mapped = _map_slurm_state_to_status(sstate)
    if not mapped and job_name:
        picked = _preferred_slurm_job(_slurm_jobs_by_name(host_27, job_name))
        if picked:
            mapped = _map_slurm_state_to_status(picked.get('state', ''))
            if picked.get('job_id'):
                job_id = picked['job_id']

    tag_finished = _remote_exists(host_27, task_dir + '/tag_finished')
    tag_failed = _remote_exists(host_27, task_dir + '/tag_failed')
    outcar_exists = _remote_exists(host_27, task_dir + '/OUTCAR')
    oszicar_exists = _remote_exists(host_27, task_dir + '/OSZICAR')

    status = str(info.get('status') or entry.get('status') or 'planned')
    if tag_finished:
        status = 'finished'
    elif tag_failed:
        status = 'failed'
    elif mapped in {'submitted', 'running', 'finished', 'failed'}:
        status = mapped
    elif outcar_exists or oszicar_exists:
        status = 'running'

    info['status'] = status
    info['job_id'] = job_id if job_id else None
    info['updated_at'] = _now()
    _remote_write_json(host_27, task_info_remote, info)

    entry['status'] = status
    entry['job_id'] = info['job_id']
    entry['updated_at'] = info['updated_at']
    if info.get('resource'):
        entry['resource'] = info.get('resource')
    return entry


def _refresh_entry_status(host_remote, entry, backend):
    if str(backend) in {'slurm', 'slurm_dual'}:
        return _refresh_entry_status_slurm(host_remote, entry)
    return _refresh_entry_status_pbs(host_remote, entry)


def _write_record_md(record_path, manifest):
    entries = sorted(manifest.get('entries', []), key=lambda x: (str(x.get('system_name', '')), int(x.get('frame_index_1based', 0))))
    counts = _count_statuses(entries)
    stride = int(manifest.get('selection_rule', {}).get('stride', 15))
    backend = str(manifest.get('backend') or 'pbs')

    lines = []
    lines.append('# Stage-2 Record: {0}'.format(manifest.get('job_name', '')))
    lines.append('')
    lines.append('- Updated at: {0}'.format(_now()))
    lines.append('- Backend: {0}'.format(backend))
    lines.append('- Remote host: {0}'.format(manifest.get('remote_host', '')))
    if backend == 'slurm_dual':
        lines.append('- Host 27 target: {0}'.format(manifest.get('host_27', '')))
        lines.append('- Host 28 target: {0}'.format(manifest.get('host_28', '')))
    lines.append('- Stage-1 manifest: {0}'.format(manifest.get('stage1_manifest_path', '')))
    lines.append('- Stage-1 run root: {0}'.format(manifest.get('stage1_run_root_11', '')))
    lines.append('- Stage-2 run root on remote: {0}'.format(manifest.get('run_root_remote', manifest.get('run_root_15', manifest.get('run_root_27', '')))))
    if backend == 'slurm_dual':
        lines.append('- Stage-2 run root host28: {0}'.format(manifest.get('run_root_28', '')))
    lines.append('- Selection convention: 1-based frame index, select if (frame_index_1based - 1) % {0} == 0.'.format(stride))
    lines.append('- Controller poll interval (s): {0}'.format(manifest.get('poll_seconds', '')))
    lines.append('- Controller inactivity auto-stop (h): {0}'.format(manifest.get('inactivity_stop_hours', '')))
    lines.append('- Last new eligible selected at: {0}'.format(manifest.get('last_new_eligible_at', 'None')))
    lines.append('')
    lines.append('Status summary: total={0} planned={1} selected={2} transferred={3} prepared={4} submitted={5} running={6} finished={7} failed={8}'.format(
        counts['total'], counts['planned'], counts['selected'], counts['transferred'], counts['prepared'], counts['submitted'], counts['running'], counts['finished'], counts['failed']
    ))
    lines.append('')
    for e in entries:
        lines.append('## {0} {1}'.format(e.get('system_name', ''), e.get('frame_name', '')))
        lines.append('- stage1_system_name: {0}'.format(e.get('system_name', '')))
        lines.append('- frame_index_1based: {0}'.format(e.get('frame_index_1based', '')))
        lines.append('- source_poscar_path_11: {0}'.format(e.get('source_poscar_path_11', '')))
        lines.append('- source_xyz_path: {0}'.format(e.get('source_xyz_path', '')))
        lines.append('- selected_by_rule: {0}'.format(e.get('selected_by_rule', False)))
        lines.append('- target_task_dir_remote: {0}'.format(e.get('target_task_dir_remote', e.get('target_task_dir_15', e.get('target_task_dir_27', e.get('target_task_dir_28', ''))))))
        if backend == 'slurm_dual':
            lines.append('- target_host: {0}'.format(e.get('remote_host', '')))
        lines.append('- status: {0}'.format(e.get('status', 'planned')))
        lines.append('- job_id: {0}'.format(e.get('job_id', '')))
        resource = e.get('resource', {})
        if resource:
            if backend in {'slurm', 'slurm_dual'}:
                lines.append('- partition/node/cores: {0}/{1}/{2}'.format(resource.get('partition', ''), resource.get('node', ''), resource.get('cores', '')))
            else:
                lines.append('- queue/node/cores: {0}/{1}/{2}'.format(resource.get('queue', ''), resource.get('node', ''), resource.get('cores', '')))
        lines.append('')

    ensure_dir(record_path.parent)
    record_path.write_text('\n'.join(lines) + '\n')


def _ensure_scheduler_notes(notes_path, backend, slurm_cfg):
    if str(backend) in {'slurm', 'slurm_dual'}:
        part_cores = slurm_cfg.get('partition_cores', {})
        part_lines = []
        for p in str(slurm_cfg.get('partitions', '')).split(','):
            s = p.strip()
            if not s:
                continue
            part_lines.append('- {0}: NCORE={1}'.format(s, part_cores.get(s, 'unset')))
        if str(backend) == 'slurm_dual':
            content = (
                '# Server27+28 Scheduler Notes\n\n'
                '- Backend: Slurm dual-host routing (odd systems -> host27, even systems -> host28).\n'
                '- Host 27 target: {0}\n'
                '- Host 28 target: {1}\n'
                '- Excluded partition: p_debug.\n'
                '- Allowed partitions: {2}\n'
                '- Partition core mapping:\n'
                '{3}\n'
                '- Consistency: Slurm ntasks-per-node and INCAR NCORE always match selected partition core count.\n'
            ).format(
                slurm_cfg.get('host_27', ''),
                slurm_cfg.get('host_28', ''),
                slurm_cfg.get('partitions', ''),
                '\n'.join(part_lines) if part_lines else '- none',
            )
        else:
            content = (
                '# Server27 Scheduler Notes\n\n'
                '- Backend: Slurm on server 27.\n'
                '- Excluded partition: p_debug.\n'
                '- Allowed partitions: {0}\n'
                '- Partition core mapping:\n'
                '{1}\n'
                '- Consistency: Slurm ntasks-per-node and INCAR NCORE always match selected partition core count.\n'
            ).format(
                slurm_cfg.get('partitions', ''),
                '\n'.join(part_lines) if part_lines else '- none',
            )
    else:
        content = (
            '# Server15 Scheduler Notes\n\n'
            '- Queue priority: normal3 > normal4 > normal2 > normal1.\n'
            '- Reserve 4 free normal1 nodes for other users.\n'
            '- Exclude nodes: node01 node02 node33 node34 node36.\n'
            '- Core mapping: normal3=24, normal4 node41/node42=28, normal4 node43=32, normal2=20, normal1=16.\n'
            '- Consistency: PBS ppn and INCAR NCORE always equal selected real cores.\n'
        )
    ensure_dir(notes_path.parent)
    notes_path.write_text(content)

def _candidate_from_manifest_entry(e):
    if str(e.get('status', '')).lower() != 'finished':
        return None
    pooled = str(e.get('pooled_poscar_path') or '').strip()
    if not pooled or not Path(pooled).exists():
        return None
    system_name = str(e.get('system_name') or '')
    frame_name = str(e.get('frame_name') or '')
    if not frame_name:
        frame_name = 'frame_{0:06d}'.format(_parse_frame_index_1based(e))
    idx1 = _parse_frame_index_1based({'frame_name': frame_name, 'source_frame_index_0based': e.get('source_frame_index_0based')})
    return {
        'task_key': _entry_key(system_name, frame_name),
        'system_name': system_name,
        'frame_name': frame_name,
        'frame_index_1based': idx1,
        'source_poscar_path_11': pooled,
        'source_xyz_path': str(e.get('source_xyz_path', '')),
        'stage1_task_name': str(e.get('task_name', '')),
    }


def _candidate_from_frame_info(path):
    try:
        d = read_json(path)
    except Exception:
        return None
    if str(d.get('status', '')).lower() != 'finished':
        return None
    system_name = str(d.get('system_name') or path.parent.parent.name)
    frame_name = str(path.parent.name)
    idx1 = _parse_frame_index_1based(frame_name)
    pooled = str(d.get('paths', {}).get('pooled_poscar', '')).strip()
    if not pooled or not Path(pooled).exists():
        return None
    source_xyz = str(d.get('source_input') or d.get('source_filename') or '')
    return {
        'task_key': _entry_key(system_name, frame_name),
        'system_name': system_name,
        'frame_name': frame_name,
        'frame_index_1based': idx1,
        'source_poscar_path_11': pooled,
        'source_xyz_path': source_xyz,
        'stage1_task_name': '{0}__{1}'.format(system_name, frame_name),
    }


def _discover_stage1_finished_candidates(stage1_manifest, stage1_run_root):
    by_key = {}

    for e in stage1_manifest.get('entries', []):
        c = _candidate_from_manifest_entry(e)
        if c is not None:
            by_key[c['task_key']] = c

    frame_infos = list(stage1_run_root.glob('system_*/frame_*/frame_info.json'))
    for p in frame_infos:
        c = _candidate_from_frame_info(p)
        if c is not None:
            by_key[c['task_key']] = c

    candidates = sorted(by_key.values(), key=lambda x: (x['system_name'], int(x['frame_index_1based'])))
    system_dirs = sorted([p.name for p in stage1_run_root.glob('system_*') if p.is_dir()])
    return candidates, system_dirs


def _prepare_and_submit_entry_pbs(host_15, entry, slot, walltime, auto_submit, retry_failed, controller_job_name):
    task_dir = _entry_task_dir(entry)
    _assert_under_15(task_dir)
    _remote_mkdir(host_15, task_dir)

    poscar_11 = Path(str(entry['source_poscar_path_11'])).resolve()
    poscar_15 = task_dir + '/POSCAR'
    if not _remote_exists(host_15, poscar_15):
        _scp_to_remote_atomic(poscar_11, host_15, poscar_15)
        entry['status'] = 'transferred'
        entry['transferred_at'] = _now()

    if entry.get('status') in {'submitted', 'running', 'finished'}:
        return _refresh_entry_status(host_15, entry, 'pbs')
    if entry.get('status') == 'failed' and not retry_failed:
        return _refresh_entry_status(host_15, entry, 'pbs')

    pinfo = _parse_poscar_local(poscar_11)
    elements = pinfo['elements']

    queue = slot['queue']
    node = slot['node']
    cores = int(slot['cores'])

    potcar_sources = [str(TEMPLATE_POTCAR_ORG_15 / ('POTCAR_' + el)) for el in elements]
    pot_src_expr = ' '.join(shlex.quote(x) for x in potcar_sources)
    potcar_15 = task_dir + '/POTCAR'
    pot_cmd = 'set -e; for f in {0}; do [ -f "$f" ] || {{ echo "Missing POTCAR source: $f" >&2; exit 1; }}; done; cat {0} > {1}'.format(
        pot_src_expr, shlex.quote(potcar_15)
    )
    _ssh(host_15, pot_cmd)

    incar_text = _ssh(host_15, 'cat {0}'.format(shlex.quote(str(TEMPLATE_INCAR_15))))
    incar_patched = _patch_incar_ncore(incar_text, cores)
    incar_15 = task_dir + '/INCAR'
    _remote_write_text(host_15, incar_15, incar_patched)

    pbs_job_name = str(entry.get('pbs_job_name') or _pbs_job_name(controller_job_name, entry['system_name'], int(entry['frame_index_1based'])))
    run_pbs_15 = task_dir + '/run.pbs'
    run_pbs_text = _build_run_pbs(task_dir, queue, node, cores, walltime, pbs_job_name)
    _remote_write_text(host_15, run_pbs_15, run_pbs_text)
    _ssh(host_15, 'chmod 755 {0}'.format(shlex.quote(run_pbs_15)))

    entry['resource'] = {'queue': queue, 'node': node, 'cores': cores}
    entry['potcar_elements_order'] = elements
    entry['status'] = 'prepared'
    entry['prepared_at'] = _now()
    entry['pbs_job_name'] = pbs_job_name

    task_info = {
        'scheduler_backend': 'pbs',
        'task_key': entry['task_key'],
        'system_name': entry['system_name'],
        'frame_name': entry['frame_name'],
        'frame_index_1based': entry['frame_index_1based'],
        'source_poscar_path_11': entry['source_poscar_path_11'],
        'source_xyz_path': entry.get('source_xyz_path', ''),
        'target_task_dir_15': task_dir,
        'target_task_dir_remote': task_dir,
        'status': entry['status'],
        'job_id': entry.get('job_id'),
        'pbs_job_name': pbs_job_name,
        'potcar_elements_order': elements,
        'resource': entry['resource'],
        'updated_at': _now(),
    }
    _remote_write_json(host_15, task_dir + '/task_info.json', task_info)

    if not auto_submit:
        return entry

    picked = _preferred_job(_jobs_by_name(host_15, pbs_job_name))
    if picked:
        st = _map_qstate_to_status(picked.get('state', ''))
        if st:
            entry['job_id'] = picked.get('job_id')
            entry['status'] = st
            entry['submitted_at'] = _now()
            task_info['job_id'] = entry['job_id']
            task_info['status'] = entry['status']
            task_info['updated_at'] = _now()
            _remote_write_json(host_15, task_dir + '/task_info.json', task_info)
            return entry

    qsub_out = _ssh(host_15, 'cd {0} && qsub run.pbs'.format(shlex.quote(task_dir))).strip()
    job_id = qsub_out.split()[0] if qsub_out else ''
    entry['job_id'] = job_id
    entry['status'] = 'submitted'
    entry['submitted_at'] = _now()

    task_info['job_id'] = job_id
    task_info['status'] = 'submitted'
    task_info['updated_at'] = _now()
    _remote_write_json(host_15, task_dir + '/task_info.json', task_info)
    return entry


def _prepare_and_submit_entry_slurm(host_remote, entry, slot, walltime, auto_submit, retry_failed, controller_job_name, slurm_cfg):
    task_dir = _entry_task_dir(entry)
    _assert_under_27(task_dir)
    _remote_mkdir(host_remote, task_dir)

    poscar_11 = Path(str(entry['source_poscar_path_11'])).resolve()
    poscar_27 = task_dir + '/POSCAR'
    if not _remote_exists(host_remote, poscar_27):
        _scp_to_remote_atomic(poscar_11, host_remote, poscar_27)
        entry['status'] = 'transferred'
        entry['transferred_at'] = _now()

    if entry.get('status') in {'submitted', 'running', 'finished'}:
        return _refresh_entry_status(host_remote, entry, 'slurm')
    if entry.get('status') == 'failed' and not retry_failed:
        return _refresh_entry_status(host_remote, entry, 'slurm')

    pinfo = _parse_poscar_local(poscar_11)
    elements = pinfo['elements']

    partition = str(slot['partition'])
    node = str(slot.get('node') or '')
    cores = int(slot['cores'])
    pin_node = node if bool(slurm_cfg.get('pin_node')) else ''
    incar_template = str(slurm_cfg.get('incar_template_by_host', {}).get(host_remote, slurm_cfg.get('incar_template_27', '')))
    potcar_source = str(slurm_cfg.get('potcar_source_by_host', {}).get(host_remote, slurm_cfg.get('potcar_source_27', '')))
    env_script = str(slurm_cfg.get('env_script_by_host', {}).get(host_remote, slurm_cfg.get('env_script_27', '')))

    potcar_27 = task_dir + '/POTCAR'
    _assemble_potcar_from_piece_dir(host_remote, potcar_source, elements, potcar_27)

    incar_text = _ssh(host_remote, 'cat {0}'.format(shlex.quote(incar_template)))
    incar_patched = _patch_incar_ncore(incar_text, cores)
    incar_27 = task_dir + '/INCAR'
    _remote_write_text(host_remote, incar_27, incar_patched)

    slurm_job_name = str(entry.get('slurm_job_name') or _slurm_job_name(controller_job_name, entry['system_name'], int(entry['frame_index_1based'])))
    run_slurm_27 = task_dir + '/run.slurm'
    run_slurm_text = _build_run_slurm(task_dir, partition, cores, walltime, slurm_job_name, env_script, pin_node)
    _remote_write_text(host_remote, run_slurm_27, run_slurm_text)
    _ssh(host_remote, 'chmod 755 {0}'.format(shlex.quote(run_slurm_27)))

    entry['resource'] = {'partition': partition, 'node': node, 'node_pinned': bool(slurm_cfg.get('pin_node')), 'cores': cores}
    entry['potcar_elements_order'] = elements
    entry['status'] = 'prepared'
    entry['prepared_at'] = _now()
    entry['slurm_job_name'] = slurm_job_name
    entry['remote_host'] = str(entry.get('remote_host') or host_remote)

    task_info = {
        'scheduler_backend': 'slurm',
        'task_key': entry['task_key'],
        'system_name': entry['system_name'],
        'frame_name': entry['frame_name'],
        'frame_index_1based': entry['frame_index_1based'],
        'source_poscar_path_11': entry['source_poscar_path_11'],
        'source_xyz_path': entry.get('source_xyz_path', ''),
        'target_task_dir_27': task_dir if str(entry.get('remote_host')) == str(slurm_cfg.get('host_27', '')) else None,
        'target_task_dir_28': task_dir if str(entry.get('remote_host')) == str(slurm_cfg.get('host_28', '')) else None,
        'target_task_dir_remote': task_dir,
        'remote_host': entry.get('remote_host'),
        'status': entry['status'],
        'job_id': entry.get('job_id'),
        'slurm_job_name': slurm_job_name,
        'potcar_elements_order': elements,
        'resource': entry['resource'],
        'updated_at': _now(),
    }
    _remote_write_json(host_remote, task_dir + '/task_info.json', task_info)

    if not auto_submit:
        return entry

    picked = _preferred_slurm_job(_slurm_jobs_by_name(host_remote, slurm_job_name))
    if picked:
        st = _map_slurm_state_to_status(picked.get('state', ''))
        if st in {'submitted', 'running'}:
            entry['job_id'] = picked.get('job_id')
            entry['status'] = st
            entry['submitted_at'] = _now()
            task_info['job_id'] = entry['job_id']
            task_info['status'] = entry['status']
            task_info['updated_at'] = _now()
            _remote_write_json(host_remote, task_dir + '/task_info.json', task_info)
            return entry

    sbatch_out = _ssh(host_remote, 'cd {0} && sbatch run.slurm'.format(shlex.quote(task_dir))).strip()
    m = re.search(r'(\d+)\s*$', sbatch_out)
    job_id = m.group(1) if m else ''
    entry['job_id'] = job_id
    entry['status'] = 'submitted'
    entry['submitted_at'] = _now()

    task_info['job_id'] = job_id
    task_info['status'] = 'submitted'
    task_info['updated_at'] = _now()
    _remote_write_json(host_remote, task_dir + '/task_info.json', task_info)
    return entry


def _prepare_and_submit_entry(host_remote, entry, slot, walltime, auto_submit, retry_failed, controller_job_name, backend, slurm_cfg):
    if str(backend) in {'slurm', 'slurm_dual'}:
        return _prepare_and_submit_entry_slurm(host_remote, entry, slot, walltime, auto_submit, retry_failed, controller_job_name, slurm_cfg)
    return _prepare_and_submit_entry_pbs(host_remote, entry, slot, walltime, auto_submit, retry_failed, controller_job_name)


def _run_one_cycle(
    manifest,
    stage1_manifest,
    stage1_run_root,
    run_root_remote,
    host_remote,
    walltime,
    auto_submit,
    retry_failed,
    stride,
    backend,
    slurm_cfg,
):
    entries = manifest.get('entries', [])

    for e in entries:
        if e.get('status') in {'submitted', 'running', 'failed'}:
            _refresh_entry_status(host_remote, e, backend)
            e['updated_at'] = _now()

    discovered, system_dirs = _discover_stage1_finished_candidates(stage1_manifest, stage1_run_root)
    eligible_all = [c for c in discovered if (int(c['frame_index_1based']) - 1) % int(stride) == 0]

    existing = {str(e.get('task_key')): e for e in entries}
    newly_selected = 0
    for c in eligible_all:
        if c['task_key'] in existing:
            continue
        target_task_dir_remote = '{0}/{1}/{2}'.format(str(run_root_remote).rstrip('/'), c['system_name'], c['frame_name'])
        if str(backend) == 'slurm':
            _assert_under_27(target_task_dir_remote)
        else:
            _assert_under_15(target_task_dir_remote)
        e = {
            'task_key': c['task_key'],
            'system_name': c['system_name'],
            'frame_name': c['frame_name'],
            'frame_index_1based': int(c['frame_index_1based']),
            'frame_index_0based': int(c['frame_index_1based']) - 1,
            'source_poscar_path_11': c['source_poscar_path_11'],
            'source_xyz_path': c.get('source_xyz_path', ''),
            'stage1_task_name': c.get('stage1_task_name', ''),
            'target_task_dir_remote': target_task_dir_remote,
            'status': 'selected',
            'selected_by_rule': True,
            'selection_rule_name': 'per-system-step',
            'job_id': None,
            'selected_at': _now(),
            'updated_at': _now(),
        }
        if str(backend) == 'slurm':
            e['target_task_dir_27'] = target_task_dir_remote
            e['slurm_job_name'] = _slurm_job_name(manifest.get('job_name', 'stage2'), c['system_name'], int(c['frame_index_1based']))
        else:
            e['target_task_dir_15'] = target_task_dir_remote
            e['pbs_job_name'] = _pbs_job_name(manifest.get('job_name', 'stage2'), c['system_name'], int(c['frame_index_1based']))
        entries.append(e)
        existing[e['task_key']] = e
        newly_selected += 1

    if auto_submit:
        if str(backend) == 'slurm':
            slots = _collect_submit_slots_slurm(host_remote, slurm_cfg.get('partitions', DEFAULT_SLURM_PARTITIONS), slurm_cfg.get('partition_cores', {}))
        else:
            slots = _collect_submit_slots(host_remote)
    else:
        slots = []

    pending = []
    for e in sorted(entries, key=lambda x: (str(x.get('system_name', '')), int(x.get('frame_index_1based', 0)))):
        s = str(e.get('status', 'planned'))
        if s in {'selected', 'planned', 'transferred', 'prepared'}:
            pending.append(e)
        elif s == 'failed' and retry_failed:
            pending.append(e)

    transferred_now = 0
    prepared_now = 0
    submitted_now = 0

    for e, slot in zip(pending, slots):
        before = str(e.get('status', 'planned'))
        _prepare_and_submit_entry(
            host_remote,
            e,
            slot,
            walltime,
            auto_submit,
            retry_failed,
            manifest.get('job_name', 'stage2'),
            backend,
            slurm_cfg,
        )
        e['updated_at'] = _now()
        after = str(e.get('status', 'planned'))
        if before != 'transferred' and after == 'transferred':
            transferred_now += 1
        if before != 'prepared' and after == 'prepared':
            prepared_now += 1
        if before != 'submitted' and after == 'submitted':
            submitted_now += 1

    for e in entries:
        if e.get('status') in {'submitted', 'running'}:
            _refresh_entry_status(host_remote, e, backend)
            e['updated_at'] = _now()

    manifest['entries'] = entries
    manifest['updated_at'] = _now()
    manifest['stage1_system_dirs_detected'] = len(system_dirs)
    if newly_selected > 0:
        manifest['last_new_eligible_at'] = _now()
        manifest['consecutive_no_new_seconds'] = 0
    else:
        manifest['consecutive_no_new_seconds'] = int(manifest.get('consecutive_no_new_seconds', 0)) + int(manifest.get('poll_seconds', 1800))

    return {
        'stage1_system_dirs_detected': len(system_dirs),
        'eligible_finished_candidates_all_systems': len(discovered),
        'eligible_finished_step_candidates': len(eligible_all),
        'newly_selected': newly_selected,
        'transferred_now': transferred_now,
        'prepared_now': prepared_now,
        'submitted_now': submitted_now,
        'slots_available': len(slots),
        'counts': _count_statuses(entries),
    }


def _run_one_cycle_slurm_dual(
    manifest,
    stage1_manifest,
    stage1_run_root,
    run_root_27,
    run_root_28,
    host_27,
    host_28,
    walltime,
    auto_submit,
    retry_failed,
    stride,
    slurm_cfg,
):
    entries = manifest.get('entries', [])

    for e in entries:
        if e.get('status') in {'submitted', 'running', 'failed'}:
            target_host = str(e.get('remote_host') or _slurm_route_host(e.get('system_name', ''), host_27, host_28))
            e['remote_host'] = target_host
            _refresh_entry_status(target_host, e, 'slurm')
            e['updated_at'] = _now()

    discovered, system_dirs = _discover_stage1_finished_candidates(stage1_manifest, stage1_run_root)
    eligible_all = [c for c in discovered if (int(c['frame_index_1based']) - 1) % int(stride) == 0]

    existing = {str(e.get('task_key')): e for e in entries}
    newly_selected = 0
    for c in eligible_all:
        if c['task_key'] in existing:
            continue
        target_host = _slurm_route_host(c['system_name'], host_27, host_28)
        run_root_remote = str(run_root_27 if target_host == host_27 else run_root_28).rstrip('/')
        target_task_dir_remote = '{0}/{1}/{2}'.format(run_root_remote, c['system_name'], c['frame_name'])
        _assert_under_27(target_task_dir_remote)
        e = {
            'task_key': c['task_key'],
            'system_name': c['system_name'],
            'frame_name': c['frame_name'],
            'frame_index_1based': int(c['frame_index_1based']),
            'frame_index_0based': int(c['frame_index_1based']) - 1,
            'source_poscar_path_11': c['source_poscar_path_11'],
            'source_xyz_path': c.get('source_xyz_path', ''),
            'stage1_task_name': c.get('stage1_task_name', ''),
            'target_task_dir_remote': target_task_dir_remote,
            'remote_host': target_host,
            'target_server_label': '27' if target_host == host_27 else '28',
            'status': 'selected',
            'selected_by_rule': True,
            'selection_rule_name': 'per-system-step',
            'job_id': None,
            'selected_at': _now(),
            'updated_at': _now(),
            'slurm_job_name': _slurm_job_name(manifest.get('job_name', 'stage2'), c['system_name'], int(c['frame_index_1based'])),
        }
        if target_host == host_27:
            e['target_task_dir_27'] = target_task_dir_remote
        else:
            e['target_task_dir_28'] = target_task_dir_remote
        entries.append(e)
        existing[e['task_key']] = e
        newly_selected += 1

    if auto_submit:
        slots_27 = _collect_submit_slots_slurm(host_27, slurm_cfg.get('partitions', DEFAULT_SLURM_PARTITIONS), slurm_cfg.get('partition_cores', {}))
        slots_28 = _collect_submit_slots_slurm(host_28, slurm_cfg.get('partitions', DEFAULT_SLURM_PARTITIONS), slurm_cfg.get('partition_cores', {}))
    else:
        slots_27 = []
        slots_28 = []

    pending_27 = []
    pending_28 = []
    for e in sorted(entries, key=lambda x: (str(x.get('system_name', '')), int(x.get('frame_index_1based', 0)))):
        s = str(e.get('status', 'planned'))
        pick = False
        if s in {'selected', 'planned', 'transferred', 'prepared'}:
            pick = True
        elif s == 'failed' and retry_failed:
            pick = True
        if not pick:
            continue
        target_host = str(e.get('remote_host') or _slurm_route_host(e.get('system_name', ''), host_27, host_28))
        e['remote_host'] = target_host
        if target_host == host_27:
            pending_27.append(e)
        else:
            pending_28.append(e)

    transferred_now = 0
    prepared_now = 0
    submitted_now = 0
    submitted_now_27 = 0
    submitted_now_28 = 0

    for e, slot in zip(pending_27, slots_27):
        before = str(e.get('status', 'planned'))
        _prepare_and_submit_entry(
            host_27,
            e,
            slot,
            walltime,
            auto_submit,
            retry_failed,
            manifest.get('job_name', 'stage2'),
            'slurm_dual',
            slurm_cfg,
        )
        e['updated_at'] = _now()
        after = str(e.get('status', 'planned'))
        if before != 'transferred' and after == 'transferred':
            transferred_now += 1
        if before != 'prepared' and after == 'prepared':
            prepared_now += 1
        if before != 'submitted' and after == 'submitted':
            submitted_now += 1
            submitted_now_27 += 1

    for e, slot in zip(pending_28, slots_28):
        before = str(e.get('status', 'planned'))
        _prepare_and_submit_entry(
            host_28,
            e,
            slot,
            walltime,
            auto_submit,
            retry_failed,
            manifest.get('job_name', 'stage2'),
            'slurm_dual',
            slurm_cfg,
        )
        e['updated_at'] = _now()
        after = str(e.get('status', 'planned'))
        if before != 'transferred' and after == 'transferred':
            transferred_now += 1
        if before != 'prepared' and after == 'prepared':
            prepared_now += 1
        if before != 'submitted' and after == 'submitted':
            submitted_now += 1
            submitted_now_28 += 1

    for e in entries:
        if e.get('status') in {'submitted', 'running'}:
            target_host = str(e.get('remote_host') or _slurm_route_host(e.get('system_name', ''), host_27, host_28))
            e['remote_host'] = target_host
            _refresh_entry_status(target_host, e, 'slurm')
            e['updated_at'] = _now()

    manifest['entries'] = entries
    manifest['updated_at'] = _now()
    manifest['stage1_system_dirs_detected'] = len(system_dirs)
    manifest['host_27'] = str(host_27)
    manifest['host_28'] = str(host_28)
    manifest['run_root_27'] = str(run_root_27)
    manifest['run_root_28'] = str(run_root_28)
    if newly_selected > 0:
        manifest['last_new_eligible_at'] = _now()
        manifest['consecutive_no_new_seconds'] = 0
    else:
        manifest['consecutive_no_new_seconds'] = int(manifest.get('consecutive_no_new_seconds', 0)) + int(manifest.get('poll_seconds', 1800))

    return {
        'stage1_system_dirs_detected': len(system_dirs),
        'eligible_finished_candidates_all_systems': len(discovered),
        'eligible_finished_step_candidates': len(eligible_all),
        'newly_selected': newly_selected,
        'transferred_now': transferred_now,
        'prepared_now': prepared_now,
        'submitted_now': submitted_now,
        'submitted_now_27': submitted_now_27,
        'submitted_now_28': submitted_now_28,
        'slots_available': len(slots_27) + len(slots_28),
        'slots_available_27': len(slots_27),
        'slots_available_28': len(slots_28),
        'pending_27': len(pending_27),
        'pending_28': len(pending_28),
        'counts': _count_statuses(entries),
    }


def _controller_active_from_pid(pid_file):
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _build_ssh_extra_args(args):
    out = []
    ssh_key = str(getattr(args, 'ssh_key', '') or '').strip()
    if ssh_key:
        key_path = abs_path(ssh_key)
        _assert_under_11(key_path)
        out.extend(['-i', str(key_path)])
    ssh_config = str(getattr(args, 'ssh_config', '') or '').strip()
    if ssh_config:
        cfg_path = abs_path(ssh_config)
        _assert_under_11(cfg_path)
        out.extend(['-F', str(cfg_path)])
    for opt in str(getattr(args, 'ssh_option', '') or '').split(','):
        s = opt.strip()
        if s:
            out.extend(['-o', s])
    return out


def add_controller_arguments(parser):
    parser.add_argument('--job-name', default=DEFAULT_JOB_NAME)
    parser.add_argument('--stage1-manifest', default=str(DEFAULT_STAGE1_MANIFEST))
    parser.add_argument('--stage1-run-root', default=str(DEFAULT_STAGE1_RUN_ROOT))
    parser.add_argument('--stage2-manifest', default=str(DEFAULT_STAGE2_MANIFEST))
    parser.add_argument('--record-path', default=str(DEFAULT_STAGE2_RECORD))
    parser.add_argument('--backend', choices=['pbs', 'slurm', 'slurm_dual'], default='pbs')
    parser.add_argument('--run-root-15', default=str(DEFAULT_RUN_ROOT_15))
    parser.add_argument('--run-root-27', default=str(DEFAULT_RUN_ROOT_27))
    parser.add_argument('--run-root-28', default=str(DEFAULT_RUN_ROOT_28))
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)
    parser.add_argument('--host-27', default=os.environ.get('NNPGEN_SLURM_HOST', DEFAULT_REMOTE_HOST))
    parser.add_argument('--host-28', default=os.environ.get('NNPGEN_SLURM_ALT_HOST', DEFAULT_REMOTE_HOST))
    parser.add_argument('--remote-host', default='')
    parser.add_argument('--poll-seconds', type=int, default=1800)
    parser.add_argument('--inactivity-stop-hours', type=int, default=24)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--walltime', default='1000:00:00')
    parser.add_argument('--slurm-partitions', default=DEFAULT_SLURM_PARTITIONS)
    parser.add_argument('--slurm-partition-cores', default=DEFAULT_SLURM_PARTITION_CORES)
    parser.add_argument('--slurm-pin-node', action='store_true')
    parser.add_argument('--incar-template-27', default=DEFAULT_TEMPLATE_INCAR_27)
    parser.add_argument('--potcar-source-27', default=DEFAULT_TEMPLATE_POTCAR_27)
    parser.add_argument('--vasp-env-script-27', default=DEFAULT_VASP_ENV_SCRIPT_27)
    parser.add_argument('--incar-template-28', default=DEFAULT_TEMPLATE_INCAR_28)
    parser.add_argument('--potcar-source-28', default=DEFAULT_TEMPLATE_POTCAR_28)
    parser.add_argument('--vasp-env-script-28', default=DEFAULT_VASP_ENV_SCRIPT_28)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--no-submit', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--ssh-key', default='')
    parser.add_argument('--ssh-config', default='')
    parser.add_argument('--ssh-option', default='BatchMode=yes,StrictHostKeyChecking=accept-new')
    parser.add_argument('--controller-state', default=str(DEFAULT_STAGE2_CONTROLLER_STATE))
    parser.add_argument('--controller-log', default=str(DEFAULT_STAGE2_CONTROLLER_LOG))
    parser.add_argument('--pid-file', default=str(DEFAULT_STAGE2_PID_FILE))
    parser.add_argument('--scheduler-notes', default=str(DEFAULT_PLAN_ROOT / 'scheduler_notes.md'))


def run_controller(args):
    stage1_manifest_path = abs_path(args.stage1_manifest)
    stage1_run_root = abs_path(args.stage1_run_root)
    stage2_manifest_path = abs_path(args.stage2_manifest)
    record_path = abs_path(args.record_path)
    controller_state_path = abs_path(args.controller_state)
    controller_log_path = abs_path(args.controller_log)
    pid_file = abs_path(args.pid_file)
    scheduler_notes = abs_path(args.scheduler_notes)

    _assert_under_11(stage1_manifest_path)
    _assert_under_11(stage1_run_root)
    _assert_under_11(stage2_manifest_path)
    _assert_under_11(record_path)
    _assert_under_11(controller_state_path)
    _assert_under_11(controller_log_path)
    _assert_under_11(pid_file)
    _assert_under_11(scheduler_notes)
    _set_ssh_extra_args(_build_ssh_extra_args(args))

    backend = str(args.backend)
    host_27 = str(args.host_27)
    host_28 = str(args.host_28)
    run_root_remote_28 = ''
    if backend == 'slurm':
        run_root_remote = str(args.run_root_27).rstrip('/')
        _assert_under_27(run_root_remote)
        host_remote = str(args.remote_host or args.host_27)
    elif backend == 'slurm_dual':
        run_root_remote = str(args.run_root_27).rstrip('/')
        run_root_remote_28 = str(args.run_root_28).rstrip('/')
        _assert_under_27(run_root_remote)
        _assert_under_27(run_root_remote_28)
        host_remote = '{0},{1}'.format(host_27, host_28)
    else:
        run_root_remote = str(args.run_root_15).rstrip('/')
        _assert_under_15(run_root_remote)
        host_remote = str(args.remote_host or args.host_15)

    slurm_cfg = {
        'partitions': str(args.slurm_partitions),
        'partition_cores': _parse_partition_cores(args.slurm_partition_cores),
        'pin_node': bool(args.slurm_pin_node),
        'host_27': host_27,
        'host_28': host_28,
        'incar_template_27': str(args.incar_template_27),
        'potcar_source_27': str(args.potcar_source_27),
        'env_script_27': str(args.vasp_env_script_27),
        'incar_template_by_host': {
            host_27: str(args.incar_template_27),
            host_28: str(args.incar_template_28),
        },
        'potcar_source_by_host': {
            host_27: str(args.potcar_source_27),
            host_28: str(args.potcar_source_28),
        },
        'env_script_by_host': {
            host_27: str(args.vasp_env_script_27),
            host_28: str(args.vasp_env_script_28),
        },
    }

    ensure_dir(controller_state_path.parent)
    ensure_dir(controller_log_path.parent)
    ensure_dir(pid_file.parent)

    if _controller_active_from_pid(pid_file):
        existing_pid = pid_file.read_text().strip()
        print('controller_already_running pid={0} pid_file={1}'.format(existing_pid, pid_file))
        return {'already_running': True}

    pid_file.write_text(str(os.getpid()) + '\n')
    _ensure_scheduler_notes(scheduler_notes, backend, slurm_cfg)

    stage1_manifest = read_json(stage1_manifest_path)
    manifest = _load_or_init_manifest(
        stage2_manifest_path,
        stage1_manifest_path,
        stage1_run_root,
        run_root_remote,
        run_root_remote_28,
        args.job_name,
        int(args.poll_seconds),
        int(args.inactivity_stop_hours),
        int(args.stride),
        backend,
        host_remote,
        host_27,
        host_28,
    )

    inactivity_stop_seconds = int(args.inactivity_stop_hours) * 3600
    auto_submit = not bool(args.no_submit)
    cycle = 0
    stop_reason = ''
    last_result = {}

    try:
        while True:
            cycle += 1
            stage1_manifest = read_json(stage1_manifest_path)
            if backend == 'slurm_dual':
                result = _run_one_cycle_slurm_dual(
                    manifest,
                    stage1_manifest,
                    stage1_run_root,
                    run_root_remote,
                    run_root_remote_28,
                    host_27,
                    host_28,
                    args.walltime,
                    auto_submit,
                    bool(args.retry_failed),
                    int(args.stride),
                    slurm_cfg,
                )
            else:
                result = _run_one_cycle(
                    manifest,
                    stage1_manifest,
                    stage1_run_root,
                    run_root_remote,
                    host_remote,
                    args.walltime,
                    auto_submit,
                    bool(args.retry_failed),
                    int(args.stride),
                    backend,
                    slurm_cfg,
                )
            last_result = result
            counts = result['counts']
            if backend == 'slurm_dual':
                line = 'stage2_cycle={0} job_name={1} backend={2} host27={3} host28={4} stride={5} stage1_system_dirs={6} eligible_finished_all={7} eligible_step={8} new_selected={9} slots_total={10} slots_27={11} slots_28={12} pending_27={13} pending_28={14} transferred_now={15} prepared_now={16} submitted_now={17} submitted_now_27={18} submitted_now_28={19} total={20} selected={21} transferred={22} prepared={23} submitted={24} running={25} finished={26} failed={27}'.format(
                    cycle,
                    args.job_name,
                    backend,
                    host_27,
                    host_28,
                    int(args.stride),
                    result['stage1_system_dirs_detected'],
                    result['eligible_finished_candidates_all_systems'],
                    result['eligible_finished_step_candidates'],
                    result['newly_selected'],
                    result['slots_available'],
                    result.get('slots_available_27', 0),
                    result.get('slots_available_28', 0),
                    result.get('pending_27', 0),
                    result.get('pending_28', 0),
                    result['transferred_now'],
                    result['prepared_now'],
                    result['submitted_now'],
                    result.get('submitted_now_27', 0),
                    result.get('submitted_now_28', 0),
                    counts['total'],
                    counts['selected'],
                    counts['transferred'],
                    counts['prepared'],
                    counts['submitted'],
                    counts['running'],
                    counts['finished'],
                    counts['failed'],
                )
            else:
                line = 'stage2_cycle={0} job_name={1} backend={2} remote_host={3} stride={4} stage1_system_dirs={5} eligible_finished_all={6} eligible_step={7} new_selected={8} slots_available={9} transferred_now={10} prepared_now={11} submitted_now={12} total={13} selected={14} transferred={15} prepared={16} submitted={17} running={18} finished={19} failed={20}'.format(
                    cycle,
                    args.job_name,
                    backend,
                    host_remote,
                    int(args.stride),
                    result['stage1_system_dirs_detected'],
                    result['eligible_finished_candidates_all_systems'],
                    result['eligible_finished_step_candidates'],
                    result['newly_selected'],
                    result['slots_available'],
                    result['transferred_now'],
                    result['prepared_now'],
                    result['submitted_now'],
                    counts['total'],
                    counts['selected'],
                    counts['transferred'],
                    counts['prepared'],
                    counts['submitted'],
                    counts['running'],
                    counts['finished'],
                    counts['failed'],
                )
            print(line)
            with controller_log_path.open('a') as lf:
                lf.write('{0} {1}\n'.format(_now(), line))

            manifest['controller_status'] = 'active' if args.continuous else 'once_completed'
            manifest['updated_at'] = _now()
            write_json(stage2_manifest_path, manifest)
            _write_record_md(record_path, manifest)

            state = {
                'job_name': args.job_name,
                'pid': os.getpid(),
                'active': bool(args.continuous),
                'cycle': cycle,
                'poll_seconds': int(args.poll_seconds),
                'inactivity_stop_hours': int(args.inactivity_stop_hours),
                'stride': int(args.stride),
                'manifest_path': str(stage2_manifest_path),
                'record_path': str(record_path),
                'backend': backend,
                'remote_host': host_remote,
                'run_root_remote': run_root_remote,
                'host_27': host_27,
                'host_28': host_28,
                'run_root_28': run_root_remote_28,
                'last_cycle_at': _now(),
                'last_new_eligible_at': manifest.get('last_new_eligible_at'),
                'consecutive_no_new_seconds': int(manifest.get('consecutive_no_new_seconds', 0)),
                'stage1_system_dirs_detected': int(result['stage1_system_dirs_detected']),
                'eligible_finished_all': int(result['eligible_finished_candidates_all_systems']),
                'eligible_finished_step': int(result['eligible_finished_step_candidates']),
                'slots_available': int(result['slots_available']),
                'slots_available_27': int(result.get('slots_available_27', 0)),
                'slots_available_28': int(result.get('slots_available_28', 0)),
                'status_counts': counts,
                'stop_reason': stop_reason,
            }
            write_json(controller_state_path, state)

            if not args.continuous:
                break

            if int(manifest.get('consecutive_no_new_seconds', 0)) >= inactivity_stop_seconds:
                stop_reason = 'auto_stopped_no_new_eligible_for_{0}_hours'.format(args.inactivity_stop_hours)
                state['active'] = False
                state['stop_reason'] = stop_reason
                write_json(controller_state_path, state)
                manifest['controller_status'] = 'auto_stopped'
                manifest['updated_at'] = _now()
                write_json(stage2_manifest_path, manifest)
                _write_record_md(record_path, manifest)
                print('stage2_controller_stop reason={0}'.format(stop_reason))
                with controller_log_path.open('a') as lf:
                    lf.write('{0} stage2_controller_stop reason={1}\n'.format(_now(), stop_reason))
                break

            time.sleep(int(args.poll_seconds))
    finally:
        if pid_file.exists():
            try:
                cur = int(pid_file.read_text().strip())
            except Exception:
                cur = -1
            if cur == os.getpid():
                pid_file.unlink()

    return last_result

def add_controller_status_arguments(parser):
    parser.add_argument('--stage1-run-root', default=str(DEFAULT_STAGE1_RUN_ROOT))
    parser.add_argument('--stage2-manifest', default=str(DEFAULT_STAGE2_MANIFEST))
    parser.add_argument('--controller-state', default=str(DEFAULT_STAGE2_CONTROLLER_STATE))
    parser.add_argument('--pid-file', default=str(DEFAULT_STAGE2_PID_FILE))
    parser.add_argument('--backend', default='')
    parser.add_argument('--host-15', default=DEFAULT_REMOTE_HOST)
    parser.add_argument('--host-27', default=os.environ.get('NNPGEN_SLURM_HOST', DEFAULT_REMOTE_HOST))
    parser.add_argument('--host-28', default=os.environ.get('NNPGEN_SLURM_ALT_HOST', DEFAULT_REMOTE_HOST))
    parser.add_argument('--remote-host', default='')
    parser.add_argument('--ssh-key', default='')
    parser.add_argument('--ssh-config', default='')
    parser.add_argument('--ssh-option', default='BatchMode=yes,StrictHostKeyChecking=accept-new')
    parser.add_argument('--refresh', action='store_true')


def run_controller_status(args):
    stage1_run_root = abs_path(args.stage1_run_root)
    manifest_path = abs_path(args.stage2_manifest)
    state_path = abs_path(args.controller_state)
    pid_file = abs_path(args.pid_file)
    _assert_under_11(stage1_run_root)
    _assert_under_11(manifest_path)
    _assert_under_11(state_path)
    _assert_under_11(pid_file)
    _set_ssh_extra_args(_build_ssh_extra_args(args))

    manifest = read_json(manifest_path)
    backend = str(args.backend or manifest.get('backend') or 'pbs')
    host_27 = str(manifest.get('host_27') or args.host_27)
    host_28 = str(manifest.get('host_28') or args.host_28)
    host_remote = str(args.remote_host or manifest.get('remote_host') or (args.host_27 if backend in {'slurm', 'slurm_dual'} else args.host_15))
    if args.refresh:
        for e in manifest.get('entries', []):
            if backend == 'slurm_dual':
                target_host = str(e.get('remote_host') or _slurm_route_host(e.get('system_name', ''), host_27, host_28))
                e['remote_host'] = target_host
                _refresh_entry_status(target_host, e, 'slurm')
            else:
                _refresh_entry_status(host_remote, e, backend)
            e['updated_at'] = _now()
        manifest['updated_at'] = _now()
        write_json(manifest_path, manifest)

    counts = _count_statuses(manifest.get('entries', []))
    stride = int(manifest.get('selection_rule', {}).get('stride', 15))

    stage1_manifest_path = Path(str(manifest.get('stage1_manifest_path')))
    s1 = {'entries': []}
    if stage1_manifest_path.exists():
        s1 = read_json(stage1_manifest_path)

    discovered, system_dirs = _discover_stage1_finished_candidates(s1, stage1_run_root)
    eligible_step = [c for c in discovered if (int(c['frame_index_1based']) - 1) % stride == 0]

    active = _controller_active_from_pid(pid_file)
    no_new_seconds = int(manifest.get('consecutive_no_new_seconds', 0))
    if backend == 'slurm_dual':
        host_counts = {
            'host27_total': 0,
            'host27_running': 0,
            'host27_submitted': 0,
            'host28_total': 0,
            'host28_running': 0,
            'host28_submitted': 0,
        }
        for e in manifest.get('entries', []):
            target_host = str(e.get('remote_host') or _slurm_route_host(e.get('system_name', ''), host_27, host_28))
            status = str(e.get('status') or '')
            if target_host == host_27:
                host_counts['host27_total'] += 1
                if status == 'running':
                    host_counts['host27_running'] += 1
                if status == 'submitted':
                    host_counts['host27_submitted'] += 1
            else:
                host_counts['host28_total'] += 1
                if status == 'running':
                    host_counts['host28_running'] += 1
                if status == 'submitted':
                    host_counts['host28_submitted'] += 1
        print('job_name={0} backend={1} host27={2} host28={3} stride={4} stage1_system_dirs={5} eligible_finished_all={6} eligible_finished_step={7} total={8} planned={9} selected={10} transferred={11} prepared={12} submitted={13} running={14} finished={15} failed={16} host27_total={17} host27_submitted={18} host27_running={19} host28_total={20} host28_submitted={21} host28_running={22} no_new_seconds={23} controller_active={24} controller_status={25}'.format(
            manifest.get('job_name', ''),
            backend,
            host_27,
            host_28,
            stride,
            len(system_dirs),
            len(discovered),
            len(eligible_step),
            counts['total'],
            counts['planned'],
            counts['selected'],
            counts['transferred'],
            counts['prepared'],
            counts['submitted'],
            counts['running'],
            counts['finished'],
            counts['failed'],
            host_counts['host27_total'],
            host_counts['host27_submitted'],
            host_counts['host27_running'],
            host_counts['host28_total'],
            host_counts['host28_submitted'],
            host_counts['host28_running'],
            no_new_seconds,
            active,
            manifest.get('controller_status', ''),
        ))
    else:
        print('job_name={0} backend={1} remote_host={2} stride={3} stage1_system_dirs={4} eligible_finished_all={5} eligible_finished_step={6} total={7} planned={8} selected={9} transferred={10} prepared={11} submitted={12} running={13} finished={14} failed={15} no_new_seconds={16} controller_active={17} controller_status={18}'.format(
            manifest.get('job_name', ''),
            backend,
            host_remote,
            stride,
            len(system_dirs),
            len(discovered),
            len(eligible_step),
            counts['total'],
            counts['planned'],
            counts['selected'],
            counts['transferred'],
            counts['prepared'],
            counts['submitted'],
            counts['running'],
            counts['finished'],
            counts['failed'],
            no_new_seconds,
            active,
            manifest.get('controller_status', ''),
        ))
    if manifest.get('last_new_eligible_at'):
        print('last_new_eligible_at={0}'.format(manifest.get('last_new_eligible_at')))
    if state_path.exists():
        state = read_json(state_path)
        if state.get('stop_reason'):
            print('controller_stop_reason={0}'.format(state.get('stop_reason')))

    return {
        'counts': counts,
        'backend': backend,
        'remote_host': host_remote,
        'host_27': host_27,
        'host_28': host_28,
        'stage1_system_dirs': len(system_dirs),
        'eligible_finished_all': len(discovered),
        'eligible_finished_step': len(eligible_step),
        'controller_active': active,
    }


def build_parser():
    parser = argparse.ArgumentParser(description='DFT controller')
    sub = parser.add_subparsers(dest='command')
    p_controller = sub.add_parser('controller', help='Run one cycle or continuous periodic controller')
    add_controller_arguments(p_controller)
    p_status = sub.add_parser('status', help='Show DFT summary')
    add_controller_status_arguments(p_status)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    if args.command == 'controller':
        run_controller(args)
    elif args.command == 'status':
        run_controller_status(args)
    else:
        raise ValueError('Unsupported command: {0}'.format(args.command))


if __name__ == '__main__':
    main()
