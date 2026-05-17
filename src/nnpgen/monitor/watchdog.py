#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, deque
import re
from pathlib import Path

from ..config import DEFAULT_CONDA_ENV, PROJECT_ROOT, RUN_ROOT

DEFAULT_CONFIG = RUN_ROOT / 'controller_watchdog_config.json'
DEFAULT_LOG = RUN_ROOT / 'controller_restart_log.txt'
DEFAULT_PID = RUN_ROOT / 'controller_watchdog.pid'
CONDA_SH = os.environ.get('NNPGEN_CONDA_SH', '')
CONDA_ENV = DEFAULT_CONDA_ENV


def utc_now():
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def run_shell(cmd):
    return subprocess.run(['bash', '-lc', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    with p.open('r') as f:
        return json.load(f)


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(str(tmp), str(p))


def append_log(log_path, msg):
    line = f'{utc_now()} {msg}'
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        f.write(line + '\n')
    print(line)


def list_processes():
    out = run_shell('ps -eo pid=,args=').stdout.splitlines()
    procs = []
    for line in out:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        pid = int(parts[0])
        args = parts[1] if len(parts) > 1 else ''
        procs.append((pid, args))
    return procs


def count_manifest(manifest_path):
    d = read_json(manifest_path)
    entries = d.get('entries', []) if isinstance(d, dict) else []
    c = Counter(str(e.get('status', 'planned')) for e in entries)
    total = len(entries)
    finished = int(c.get('finished', 0))
    return {
        'total': total,
        'finished': finished,
        'unfinished': max(total - finished, 0),
        'counts': dict(c),
        'source': 'manifest',
    }


def _stage1_run_root(ctrl):
    run_root = str(ctrl.get('run_root', '')).strip()
    if run_root:
        return run_root
    start_cmd = str(ctrl.get('start_cmd', '')).strip()
    m = re.search(r'--run-root\s+([^\s]+)', start_cmd)
    return m.group(1) if m else ''


def _map_stage1_status(raw_status):
    s = str(raw_status or "").strip().lower()
    if s in {"prepared", "planned"}:
        return "planned"
    if s in {"submitted", "running", "finished", "failed"}:
        return s
    return "planned"


def _collect_stage1_status_paths(run_root):
    rr = Path(str(run_root)).resolve()
    if not rr.exists() or not rr.is_dir():
        return []
    frame_info = sorted(rr.glob("system_*/frame_*/frame_info.json"))
    frame_records = sorted(rr.glob("system_*/frame_records/*.json"))
    return frame_info + frame_records


def count_stage1_live(run_root):
    if not run_root:
        return None

    status_paths = _collect_stage1_status_paths(run_root)
    if not status_paths:
        return None

    counts = Counter()
    for p in status_paths:
        info = read_json(p)
        status = _map_stage1_status(info.get("status", "planned"))
        counts[status] += 1

    total = sum(counts.values())
    finished = int(counts.get("finished", 0))
    return {
        "total": total,
        "finished": finished,
        "unfinished": max(total - finished, 0),
        "counts": dict(counts),
        "source": "stage1_live_status",
    }


def _stage2_stage1_run_root(ctrl):
    run_root = str(ctrl.get("stage1_run_root", "")).strip()
    if run_root:
        return run_root
    start_cmd = str(ctrl.get("start_cmd", "")).strip()
    m = re.search(r"--stage1-run-root\s+([^\s]+)", start_cmd)
    return m.group(1) if m else ""


def count_stage1_log(run_root):
    if not run_root:
        return None
    log_path = Path(run_root) / 'stage1_submit_controller.log'
    if not log_path.exists():
        return None

    recent = deque(maxlen=600)
    with log_path.open('r', errors='replace') as f:
        for line in f:
            recent.append(line.rstrip('\n'))

    pat = re.compile(
        r'\btotal=(?P<total>\d+)\b.*\bsubmitted=(?P<submitted>\d+)\b.*'
        r'\brunning=(?P<running>\d+)\b.*\bfinished=(?P<finished>\d+)\b.*'
        r'\bfailed=(?P<failed>\d+)\b'
    )
    for line in reversed(recent):
        m = pat.search(line)
        if not m:
            continue
        total = int(m.group('total'))
        submitted = int(m.group('submitted'))
        running = int(m.group('running'))
        finished = int(m.group('finished'))
        failed = int(m.group('failed'))
        planned = max(total - submitted - running - finished - failed, 0)
        return {
            'total': total,
            'finished': finished,
            'unfinished': max(total - finished, 0),
            'counts': {
                'planned': planned,
                'submitted': submitted,
                'running': running,
                'finished': finished,
                'failed': failed,
            },
            'source': 'stage1_log',
            'progress_log': str(log_path),
        }
    return None


def stage1_complete(ctrl):
    stats = count_stage1_live(_stage1_run_root(ctrl))
    if not stats:
        stats = count_manifest(ctrl["manifest_path"])
    complete = stats["total"] > 0 and stats["unfinished"] == 0
    return complete, stats


def stage2_complete(ctrl):
    s1 = count_stage1_live(_stage2_stage1_run_root(ctrl))
    if not s1:
        s1 = count_manifest(ctrl["stage1_manifest_path"])
    s2 = count_manifest(ctrl["manifest_path"])
    s1_complete = s1["total"] > 0 and s1["unfinished"] == 0
    complete = s1_complete and s2["unfinished"] == 0
    return complete, s1, s2


def controller_pids(ctrl, processes):
    pids = set()
    pid_file = str(ctrl.get('pid_file', '')).strip()
    if pid_file:
        p = Path(pid_file)
        if p.exists():
            try:
                pid = int(p.read_text().strip())
            except Exception:
                pid = None
            if pid_alive(pid):
                pids.add(pid)
    match = str(ctrl.get('match_substring', '')).strip()
    if match:
        for pid, args in processes:
            if match in args:
                pids.add(pid)
    return sorted(pids)


def start_controller(ctrl):
    start_cmd = str(ctrl['start_cmd']).strip()
    nohup_log = str(ctrl.get('nohup_log', DEFAULT_LOG)).strip()
    setup_parts = ["module load conda >/dev/null 2>&1 || true"]
    if CONDA_SH:
        setup_parts.append(f"source {CONDA_SH} >/dev/null 2>&1 || true")
    if CONDA_ENV:
        setup_parts.append(f"conda activate {CONDA_ENV} >/dev/null 2>&1")
    setup_parts.append(f"cd {PROJECT_ROOT}")
    setup_parts.append(f"nohup {start_cmd} >{nohup_log} 2>&1 & echo $!")
    wrapper = "; ".join(setup_parts)
    r = run_shell(wrapper)
    pid = None
    if r.returncode == 0:
        out = (r.stdout or '').strip().splitlines()
        if out:
            last = out[-1].strip()
            if last.isdigit():
                pid = int(last)
    return r.returncode == 0 and pid_alive(pid), pid, r


def stop_controller_pids(pids):
    stopped = []
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
            stopped.append(int(pid))
        except Exception:
            pass
    return stopped


def _inactive_reason(ctrl):
    pid_file = str(ctrl.get('pid_file', '')).strip()
    if pid_file:
        pp = Path(pid_file)
        if not pp.exists():
            return 'controller PID file missing'
        raw = pp.read_text().strip()
        if not raw:
            return 'controller PID file empty'
        try:
            pid = int(raw)
        except Exception:
            return 'controller PID file invalid'
        if not pid_alive(pid):
            return 'controller PID not found'
    return 'controller process not found by match'


def _restart_action(ctrl):
    ok, new_pid, result = start_controller(ctrl)
    if ok and new_pid:
        return 'restarted pid={0}'.format(new_pid), [new_pid]
    err = ((result.stderr or '') + ' ' + (result.stdout or '')).strip().replace('\n', ' ')
    if len(err) > 220:
        err = err[:220] + '...'
    if not err:
        err = 'unknown_error'
    return 'restart_failed reason={0}'.format(err), []


def run_check(config, log_path):
    processes = list_processes()
    auto_restart = bool(config.get('auto_restart', True))
    stop_when_all_complete = bool(config.get('stop_controllers_when_all_complete', True))

    all_complete = True
    active_map = {}

    for ctrl in config.get('stage1_controllers', []):
        complete, stats = stage1_complete(ctrl)
        pids = controller_pids(ctrl, processes)
        should_run = not complete
        active_map[ctrl['name']] = pids
        all_complete = all_complete and complete

        action = 'running_ok' if pids else 'none'
        reason = ''
        if complete:
            action = 'completed_skip_restart'
        elif not pids:
            reason = _inactive_reason(ctrl)
            if auto_restart:
                action, restarted_pids = _restart_action(ctrl)
                if restarted_pids:
                    pids = restarted_pids
            else:
                action = 'inactive_auto_restart_disabled'

        append_log(
            log_path,
            (
                f"stage1 name={ctrl['name']} complete={complete} total={stats['total']} "
                f"finished={stats['finished']} unfinished={stats['unfinished']} source={stats.get('source', 'manifest')} "
                f"pids={pids} action={action}"
                + (f" reason={reason}" if reason else '')
            ),
        )

    for ctrl in config.get('stage2_controllers', []):
        complete, s1, s2 = stage2_complete(ctrl)
        pids = controller_pids(ctrl, processes)
        should_run = not complete
        active_map[ctrl['name']] = pids
        all_complete = all_complete and complete

        action = 'running_ok' if pids else 'none'
        reason = ''
        if complete:
            action = 'completed_skip_restart'
        elif not pids:
            reason = _inactive_reason(ctrl)
            if auto_restart:
                action, restarted_pids = _restart_action(ctrl)
                if restarted_pids:
                    pids = restarted_pids
            else:
                action = 'inactive_auto_restart_disabled'

        append_log(
            log_path,
            (
                f"stage2 name={ctrl['name']} complete={complete} "
                f"stage1_total={s1['total']} stage1_finished={s1['finished']} "
                f"stage2_total={s2['total']} stage2_finished={s2['finished']} stage2_unfinished={s2['unfinished']} "
                f"pids={pids} action={action}"
                + (f" reason={reason}" if reason else '')
            ),
        )

    if all_complete and auto_restart:
        config['auto_restart'] = False
        write_json(config['config_path'], config)
        append_log(log_path, 'all_controllers_complete auto_restart=false written_to_config')

        if stop_when_all_complete:
            all_pids = sorted({pid for pids in active_map.values() for pid in pids})
            stopped = stop_controller_pids(all_pids)
            append_log(log_path, f'all_controllers_complete stopped_pids={stopped}')

    return all_complete


def load_config(path):
    cfg = read_json(path)
    if not isinstance(cfg, dict):
        raise RuntimeError(f'invalid config: {path}')
    cfg['config_path'] = str(path)
    cfg.setdefault('auto_restart', True)
    cfg.setdefault('interval_minutes', 180)
    cfg.setdefault('stop_controllers_when_all_complete', True)
    cfg.setdefault('exit_when_all_complete', True)
    cfg.setdefault('watchdog_pid_file', str(DEFAULT_PID))
    cfg.setdefault('log_path', str(DEFAULT_LOG))
    cfg.setdefault('stage1_controllers', [])
    cfg.setdefault('stage2_controllers', [])
    return cfg


def acquire_lock(pid_file, log_path):
    p = Path(pid_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            old = int(p.read_text().strip())
        except Exception:
            old = None
        if pid_alive(old):
            append_log(log_path, f'watchdog_already_running pid={old} pid_file={pid_file}')
            return False
    p.write_text(str(os.getpid()))
    return True


def release_lock(pid_file):
    p = Path(pid_file)
    if p.exists():
        try:
            cur = int(p.read_text().strip())
        except Exception:
            cur = None
        if cur == os.getpid():
            p.unlink()


def main():
    ap = argparse.ArgumentParser(description='Stage1/Stage2 controller watchdog')
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval-minutes', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.interval_minutes is not None:
        cfg['interval_minutes'] = int(args.interval_minutes)

    log_path = str(cfg['log_path'])
    pid_file = str(cfg['watchdog_pid_file'])

    if args.once:
        append_log(log_path, f"watchdog_start once={args.once} interval_minutes={cfg['interval_minutes']} auto_restart={cfg['auto_restart']}")
        run_check(cfg, log_path)
        return 0

    if not acquire_lock(pid_file, log_path):
        return 0

    try:
        append_log(log_path, f"watchdog_start once={args.once} interval_minutes={cfg['interval_minutes']} auto_restart={cfg['auto_restart']}")
        while True:
            all_complete = run_check(cfg, log_path)
            if all_complete and bool(cfg.get('exit_when_all_complete', True)):
                append_log(log_path, 'watchdog_exit all_controllers_complete')
                break
            time.sleep(max(60, int(cfg['interval_minutes']) * 60))
    finally:
        release_lock(pid_file)

    return 0


if __name__ == '__main__':
    sys.exit(main())
