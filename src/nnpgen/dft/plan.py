import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..config import (
    DEFAULT_PLAN_ROOT,
    DEFAULT_POSCAR_POOL_ROOT,
    PROJECT_ROOT,
    VASP15_INCAR_TEMPLATE,
    VASP15_PBS_TEMPLATE,
    VASP15_POTCAR_CAT_DIR,
    VASP15_POTCAR_ORG_DIR,
    VASP15_ROOT,
    VASP_PLAN_STATUSES,
)
from ..manifest import resolve_records
from .pbs_template import build_ncore_patch_plan, infer_requested_ncore, parse_pbs_template
from ..utils import abs_path, ensure_dir, slugify, write_json


SUPPORTED_POTCAR_ELEMENTS = {'H', 'O', 'N', 'Na', 'Cl', 'Ti', 'C', 'Si'}

REFERENCE_PBS_FALLBACK = {
    'queue': 'normal3',
    'nodes_expr': 'node39:ppn=24',
    'node_name': 'node39',
    'ppn': 24,
    'mpirun_np': 24,
    'walltime': '1000:00:00',
}
REFERENCE_INCAR_NCORE_FALLBACK = 24


def _now() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def _assert_under_project(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    p = path.resolve()
    if p != root and root not in p.parents:
        raise ValueError('Path outside project root: {0}'.format(p))


def parse_poscar_element_order(poscar_path: Path) -> List[str]:
    lines = Path(poscar_path).resolve().read_text().splitlines()
    if len(lines) < 6:
        raise ValueError('Invalid POSCAR, fewer than 6 lines: {0}'.format(poscar_path))

    elems = lines[5].split()
    if not elems:
        raise ValueError('Invalid POSCAR element line: {0}'.format(poscar_path))

    for e in elems:
        if e not in SUPPORTED_POTCAR_ELEMENTS:
            raise ValueError('Unsupported element for POTCAR mapping: {0} ({1})'.format(e, poscar_path))

    return elems


def build_potcar_plan(elements_order: List[str], potcar_org_dir_15: Path, potcar_cat_dir_15: Path) -> Dict:
    source_files = []
    for elem in elements_order:
        src = Path(potcar_org_dir_15) / ('POTCAR_{0}'.format(elem))
        source_files.append(str(src))

    cat_name = 'POTCAR_{0}'.format('_'.join(elements_order))
    cat_path = Path(potcar_cat_dir_15) / cat_name

    return {
        'elements_order': elements_order,
        'source_files_15': source_files,
        'cat_target_dir_15': str(Path(potcar_cat_dir_15)),
        'cat_target_name': cat_name,
        'cat_target_path_15': str(cat_path),
        'source_check': 'deferred_to_server_15',
    }


def _task_name(system_name: str, frame_name: str) -> str:
    base = '{0}__{1}'.format(system_name, frame_name)
    safe = slugify(base)
    return safe[:120]


def _frame_index_from_name(frame_name: str) -> int:
    m = re.match(r'^frame_(\d{6})$', frame_name)
    if not m:
        return -1
    return int(m.group(1))


def _apply_pbs_fallback(pbs_info: Dict) -> Dict:
    out = dict(pbs_info)
    for key, value in REFERENCE_PBS_FALLBACK.items():
        if out.get(key) in (None, ''):
            out[key] = value
    return out


def _build_task_entry(
    record: Dict,
    run_root_11: Path,
    target_base_15: Path,
    incar_template_15: Path,
    pbs_template_15: Path,
    potcar_org_dir_15: Path,
    potcar_cat_dir_15: Path,
    pbs_info: Dict,
    ncore_plan: Dict,
) -> Dict:
    poscar_path = Path(record.get('poscar_path', '')).resolve()
    elements_order = parse_poscar_element_order(poscar_path)
    potcar_plan = build_potcar_plan(elements_order, potcar_org_dir_15, potcar_cat_dir_15)

    system_name = record.get('system_name', 'system_unknown')
    frame_name = record.get('frame_name', 'frame_unknown')
    task_name = _task_name(system_name, frame_name)

    return {
        'task_name': task_name,
        'status': 'planned',
        'system_name': system_name,
        'frame_name': frame_name,
        'frame_index_1based': _frame_index_from_name(frame_name),
        'source_xyz_path': record.get('source_xyz_path', ''),
        'source_frame_index_0based': record.get('source_frame_index_0based'),
        'source_frame_index_1based': record.get('source_frame_index_1based'),
        'poscar_path_11': str(poscar_path),
        'target_work_dir_15': str(Path(target_base_15) / task_name),
        'potcar': potcar_plan,
        'incar': {
            'template_path_15': str(Path(incar_template_15)),
            'template_ncore': ncore_plan.get('template_ncore'),
            'requested_ncore': ncore_plan.get('requested_ncore'),
            'will_patch_ncore': ncore_plan.get('will_patch'),
        },
        'pbs': {
            'template_path_15': str(Path(pbs_template_15)),
            'queue': pbs_info.get('queue', ''),
            'nodes_expr': pbs_info.get('nodes_expr', ''),
            'node_name': pbs_info.get('node_name', ''),
            'ppn': pbs_info.get('ppn'),
            'mpirun_np': pbs_info.get('mpirun_np'),
            'walltime': pbs_info.get('walltime', ''),
        },
        'provenance': {
            'run_root_11': str(run_root_11),
            'record_json_path_11': record.get('record_json_path', ''),
            'source_status': record.get('status', ''),
        },
    }


def add_vasp_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--run-root', required=True, help='11-side run/<job_name> root path')
    parser.add_argument('--job-name', default='', help='Plan name, default: run-root name')
    parser.add_argument('--target-mode', choices=['smoke', 'run'], default='smoke')
    parser.add_argument('--output', default='', help='Output manifest path on 11')
    parser.add_argument('--max-tasks', type=int, default=2, help='Optional cap for dry-run planning')

    parser.add_argument('--poscar-pool-root', default=str(DEFAULT_POSCAR_POOL_ROOT))
    parser.add_argument('--target-root-15', default=str(VASP15_ROOT))
    parser.add_argument('--incar-template-15', default=str(VASP15_INCAR_TEMPLATE))
    parser.add_argument('--pbs-template-15', default=str(VASP15_PBS_TEMPLATE))
    parser.add_argument('--potcar-org-dir-15', default=str(VASP15_POTCAR_ORG_DIR))
    parser.add_argument('--potcar-cat-dir-15', default=str(VASP15_POTCAR_CAT_DIR))
    parser.add_argument('--requested-ncore', type=int, default=None)


def run_vasp_plan(args: argparse.Namespace) -> Path:
    run_root = abs_path(args.run_root)
    _assert_under_project(run_root)

    job_name = args.job_name.strip() if args.job_name else run_root.name
    if not job_name:
        raise ValueError('Empty job name')

    records = resolve_records(
        job_name=job_name,
        run_root=run_root,
        poscar_pool_root=abs_path(args.poscar_pool_root),
    )

    valid_records = [r for r in records if r.get('poscar_path') and Path(r.get('poscar_path')).exists()]

    if args.max_tasks is not None and args.max_tasks > 0:
        valid_records = valid_records[: args.max_tasks]

    target_root_15 = Path(args.target_root_15)
    target_base_15 = target_root_15 / args.target_mode / job_name

    incar_template_15 = Path(args.incar_template_15)
    pbs_template_15 = Path(args.pbs_template_15)
    potcar_org_dir_15 = Path(args.potcar_org_dir_15)
    potcar_cat_dir_15 = Path(args.potcar_cat_dir_15)

    pbs_info = _apply_pbs_fallback(parse_pbs_template(pbs_template_15))

    requested_ncore = args.requested_ncore
    if requested_ncore is None:
        requested_ncore = infer_requested_ncore(pbs_info, default_ncore=REFERENCE_INCAR_NCORE_FALLBACK)

    ncore_plan = build_ncore_patch_plan(incar_template_15, requested_ncore)
    if ncore_plan.get('template_ncore') is None:
        ncore_plan['template_ncore'] = REFERENCE_INCAR_NCORE_FALLBACK
        ncore_plan['will_patch'] = bool(ncore_plan['requested_ncore'] != ncore_plan['template_ncore'])

    tasks = []
    errors = []
    for rec in valid_records:
        try:
            task = _build_task_entry(
                record=rec,
                run_root_11=run_root,
                target_base_15=target_base_15,
                incar_template_15=incar_template_15,
                pbs_template_15=pbs_template_15,
                potcar_org_dir_15=potcar_org_dir_15,
                potcar_cat_dir_15=potcar_cat_dir_15,
                pbs_info=pbs_info,
                ncore_plan=ncore_plan,
            )
            tasks.append(task)
        except Exception as exc:
            errors.append({'record_json_path': rec.get('record_json_path', ''), 'error': str(exc)})

    out_path = abs_path(args.output) if args.output else (DEFAULT_PLAN_ROOT / '{0}_vasp_plan.json'.format(job_name)).resolve()
    _assert_under_project(out_path)
    ensure_dir(out_path.parent)

    manifest = {
        'schema_version': 'nnpgen.vasp_plan.v1',
        'created_at': _now(),
        'job_name': job_name,
        'run_root_11': str(run_root),
        'target_mode': args.target_mode,
        'target_base_15': str(target_base_15),
        'status_model': VASP_PLAN_STATUSES,
        'templates_15': {
            'incar_template_path_15': str(incar_template_15),
            'pbs_template_path_15': str(pbs_template_15),
            'potcar_org_dir_15': str(potcar_org_dir_15),
            'potcar_cat_dir_15': str(potcar_cat_dir_15),
        },
        'ncore_plan': ncore_plan,
        'pbs_template_resources': pbs_info,
        'records_scanned': len(records),
        'records_with_poscar': len(valid_records),
        'task_count': len(tasks),
        'errors': errors,
        'tasks': tasks,
    }

    write_json(out_path, manifest)

    print('VASP plan written: {0}'.format(out_path))
    print('records_scanned={0} records_with_poscar={1} task_count={2} errors={3}'.format(
        len(records), len(valid_records), len(tasks), len(errors)
    ))
    return out_path
