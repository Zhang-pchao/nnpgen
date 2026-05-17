import re
from pathlib import Path
from typing import Dict, Optional

from ..utils import read_text


def parse_pbs_template(pbs_template_path: Path) -> Dict:
    p = Path(pbs_template_path).resolve()
    info = {
        'template_path': str(p),
        'template_exists': p.exists(),
        'queue': '',
        'job_name': '',
        'nodes_expr': '',
        'walltime': '',
        'node_name': '',
        'ppn': None,
        'mpirun_np': None,
    }

    if not p.exists():
        return info

    text = read_text(p)
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('#PBS -q '):
            info['queue'] = s.split(None, 2)[-1]
        elif s.startswith('#PBS -N '):
            info['job_name'] = s.split(None, 2)[-1]
        elif s.startswith('#PBS -l nodes='):
            nodes_expr = s.split('nodes=', 1)[1].strip()
            info['nodes_expr'] = nodes_expr
            m_node = re.match(r'([^:]+)', nodes_expr)
            if m_node:
                info['node_name'] = m_node.group(1)
            m_ppn = re.search(r'ppn=(\d+)', nodes_expr)
            if m_ppn:
                info['ppn'] = int(m_ppn.group(1))
        elif s.startswith('#PBS -l walltime='):
            info['walltime'] = s.split('walltime=', 1)[1].strip()

        m_np = re.search(r'mpirun\s+-np\s+(\d+)', s)
        if m_np:
            info['mpirun_np'] = int(m_np.group(1))

    return info


def parse_incar_ncore(incar_template_path: Path) -> Optional[int]:
    p = Path(incar_template_path).resolve()
    if not p.exists():
        return None

    text = read_text(p)
    for line in text.splitlines():
        m = re.match(r'^\s*NCORE\s*=\s*(\d+)', line)
        if m:
            return int(m.group(1))
    return None


def build_ncore_patch_plan(incar_template_path: Path, requested_ncore: Optional[int]) -> Dict:
    template_ncore = parse_incar_ncore(incar_template_path)

    if requested_ncore is None:
        requested_ncore = template_ncore

    return {
        'template_ncore': template_ncore,
        'requested_ncore': requested_ncore,
        'will_patch': bool(template_ncore is not None and requested_ncore is not None and template_ncore != requested_ncore),
    }


def infer_requested_ncore(pbs_info: Dict, default_ncore: int = 24) -> int:
    for key in ('ppn', 'mpirun_np'):
        value = pbs_info.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return default_ncore
