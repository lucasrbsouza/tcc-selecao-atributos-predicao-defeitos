"""Compara duas execuções do experimento, tarefa por tarefa.

Uso: python compare_tasks.py <execução_a> <execução_b>

Cada execução é uma pasta com tasks/ e predictions/ (por exemplo, results/nohadoop/ e a pasta de uma
reexecução). Para cada tarefa presente nas duas pastas, compara o JSON da tarefa, ignorando apenas os campos de
tempo (nomes terminados em "seconds"), e os arrays das previsões gravadas. Termina com código 1 se houver
alguma diferença.
"""
import json
import sys
from pathlib import Path

import numpy as np


def sem_tempos(o):
    if isinstance(o, dict):
        return {k: sem_tempos(v) for k, v in o.items() if not k.endswith('seconds')}
    if isinstance(o, list):
        return [sem_tempos(x) for x in o]
    return o


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    comuns = sorted({f.name for f in (a / 'tasks').glob('*.json')} & {f.name for f in (b / 'tasks').glob('*.json')})
    if not comuns:
        sys.exit('nenhuma tarefa em comum')
    diferencas = 0
    for nome in comuns:
        ja = sem_tempos(json.loads((a / 'tasks' / nome).read_text()))
        jb = sem_tempos(json.loads((b / 'tasks' / nome).read_text()))
        campos = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
        pa, pb = (d / 'predictions' / nome.replace('.json', '.npz') for d in (a, b))
        if pa.exists() and pb.exists():
            with np.load(pa) as xa, np.load(pb) as xb:
                if set(xa.files) != set(xb.files) or any(not np.array_equal(xa[k], xb[k]) for k in xa.files):
                    campos.append('predictions')
        else:
            campos.append('predictions (ausente)')
        diferencas += bool(campos)
        print(f'{nome}: ' + ('idêntica (exceto tempos)' if not campos else 'difere em ' + ', '.join(campos)))
    print(f'{len(comuns)} tarefas comparadas; {diferencas} com diferença')
    sys.exit(1 if diferencas else 0)


if __name__ == '__main__':
    main()
