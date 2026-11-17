"""Análise a posteriori (g): cenário de commits sem o identificador apache/hadoop.

Não faz parte do protocolo congelado e não foi registrada antes da execução (ver ANALISE_G.md). O identificador
apache/hadoop não tem nenhum commit rotulado como indutor de defeito na base. Esta análise retira os seus registros
do desenvolvimento e do teste e refaz o cenário de commits com o código de run_experiment.py, sem alterações:
três classificadores, cinco sementes, as mesmas janelas internas por ano, o mesmo orçamento de 100 candidatos novos
por trajetória e a avaliação única no teste externo. A única diferença é o filtro aplicado aos dados carregados.

Uso: python run_nohadoop.py [--out results/nohadoop] [--workers 2] [--model LR] [--seed 101]
A execução pode ser interrompida e retomada (um JSON por tarefa e um cache por avaliação).
O filtro é aplicado no processo principal e herdado pelos processos de trabalho (início por fork).
"""
import argparse, json, multiprocessing, os, platform, time
from pathlib import Path

import run_experiment as RE

BUDGET = 100                     # candidatos novos por trajetória, como no experimento principal
TEST_RECORDS = 24690             # 30.111 registros de teste menos os 5.421 de apache/hadoop
_load_commits = RE.load_commits


def load_commits_nohadoop():
    dev, test = _load_commits()  # mesmas verificações de integridade do experimento principal
    dev = dev[dev.project != 'apache/hadoop'].reset_index(drop=True)
    test = test[test.project != 'apache/hadoop'].reset_index(drop=True)
    assert len(test) == TEST_RECORDS, len(test)
    return dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(RE.ROOT / 'results' / 'nohadoop'))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--model', choices=RE.MODELS)
    ap.add_argument('--seed', type=int, choices=RE.SEEDS)
    a = ap.parse_args()
    out = Path(a.out); (out / 'tasks').mkdir(parents=True, exist_ok=True)
    RE.load_commits = load_commits_nohadoop
    import numpy as np, pandas as pd, sklearn, scipy
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               sklearn=sklearn.__version__, scipy=scipy.__version__, os=platform.platform(),
               cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?'),
               logical_cpus=os.cpu_count(), workers=a.workers, thread_limit=1, budget_per_trajectory=BUDGET,
               data_filter='project != apache/hadoop (desenvolvimento e teste)', seeds=RE.SEEDS, models=RE.MODELS)
    (out / 'environment.json').write_text(json.dumps(env, indent=2, ensure_ascii=False))
    # a floresta aleatória concentra o custo e vai primeiro
    tasks = [('commits', 'temporal', name, seed, str(out), BUDGET)
             for name in ['RF', 'LR', 'SVM'] if a.model in (None, name)
             for seed in RE.SEEDS if a.seed in (None, seed)]
    todo = [t for t in tasks if not (out / 'tasks' / f'{t[0]}__{t[1]}__{t[2]}__{t[3]}.json').exists()]
    print(f'{len(tasks)} tarefas; {len(tasks) - len(todo)} já concluídas; {len(todo)} a executar', flush=True)
    t0 = time.perf_counter(); done = 0
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(a.workers, initializer=RE.init_worker) as pool:
        for fname, status, secs in pool.imap_unordered(RE.run_task, todo):
            done += 1
            print(f'[{done}/{len(todo)}] {fname} {status} {secs:.0f}s '
                  f'(decorrido {(time.perf_counter() - t0) / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
