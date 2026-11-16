"""Análise a posteriori (f): Hill Climbing nas classes sem limite de orçamento.

Não faz parte do protocolo congelado antes da execução definitiva. Repete as 150 buscas do cenário de
classes (10 rodadas × 3 classificadores × 5 sementes) com o código de run_experiment.py, sem alterações.
A única diferença é o orçamento: cada trajetória só termina em um ótimo local, isto é, quando nenhum
vizinho supera a solução atual por mais de 1e-12. Partições, sementes, inicializações, classificadores,
critério de escolha e avaliação externa são os do experimento principal.

O que é executado e o que é reportado estão registrados em REGISTRO_ANALISE_F.md.

Uso: python run_convergence.py [--out results/convergence] [--workers 2]
A execução pode ser interrompida e retomada (um JSON por tarefa e um cache por avaliação).
"""
import argparse, json, os, platform, time
from multiprocessing import Pool
from pathlib import Path

import run_experiment as RE

UNLIMITED = 10**9      # limite inatingível de candidatos novos por trajetória


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(RE.ROOT / 'results' / 'convergence'))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--model', choices=RE.MODELS)
    a = ap.parse_args()
    out = Path(a.out); (out / 'tasks').mkdir(parents=True, exist_ok=True)
    import numpy as np, pandas as pd, sklearn, scipy
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               sklearn=sklearn.__version__, scipy=scipy.__version__, os=platform.platform(),
               cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?'),
               logical_cpus=os.cpu_count(), workers=a.workers, thread_limit=1,
               budget_per_trajectory='sem limite (parada apenas em ótimo local)', seeds=RE.SEEDS, models=RE.MODELS)
    (out / 'environment.json').write_text(json.dumps(env, indent=2, ensure_ascii=False))
    # modelos lineares primeiro (rápidos); a floresta aleatória concentra o custo
    models = [m for m in ['LR', 'SVM', 'RF'] if a.model in (None, m)]
    tasks = [('classes', project, name, seed, str(out), UNLIMITED)
             for name in models for seed in RE.SEEDS for project in RE.VERSIONS]
    todo = [t for t in tasks if not (out / 'tasks' / f'{t[0]}__{t[1]}__{t[2]}__{t[3]}.json').exists()]
    print(f'{len(tasks)} tarefas; {len(tasks) - len(todo)} já concluídas; {len(todo)} a executar', flush=True)
    t0 = time.perf_counter(); done = 0
    with Pool(a.workers, initializer=RE.init_worker) as pool:
        for fname, status, secs in pool.imap_unordered(RE.run_task, todo):
            done += 1
            print(f'[{done}/{len(todo)}] {fname} {status} {secs:.0f}s '
                  f'(decorrido {(time.perf_counter() - t0) / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
