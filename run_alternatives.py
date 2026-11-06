"""Experimento complementar: outras formas de seleção de atributos sob o mesmo protocolo.

Para cada (cenário, rodada, classificador, semente) usados no experimento principal:
  filter : ranking univariado pelo teste F (ANOVA), ajustado no treinamento de cada divisão;
           o número k de atributos (1..p) é escolhido pelo MCC médio nas mesmas três validações internas.
  l1     : seleção embutida por regressão logística com penalização L1 (liblinear, pesos balanceados),
           ajustada no treinamento de cada divisão; a força C é escolhida pelo MCC médio nas validações internas.
Em ambos os casos, o classificador da combinação é treinado apenas com os atributos selecionados.
Após a escolha do hiperparâmetro de seleção, a seleção e o classificador são reajustados no desenvolvimento
completo e avaliados uma única vez no teste externo. Linha de base e partições são as do experimento principal.
"""
import argparse, json, os, time, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

import run_experiment as R

K_GRID = None                       # filtro: 1..p
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

def f_rank(X, y):
    """Ordem dos atributos pelo teste F, calculado após imputação pela mediana do próprio treinamento."""
    from sklearn.feature_selection import f_classif
    Xi = X.fillna(X.median())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F, _ = f_classif(Xi, y)
    F = np.nan_to_num(F, nan=-np.inf)
    return [c for _, _, c in sorted(zip(-F, range(len(X.columns)), X.columns))]

def l1_select(X, y, C, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                          LogisticRegression(penalty='l1', solver='liblinear', C=C, class_weight='balanced',
                                             max_iter=2000, random_state=seed)).fit(X, y)
    coef = m[-1].coef_.ravel()
    return [c for c, w in zip(X.columns, coef) if abs(w) > 1e-10]

def run_task(task):
    scenario, fold, name, seed, out_dir = task
    out = Path(out_dir) / 'tasks' / f'{scenario}__{fold}__{name}__{seed}.json'
    if out.exists():
        return out.name, 'skip', 0.0
    t0 = time.perf_counter()
    dev, test, cols, folds, group_cols = R.split(scenario, fold)
    from sklearn.metrics import matthews_corrcoef
    cdir = Path(out_dir) / 'eval_cache'; cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / out.with_suffix('.jsonl').name
    disk = {}
    if cfile.exists():
        for line in cfile.read_text().splitlines():
            if line.strip():
                r = json.loads(line); disk[r['key']] = r
    warn_log = []
    # seleções por divisão interna (dependem só do treinamento da divisão)
    ranks = [f_rank(dev.iloc[tr][cols], dev.target.iloc[tr]) for tr, va in folds]

    def evaluate(key, subsets):
        if key in disk:
            return disk[key]
        vals, sizes = [], []; t1 = time.perf_counter()
        for (tr, va), sub in zip(folds, subsets):
            if not sub:
                vals = None; break
            pred, _ = R.fit_predict(name, seed, dev.iloc[tr][sub], dev.target.iloc[tr], dev.iloc[va][sub], warn_log)
            vals.append(float(matthews_corrcoef(dev.target.iloc[va], pred))); sizes.append(len(sub))
        rec = dict(key=key, mcc=(float(np.mean(vals)) if vals else None), folds=vals, sizes=sizes,
                   seconds=time.perf_counter() - t1)
        with open(cfile, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        disk[key] = rec
        return rec

    result = dict(scenario=scenario, fold=fold, model=name, seed=seed, n_available=len(cols))
    # ---------------- filtro
    cand = []
    for k in range(1, len(cols) + 1):
        # mantém a ordem original das colunas (a ordem altera os sorteios internos da floresta)
        r = evaluate(f'filter_k{k}', [[c for c in cols if c in rk[:k]] for rk in ranks]); cand.append((k, r))
    best_k = min(cand, key=lambda kr: (-kr[1]['mcc'], kr[0]))[0]
    top = f_rank(dev[cols], dev.target)[:best_k]
    chosen_filter = [c for c in cols if c in top]
    # ---------------- L1
    cand_l1 = []
    for C in C_GRID:
        subs = [l1_select(dev.iloc[tr][cols], dev.target.iloc[tr], C, seed) for tr, va in folds]
        r = evaluate(f'l1_C{C}', subs)
        if r['mcc'] is not None:
            cand_l1.append((C, r))
    order = sorted(cand_l1, key=lambda cr: (-cr[1]['mcc'], np.mean(cr[1]['sizes']), cr[0]))
    chosen_l1, best_C = [], None
    for C, r in order:                       # primeira C válida cuja seleção no desenvolvimento não é vazia
        chosen_l1 = l1_select(dev[cols], dev.target, C, seed)
        if chosen_l1:
            best_C = C; break

    for method, chosen, param, cands in [('filter', chosen_filter, best_k, cand), ('l1', chosen_l1, best_C, cand_l1)]:
        t1 = time.perf_counter()
        pred, score = R.fit_predict(name, seed, dev[chosen], dev.target, test[chosen], warn_log)
        m = R.metrics(test.target, pred, score); m['fit_predict_seconds'] = time.perf_counter() - t1
        m['n_features'] = len(chosen)
        if group_cols:
            m['by_group'] = {g: {k: R.metrics(test.target[idx], pred[idx], score[idx]) for k, idx in test.groupby(g).indices.items()}
                             for g in group_cols}
        result[method] = dict(param=param, selected=chosen, validation_mcc=max(r['mcc'] for _, r in cands),
                              candidates=[dict(param=p, mcc=r['mcc'], sizes=r['sizes'], seconds=r['seconds']) for p, r in cands],
                              search_seconds=sum(r['seconds'] for _, r in cands), test=m)
    result['warnings'] = sorted(set(warn_log)); result['total_seconds'] = time.perf_counter() - t0
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(result, ensure_ascii=False)); os.replace(tmp, out)
    return out.name, 'done', result['total_seconds']

def init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    R.DATA['classes'] = R.load_classes(); R.DATA['commits'] = R.load_commits()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(R.ROOT / 'results' / 'alternatives'))
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument('--scenario'); ap.add_argument('--model'); ap.add_argument('--seed', type=int); ap.add_argument('--fold')
    a = ap.parse_args()
    out = Path(a.out); (out / 'tasks').mkdir(parents=True, exist_ok=True)
    only = lambda t: ((a.scenario is None or t[0] == a.scenario) and (a.model is None or t[2] == a.model)
                      and (a.seed is None or t[3] == a.seed) and (a.fold is None or t[1] == a.fold))
    tasks = [t[:4] + (str(out),) for t in R.all_tasks(out, 0, only)]
    todo = [t for t in tasks if not (out / 'tasks' / f'{t[0]}__{t[1]}__{t[2]}__{t[3]}.json').exists()]
    print(f'{len(tasks)} tarefas; {len(tasks) - len(todo)} já concluídas; {len(todo)} a executar', flush=True)
    t0 = time.perf_counter(); done = 0
    with Pool(a.workers, initializer=init_worker) as pool:
        for fname, status, secs in pool.imap_unordered(run_task, todo):
            done += 1
            print(f'[{done}/{len(todo)}] {fname} {status} {secs:.0f}s (decorrido {(time.perf_counter()-t0)/60:.1f} min)', flush=True)

if __name__ == '__main__':
    main()
