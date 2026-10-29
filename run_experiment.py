"""Experimento final do TCC (protocolo da versão parcial v02).

Cenários
  classes : 10 rodadas leave-one-project-out; validação interna GroupKFold(3) por projeto.
  commits : desenvolvimento = year <= 2016; validações internas 2014, 2015, 2016;
            teste externo = teste amplo publicado (year 2017-2019).
Tratamentos
  full : todos os atributos elegíveis.
  hc   : subconjunto escolhido por Hill Climbing (MCC médio nas 3 validações internas).

O teste externo só é usado DEPOIS da escolha do subconjunto, para os dois tratamentos.
Cada tarefa (cenário, rodada, modelo, semente) grava um JSON próprio; o script pode
ser interrompido e retomado sem perder o que já terminou.
"""
import argparse, json, os, sys, time, warnings, platform
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CF = 'wmc dit noc cbo rfc lcom ca ce npm lcom3 loc dam moa mfa cam ic cbm amc max_cc avg_cc'.split()
JF = 'la ld nf nd ns ent ndev age nuc aexp arexp asexp'.split()
VERSIONS = {'ant': '1.7', 'camel': '1.6', 'ivy': '2.0', 'jedit': '4.3', 'log4j': '1.2',
            'lucene': '2.4', 'synapse': '1.2', 'velocity': '1.6', 'xalan': '2.7', 'xerces': '1.4'}
SEEDS = [101, 202, 303, 404, 505]
MODELS = ['LR', 'RF', 'SVM']
FAMILY = {'apache/hadoop': 'hadoop', 'apache/hadoop-hdfs': 'hadoop', 'apache/hadoop-mapreduce': 'hadoop'}
EPS = 1e-12

# ---------------------------------------------------------------- dados
def load_classes():
    parts = []
    for project, version in VERSIONS.items():
        d = pd.read_csv(ROOT / f'data/classes/{project}-{version}.csv')
        assert list(d.columns[3:-1]) == CF, project
        d['group'] = project
        parts.append(d)
    c = pd.concat(parts, ignore_index=True)
    c['target'] = (c.bug > 0).astype(int)
    assert len(c) == 5081 and c.target.sum() == 2296
    return c

def load_commits():
    j = pd.read_csv(ROOT / 'data/commits/apachejit_total.csv')
    j['target'] = j.buggy.astype(int)
    j['project_id'] = j.project.str.replace('apache/', '', regex=False)
    j['family'] = j.project.map(FAMILY).fillna(j.project_id)
    assert len(j) == 106674 and j.target.sum() == 28239
    dev = j[j.year <= 2016].reset_index(drop=True)
    test = j[j.year >= 2017].reset_index(drop=True)
    pub = pd.read_csv(ROOT / 'data/commits/apachejit_test_large.csv', usecols=['commit_id'])
    assert set(test.commit_id) == set(pub.commit_id)
    assert not set(dev.commit_id) & set(test.commit_id)
    return dev, test

def split(scenario, fold):
    """Devolve (dev, test, colunas, folds internos, coluna de grupo para métricas)."""
    if scenario == 'classes':
        from sklearn.model_selection import GroupKFold
        c = DATA['classes']
        dev = c[c.group != fold].reset_index(drop=True)
        test = c[c.group == fold].reset_index(drop=True)
        folds = list(GroupKFold(n_splits=3).split(dev[CF], dev.target, dev.group))
        for tr, va in folds:
            assert not set(dev.group.iloc[tr]) & set(dev.group.iloc[va])
            assert fold not in set(dev.group)
        return dev, test, CF, folds, None
    dev, test = DATA['commits']
    folds = []
    for end, year in [(2013, 2014), (2014, 2015), (2015, 2016)]:
        tr = np.flatnonzero((dev.year <= end).to_numpy())
        va = np.flatnonzero((dev.year == year).to_numpy())
        folds.append((tr, va))
    assert dev.year.max() <= 2016 < test.year.min()
    return dev, test, JF, folds, ['project_id', 'family']

# ---------------------------------------------------------------- modelos
def make_model(name, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    if name == 'LR':
        m = LogisticRegression(C=1, solver='lbfgs', max_iter=2000, class_weight='balanced', random_state=seed)
    elif name == 'RF':
        m = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=None,
                                   class_weight='balanced', n_jobs=1, random_state=seed)
    else:
        m = LinearSVC(C=1, dual=False, max_iter=10000, penalty='l2', loss='squared_hinge',
                      class_weight='balanced', random_state=seed)
    steps = [SimpleImputer(strategy='median', keep_empty_features=True)]
    if name != 'RF':
        steps.append(StandardScaler())
    return make_pipeline(*steps, m)

def fit_predict(name, seed, Xtr, ytr, Xte, warn_log):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        m = make_model(name, seed).fit(Xtr, ytr)
        if name == 'SVM':
            score = m.decision_function(Xte); pred = (score > 0).astype(int)
        else:
            score = m.predict_proba(Xte)[:, 1]; pred = (score > 0.5).astype(int)
    for w in caught:
        warn_log.append(f'{w.category.__name__}: {w.message}')
    return pred, score

def metrics(y, pred, score):
    from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
    y = np.asarray(y); pred = np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(n=int(len(y)), positives=int(y.sum()), prevalence=float(y.mean()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                mcc=float(matthews_corrcoef(y, pred)),
                f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                auc=float(roc_auc_score(y, score)) if both else None,
                degenerate=bool((not both) or len(np.unique(pred)) < 2))

# ---------------------------------------------------------------- Hill Climbing
def hill_climb(evaluate, p, seed, per_traj):
    rng = np.random.default_rng(seed)
    starts = [tuple([1] * p)]
    for _ in range(2):
        m = (rng.random(p) < 0.5).astype(int)
        if not m.any():
            m[rng.integers(p)] = 1
        starts.append(tuple(int(b) for b in m))
    cache, trace, trajectories = {}, [], []
    for t, current in enumerate(starts):
        used = 0
        def score(mask):
            nonlocal used
            if mask not in cache:
                cache[mask] = evaluate(mask); used += 1
                trace.append(dict(trajectory=t, mask=''.join(map(str, mask)), n_features=sum(mask), mcc_validation=cache[mask]))
            return cache[mask]
        score(current); steps = 0; reason = 'local_optimum'
        while True:
            neigh = [tuple(1 - b if i == k else b for i, b in enumerate(current)) for k in range(p)]
            neigh = [x for x in neigh if any(x)]
            unknown = sum(x not in cache for x in neigh)
            if used + unknown > per_traj:
                reason = 'budget'; break
            for x in neigh:
                score(x)
            best = min(neigh, key=lambda x: (-cache[x], sum(x), x))
            if cache[best] > cache[current] + EPS:
                current = best; steps += 1
            else:
                break
        trajectories.append(dict(trajectory=t, start=''.join(map(str, starts[t])), end=''.join(map(str, current)),
                                 end_mcc=cache[current], steps=steps, new_evaluations=used, stop=reason))
    best = min(cache, key=lambda x: (-cache[x], sum(x), x))
    return best, cache[best], trace, trajectories

# ---------------------------------------------------------------- tarefa
DATA = {}
def init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    DATA['classes'] = load_classes()
    DATA['commits'] = load_commits()

def run_task(task):
    scenario, fold, name, seed, out_dir, per_traj = task
    out = Path(out_dir) / 'tasks' / f'{scenario}__{fold}__{name}__{seed}.json'
    if out.exists():
        return out.name, 'skip', 0.0
    t0 = time.perf_counter()
    dev, test, cols, folds, group_cols = split(scenario, fold)
    warn_log = []
    # Cache persistente: cada avaliação de candidato é gravada assim que termina.
    # Como a busca é determinística, uma execução interrompida é retomada
    # reaproveitando os valores gravados (mesma trajetória, sem novos ajustes).
    cdir = Path(out_dir) / 'eval_cache'; cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / out.with_suffix('.jsonl').name
    disk = {}
    if cfile.exists():
        for line in cfile.read_text().splitlines():
            if line.strip():
                r = json.loads(line); disk[r['mask']] = r
    from sklearn.metrics import matthews_corrcoef

    def evaluate(mask):
        key = ''.join(map(str, mask))
        if key in disk:
            return disk[key]['mcc']
        chosen = [c for c, b in zip(cols, mask) if b]
        vals = []; t1 = time.perf_counter(); n_warn = len(warn_log)
        for tr, va in folds:
            pred, _ = fit_predict(name, seed, dev.iloc[tr][chosen], dev.target.iloc[tr], dev.iloc[va][chosen], warn_log)
            vals.append(float(matthews_corrcoef(dev.target.iloc[va], pred)))
        rec = dict(mask=key, mcc=float(np.mean(vals)), folds=vals, seconds=time.perf_counter() - t1,
                   warnings=sorted(set(warn_log[n_warn:])))
        with open(cfile, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        disk[key] = rec
        return rec['mcc']

    best, best_val, trace, trajectories = hill_climb(evaluate, len(cols), seed, per_traj)
    t_search = sum(disk[r['mask']]['seconds'] for r in trace)
    for r in trace:
        r['folds'] = disk[r['mask']]['folds']; r['seconds'] = disk[r['mask']]['seconds']
        warn_log.extend(disk[r['mask']]['warnings'])
    full = tuple([1] * len(cols))

    result = dict(scenario=scenario, fold=fold, model=name, seed=seed, per_trajectory_budget=per_traj,
                  n_available=len(cols), selected=[c for c, b in zip(cols, best) if b],
                  selected_mask=''.join(map(str, best)), validation_mcc_selected=best_val,
                  # o conjunto completo é sempre o 1º início, logo está no rastro
                  validation_mcc_full=next(r['mcc_validation'] for r in trace if r['mask'] == '1' * len(cols)),
                  unique_evaluations=len(trace), search_seconds=t_search,
                  trajectories=trajectories, trace=trace)

    preds = {}
    for treat, mask in [('full', full), ('hc', best)]:
        chosen = [c for c, b in zip(cols, mask) if b]
        t1 = time.perf_counter()
        pred, score = fit_predict(name, seed, dev[chosen], dev.target, test[chosen], warn_log)
        secs = time.perf_counter() - t1
        m = metrics(test.target, pred, score); m['fit_predict_seconds'] = secs; m['n_features'] = len(chosen)
        if group_cols:
            m['by_group'] = {}
            for g in group_cols:
                m['by_group'][g] = {k: metrics(test.target[idx], pred[idx], score[idx])
                                    for k, idx in test.groupby(g).indices.items()}
        result[treat] = m
        preds[treat] = (pred, score)
    result['warnings'] = sorted(set(warn_log))
    result['n_warnings'] = len(warn_log)
    result['total_seconds'] = time.perf_counter() - t0

    pdir = Path(out_dir) / 'predictions'; pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / out.with_suffix('.npz').name, y=test.target.to_numpy(),
                        full_pred=preds['full'][0], full_score=preds['full'][1],
                        hc_pred=preds['hc'][0], hc_score=preds['hc'][1])
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False))
    os.replace(tmp, out)
    return out.name, 'done', result['total_seconds']

# ---------------------------------------------------------------- principal
def all_tasks(out_dir, per_traj, only=None):
    tasks = []
    for name in MODELS:
        for seed in SEEDS:
            tasks.append(('commits', 'temporal', name, seed))
            for project in VERSIONS:
                tasks.append(('classes', project, name, seed))
    # Mais caras primeiro (RF em commits) para equilibrar os processos.
    cost = {('commits', 'RF'): 0, ('classes', 'RF'): 1, ('commits', 'LR'): 2, ('commits', 'SVM'): 2}
    tasks.sort(key=lambda t: (cost.get((t[0], t[2]), 3), t))
    if only:
        tasks = [t for t in tasks if only(t)]
    return [t + (str(out_dir), per_traj) for t in tasks]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results'))
    ap.add_argument('--budget', type=int, default=100, help='candidatos novos por trajetória')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument('--scenario', choices=['classes', 'commits'])
    ap.add_argument('--model', choices=MODELS)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--fold')
    a = ap.parse_args()
    out = Path(a.out); (out / 'tasks').mkdir(parents=True, exist_ok=True)
    import sklearn, scipy
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               sklearn=sklearn.__version__, scipy=scipy.__version__, os=platform.platform(),
               cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?'),
               logical_cpus=os.cpu_count(), workers=a.workers, thread_limit=1,
               budget_per_trajectory=a.budget, seeds=SEEDS, models=MODELS)
    (out / 'environment.json').write_text(json.dumps(env, indent=2))
    only = lambda t: ((a.scenario is None or t[0] == a.scenario) and (a.model is None or t[2] == a.model)
                      and (a.seed is None or t[3] == a.seed) and (a.fold is None or t[1] == a.fold))
    tasks = all_tasks(out, a.budget, only)
    todo = [t for t in tasks if not (out / 'tasks' / f'{t[0]}__{t[1]}__{t[2]}__{t[3]}.json').exists()]
    print(f'{len(tasks)} tarefas; {len(tasks) - len(todo)} já concluídas; {len(todo)} a executar', flush=True)
    t0 = time.perf_counter(); done = 0
    with Pool(a.workers, initializer=init_worker) as pool:
        for fname, status, secs in pool.imap_unordered(run_task, todo):
            done += 1
            print(f'[{done}/{len(todo)}] {fname} {status} {secs:.0f}s (decorrido {(time.perf_counter()-t0)/60:.1f} min)', flush=True)

if __name__ == '__main__':
    main()
