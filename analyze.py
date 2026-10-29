"""Análise das questões de pesquisa (seção 3.7 do artigo).

Lê results/tasks/*.json e escreve results/analysis/*.csv e summary.json.
QP1: desempenho dos modelos com todos os atributos.
QP2: ΔMCC = MCC(hc) − MCC(full), pareado por modelo, unidade e semente;
     sementes agregadas pela média dentro de cada unidade (projeto ou família);
     Wilcoxon pareado unilateral (alternativa: deslocamento positivo), Holm sobre 6 testes,
     mediana, IIQ e correlação bisserial de postos pareada.
QP3: frequência de seleção dos atributos e redução relativa.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results'
OUT = RES / 'analysis'; OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['LR', 'RF', 'SVM']; SCEN = ['classes', 'commits']
METRICS = ['mcc', 'f1', 'auc', 'precision', 'recall']

def load():
    rows, units, sel = [], [], []
    for f in sorted((RES / 'tasks').glob('*.json')):
        r = json.loads(f.read_text())
        base = dict(scenario=r['scenario'], fold=r['fold'], model=r['model'], seed=r['seed'])
        for t in ['full', 'hc']:
            m = {k: v for k, v in r[t].items() if k != 'by_group'}
            rows.append({**base, 'treatment': t, **m})
            # unidade de análise: projeto de teste (classes) ou família (commits)
            if r['scenario'] == 'classes':
                units.append({**base, 'treatment': t, 'unit': r['fold'], **{k: m[k] for k in METRICS + ['n', 'positives', 'degenerate', 'tp', 'fp', 'tn', 'fn']}})
            else:
                for u, mu in r[t]['by_group']['family'].items():
                    units.append({**base, 'treatment': t, 'unit': u, **{k: mu[k] for k in METRICS + ['n', 'positives', 'degenerate', 'tp', 'fp', 'tn', 'fn']}})
        sel.append({**base, 'n_available': r['n_available'], 'n_selected': len(r['selected']),
                    'selected': ' '.join(r['selected']), 'validation_mcc_full': r['validation_mcc_full'],
                    'validation_mcc_selected': r['validation_mcc_selected'], 'unique_evaluations': r['unique_evaluations'],
                    'search_seconds': r['search_seconds'], 'n_warnings': r['n_warnings'],
                    'stops': ','.join(t['stop'] for t in r['trajectories'])})
    return pd.DataFrame(rows), pd.DataFrame(units), pd.DataFrame(sel)

def rank_biserial(d):
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d)); wp = r[d > 0].sum(); wm = r[d < 0].sum()
    return float((wp - wm) / (wp + wm))

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj

def main():
    runs, units, sel = load()
    runs.to_csv(OUT / 'runs_overall.csv', index=False)
    units.to_csv(OUT / 'runs_by_unit.csv', index=False)
    sel.to_csv(OUT / 'selections.csv', index=False)
    summary = {'n_tasks': int(len(sel)), 'tasks_by_scenario_model': sel.groupby(['scenario', 'model']).size().to_dict()}
    summary['tasks_by_scenario_model'] = {f'{a}|{b}': int(v) for (a, b), v in summary['tasks_by_scenario_model'].items()}

    # ---------------- QP1: modelos completos
    full = units[units.treatment == 'full']
    qp1_unit = full.groupby(['scenario', 'model', 'unit'])[METRICS + ['n', 'positives']].mean().reset_index()
    qp1_unit['prevalence'] = qp1_unit.positives / qp1_unit.n
    qp1_unit.to_csv(OUT / 'qp1_full_by_unit.csv', index=False)
    fo = runs[runs.treatment == 'full']
    qp1_overall = fo.groupby(['scenario', 'model'])[METRICS].agg(['mean', 'median', 'std']).round(4)
    qp1_overall.to_csv(OUT / 'qp1_full_overall.csv')
    # commits: teste agregado (uma única partição externa)
    cm = fo[fo.scenario == 'commits'].groupby('model')[METRICS + ['tp', 'fp', 'tn', 'fn']].mean().round(4)
    cm.to_csv(OUT / 'qp1_commits_aggregate.csv')
    # classes: média entre os 10 projetos
    cl = qp1_unit[qp1_unit.scenario == 'classes'].groupby('model')[METRICS].agg(['mean', 'median', 'min', 'max']).round(4)
    cl.to_csv(OUT / 'qp1_classes_across_projects.csv')

    # ---------------- QP2: efeito da seleção
    w = units.pivot_table(index=['scenario', 'model', 'unit', 'seed'], columns='treatment', values=METRICS).reset_index()
    w.columns = ['_'.join(c).strip('_') for c in w.columns]
    for m in METRICS:
        w[f'd_{m}'] = w[f'{m}_hc'] - w[f'{m}_full']
    w.to_csv(OUT / 'qp2_differences_by_seed.csv', index=False)
    seedvar = w.groupby(['scenario', 'model', 'unit']).d_mcc.agg(['mean', 'std', 'min', 'max']).reset_index()
    seedvar['range'] = seedvar['max'] - seedvar['min']
    seedvar.to_csv(OUT / 'qp2_seed_variation.csv', index=False)
    per_unit = w.groupby(['scenario', 'model', 'unit'])[[f'd_{m}' for m in METRICS]].mean().reset_index()
    per_unit.to_csv(OUT / 'qp2_differences_by_unit.csv', index=False)

    tests = []
    for s in SCEN:
        for mo in MODELS:
            d = per_unit[(per_unit.scenario == s) & (per_unit.model == mo)].d_mcc.to_numpy()
            if len(d) == 0:
                continue
            nz = d[np.abs(d) > 1e-12]
            if len(nz) == 0:
                stat, p = np.nan, 1.0
            else:
                res = wilcoxon(d, alternative='greater', zero_method='wilcox')
                stat, p = float(res.statistic), float(res.pvalue)
            q1, q3 = np.percentile(d, [25, 75])
            tests.append(dict(scenario=s, model=mo, n_units=len(d), n_zero=int(len(d) - len(nz)),
                              n_positive=int((d > 1e-12).sum()), n_negative=int((d < -1e-12).sum()),
                              median_dmcc=float(np.median(d)), mean_dmcc=float(np.mean(d)), q1=float(q1), q3=float(q3),
                              iqr=float(q3 - q1), W=stat, p=p, rank_biserial=rank_biserial(d),
                              median_df1=float(per_unit[(per_unit.scenario == s) & (per_unit.model == mo)].d_f1.median()),
                              median_dauc=float(per_unit[(per_unit.scenario == s) & (per_unit.model == mo)].d_auc.median())))
    t = pd.DataFrame(tests)
    if len(t):
        t['p_holm'] = holm(t.p.to_numpy())
        t['reject_H0_5pct'] = t.p_holm < 0.05
    t.to_csv(OUT / 'qp2_wilcoxon_holm.csv', index=False)
    summary['qp2'] = t.to_dict(orient='records')

    # Análise de sensibilidade (não planejada no protocolo; sugerida na revisão):
    # classes sem os projetos de prevalência extrema (Log4j 92,2% e Xalan 98,8%).
    EXTREME = ['log4j', 'xalan']
    sens = []
    for mo in MODELS:
        d = per_unit[(per_unit.scenario == 'classes') & (per_unit.model == mo) & (~per_unit.unit.isin(EXTREME))].d_mcc.to_numpy()
        res = wilcoxon(d, alternative='greater', zero_method='wilcox')
        fullm = units[(units.scenario == 'classes') & (units.model == mo) & (units.treatment == 'full') & (~units.unit.isin(EXTREME))].groupby('unit').mcc.mean().mean()
        sens.append(dict(scenario='classes_sem_log4j_xalan', model=mo, n_units=len(d), n_positive=int((d > 1e-12).sum()),
                         n_negative=int((d < -1e-12).sum()), median_dmcc=float(np.median(d)), p=float(res.pvalue),
                         rank_biserial=rank_biserial(d), mean_full_mcc=float(fullm)))
    s = pd.DataFrame(sens); s['p_holm_3'] = holm(s.p.to_numpy())
    s.to_csv(OUT / 'qp2_sensitivity_without_extreme_prevalence.csv', index=False)
    summary['qp2_sensitivity'] = s.to_dict(orient='records')

    # ---------------- QP3: frequência e redução
    freq = []
    for (s, mo), g in sel.groupby(['scenario', 'model']):
        p = int(g.n_available.iloc[0]); runs_n = len(g)
        counts = {}
        for x in g.selected:
            for a in x.split():
                counts[a] = counts.get(a, 0) + 1
        cols = (__import__('run_experiment').CF if s == 'classes' else __import__('run_experiment').JF)
        for a in cols:
            freq.append(dict(scenario=s, model=mo, attribute=a, count=counts.get(a, 0), runs=runs_n,
                             frequency=counts.get(a, 0) / runs_n))
    pd.DataFrame(freq).to_csv(OUT / 'qp3_attribute_frequency.csv', index=False)
    sel['reduction'] = 1 - sel.n_selected / sel.n_available
    red = sel.groupby(['scenario', 'model']).agg(n_available=('n_available', 'first'),
                                                 selected_mean=('n_selected', 'mean'), selected_median=('n_selected', 'median'),
                                                 selected_min=('n_selected', 'min'), selected_max=('n_selected', 'max'),
                                                 reduction_mean=('reduction', 'mean'), reduction_median=('reduction', 'median')).round(4)
    red.to_csv(OUT / 'qp3_reduction.csv')
    summary['qp3_reduction'] = {f'{a}|{b}': v for (a, b), v in red.to_dict(orient='index').items()}

    # ---------------- custo e avisos
    cost = sel.groupby(['scenario', 'model']).agg(searches=('seed', 'size'), evaluations_mean=('unique_evaluations', 'mean'),
                                                  search_hours_total=('search_seconds', lambda x: x.sum() / 3600),
                                                  warnings_total=('n_warnings', 'sum')).round(3)
    cost.to_csv(OUT / 'cost.csv')
    stops = sel.stops.str.split(',').explode().value_counts().to_dict()
    summary['trajectory_stops'] = stops
    summary['degenerate_unit_runs'] = int(units.degenerate.sum())
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float)[:3000])

if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
