"""Análise a posteriori (g): cenário de commits sem o identificador apache/hadoop.

Compara o cenário de commits refeito sem apache/hadoop (results/nohadoop, de run_nohadoop.py) com a análise
principal (results/tasks) e com o recálculo da análise (c), que retirava o identificador apenas do teste
(results/analysis/posthoc_commits_without_hadoop.csv). Regras de analyze.py e analyze_posthoc.py: ΔMCC por família
(média das cinco sementes); Wilcoxon pareado unilateral para ganho, com Holm sobre os três classificadores;
correlação bisserial de postos pareada; Hodges-Lehmann com intervalo de confiança exato de 95%.

Uso: python analyze_nohadoop.py [results] [results/nohadoop] [data]
Saídas em results/analysis/nohadoop_*.csv.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parent
ART = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results'
NOH = Path(sys.argv[2]) if len(sys.argv) > 2 else ART / 'nohadoop'
DATA = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / 'data'
OUT = ART / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ['mcc', 'f1', 'auc', 'precision', 'recall']
MODELS = ['LR', 'RF', 'SVM']
EPS = 1e-12


def tasks(d, scenario, models):
    out = {}
    for f in sorted((d / 'tasks').glob(f'{scenario}__*.json')):
        r = json.loads(f.read_text())
        if r['model'] in models:
            out[f.name] = r
    return out


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


def signed_rank_cdf(n):
    counts = np.zeros(n * (n + 1) // 2 + 1); counts[0] = 1.0
    for r in range(1, n + 1):
        shifted = np.zeros_like(counts); shifted[r:] = counts[:-r]; counts = counts + shifted
    return np.cumsum(counts / counts.sum())


def hodges_lehmann(v, conf=0.95):
    v = np.asarray(v, float); n = len(v)
    i, j = np.triu_indices(n); walsh = np.sort((v[i] + v[j]) / 2)
    cdf = signed_rank_cdf(n)
    k = int(np.searchsorted(cdf, (1 - conf) / 2, side='right')) - 1
    if k < 0:
        return float(np.median(walsh)), float(walsh[0]), float(walsh[-1]), 1.0
    return float(np.median(walsh)), float(walsh[k]), float(walsh[len(walsh) - 1 - k]), float(1 - 2 * cdf[k])


def unit_rows(rs, scenario):
    rows = []
    for r in rs.values():
        for t in ['full', 'hc']:
            if scenario == 'classes':
                rows.append(dict(model=r['model'], seed=r['seed'], unit=r['fold'], treatment=t, **{k: r[t][k] for k in METRICS}))
            else:
                for u, mu in r[t]['by_group']['family'].items():
                    rows.append(dict(model=r['model'], seed=r['seed'], unit=u, treatment=t, **{k: mu[k] for k in METRICS}))
    return pd.DataFrame(rows)


def per_unit_diff(rs, scenario):
    u = unit_rows(rs, scenario)
    w = u.pivot_table(index=['model', 'unit', 'seed'], columns='treatment', values=METRICS).reset_index()
    w.columns = ['_'.join(c).strip('_') for c in w.columns]
    for m in METRICS:
        w[f'd_{m}'] = w[f'{m}_hc'] - w[f'{m}_full']
    cols = [f'{m}_full' for m in METRICS] + [f'{m}_hc' for m in METRICS] + [f'd_{m}' for m in METRICS]
    return w.groupby(['model', 'unit'])[cols].mean().reset_index()


def tests(pu, models, label):
    rows = []
    for mo in models:
        g = pu[pu.model == mo]; d = g.d_mcc.to_numpy()
        nz = d[np.abs(d) > EPS]
        p = 1.0 if len(nz) == 0 else float(wilcoxon(d, alternative='greater', zero_method='wilcox').pvalue)
        p2 = 1.0 if len(nz) == 0 else float(wilcoxon(d, alternative='two-sided', zero_method='wilcox').pvalue)
        pl = 1.0 if len(nz) == 0 else float(wilcoxon(d, alternative='less', zero_method='wilcox').pvalue)
        q1, q3 = np.percentile(d, [25, 75]); hl, lo, hi, cf = hodges_lehmann(d)
        rows.append(dict(analysis=label, model=mo, n_units=len(d), n_positive=int((d > EPS).sum()),
                         n_negative=int((d < -EPS).sum()), median_dmcc=float(np.median(d)), mean_dmcc=float(d.mean()),
                         iqr=float(q3 - q1), rank_biserial=rank_biserial(d), p_greater=p, p_two_sided=p2, p_less=pl,
                         hodges_lehmann=hl, ci_low=lo, ci_high=hi, ci_conf=cf,
                         median_df1=float(g.d_f1.median()), median_dauc=float(g.d_auc.median())))
    t = pd.DataFrame(rows); t['p_holm'] = holm(t.p_greater.to_numpy())
    return t


def selections(rs):
    rows = []
    for r in rs.values():
        tr = r['trajectories']
        rows.append(dict(model=r['model'], fold=r['fold'], seed=r['seed'], n_available=r['n_available'],
                         n_selected=len(r['selected']), selected=' '.join(r['selected']), selected_mask=r['selected_mask'],
                         validation_mcc_full=r['validation_mcc_full'], validation_mcc_selected=r['validation_mcc_selected'],
                         validation_gain=r['validation_mcc_selected'] - r['validation_mcc_full'],
                         test_mcc_full=r['full']['mcc'], test_mcc_hc=r['hc']['mcc'], test_gain=r['hc']['mcc'] - r['full']['mcc'],
                         unique_evaluations=r['unique_evaluations'], search_seconds=r['search_seconds'],
                         n_warnings=r['n_warnings'], stops=','.join(t['stop'] for t in tr),
                         steps=','.join(str(t['steps']) for t in tr), max_steps=max(t['steps'] for t in tr),
                         origin=['completa', 'aleatória', 'aleatória'][[t['end'] for t in tr].index(r['selected_mask'])]
                         if r['selected_mask'] in [t['end'] for t in tr] else 'ponto avaliado fora do fim'))
    return pd.DataFrame(rows)


def trajectories(rs, label):
    rows = []
    for r in rs.values():
        for t in r['trajectories']:
            rows.append(dict(analysis=label, model=r['model'], fold=r['fold'], seed=r['seed'], trajectory=t['trajectory'],
                             init='completa' if t['trajectory'] == 0 else 'aleatória', start_size=t['start'].count('1'),
                             end_size=t['end'].count('1'), steps=t['steps'], new_evaluations=t['new_evaluations'],
                             stop=t['stop'], end_mcc=t['end_mcc']))
    return pd.DataFrame(rows)


new = tasks(NOH, 'commits', MODELS)
assert len(new) == 15 and all(r['full']['n'] == 24690 for r in new.values()), 'resultados incompletos'
art = tasks(ART, 'commits', MODELS)

# 1. Conjuntos de dados sem apache/hadoop (contagens e prevalência)
j = DATA / 'commits' / 'apachejit_total.csv'
if j.exists():
    d = pd.read_csv(j); d = d[d.project != 'apache/hadoop']
    dev = d[d.year <= 2016]; te = d[d.year >= 2017]
    rows = [dict(conjunto='desenvolvimento (até 2016)', registros=len(dev), positivos=int(dev.buggy.sum()))]
    rows += [dict(conjunto=f'validação {y}', registros=int((dev.year == y).sum()), positivos=int(dev[dev.year == y].buggy.sum()))
             for y in (2014, 2015, 2016)]
    rows += [dict(conjunto='teste (2017 a 2019)', registros=len(te), positivos=int(te.buggy.sum()))]
    s = pd.DataFrame(rows); s['prevalencia_pct'] = 100 * s.positivos / s.registros
    s.round(4).to_csv(OUT / 'nohadoop_datasets.csv', index=False)

# 2. Desempenho agregado no teste: por semente e médio; comparação com o artigo e com a análise (c)
agg = []
for r in new.values():
    for t in ['full', 'hc']:
        agg.append(dict(model=r['model'], seed=r['seed'], treatment=t, **{k: r[t][k] for k in METRICS + ['tp', 'fp', 'tn', 'fn']}))
agg = pd.DataFrame(agg).sort_values(['model', 'seed', 'treatment'])
agg.round(6).to_csv(OUT / 'nohadoop_test_by_seed.csv', index=False)
cmp = agg.groupby(['model', 'treatment'])[METRICS].mean().reset_index().assign(setting='g_retreinado_sem_hadoop')
ref = []
for r in art.values():
    for t in ['full', 'hc']:
        ref.append(dict(model=r['model'], seed=r['seed'], treatment=t, **{k: r[t][k] for k in METRICS}))
ref = pd.DataFrame(ref).groupby(['model', 'treatment'])[METRICS].mean().reset_index().assign(setting='principal_teste_completo')
parts = [ref, cmp]
ph = ART / 'analysis' / 'posthoc_commits_without_hadoop.csv'
if ph.exists():
    c = pd.read_csv(ph)[['model', 'treatment'] + METRICS].assign(setting='c_hadoop_fora_do_teste')
    parts.insert(1, c)
pd.concat(parts)[['setting', 'model', 'treatment'] + METRICS].round(6).to_csv(OUT / 'nohadoop_test_aggregate.csv', index=False)

# 3. ΔMCC por família e testes
pu = per_unit_diff(new, 'commits'); pu.round(6).to_csv(OUT / 'nohadoop_by_family.csv', index=False)
t = tests(pu, MODELS, 'g_retreinado_sem_hadoop'); t.round(6).to_csv(OUT / 'nohadoop_tests.csv', index=False)

# 4. Subconjuntos e trajetórias
sel = selections(new); sel.to_csv(OUT / 'nohadoop_selections_raw.csv', index=False)
rows = []
for mo, g in sel.groupby('model'):
    rows.append(dict(model=mo, searches=len(g), distinct_subsets=g.selected.nunique(), size_mean=g.n_selected.mean(),
                     size_min=int(g.n_selected.min()), size_max=int(g.n_selected.max()),
                     reduction_pct=100 * (1 - g.n_selected / g.n_available).mean(), validation_gain=g.validation_gain.mean(),
                     test_gain=g.test_gain.mean(), search_hours=g.search_seconds.sum() / 3600, max_steps=int(g.max_steps.max()),
                     stopped_by_budget=int(sum(s.count('budget') for s in g.stops)), warnings=int(g.n_warnings.sum())))
pd.DataFrame(rows).round(6).to_csv(OUT / 'nohadoop_selections.csv', index=False)
trajectories(new, 'g_retreinado_sem_hadoop').to_csv(OUT / 'nohadoop_trajectories.csv', index=False)

print(pd.read_csv(OUT / 'nohadoop_test_aggregate.csv').round(4).to_string(index=False))
print(t[['model', 'n_units', 'n_positive', 'n_negative', 'median_dmcc', 'iqr', 'rank_biserial', 'p_greater', 'p_holm',
         'hodges_lehmann', 'ci_low', 'ci_high', 'median_df1', 'median_dauc']].round(4).to_string(index=False))
print(pd.read_csv(OUT / 'nohadoop_selections.csv').round(4).to_string(index=False))
for mo, g in sel.groupby('model'):
    print(mo, sorted(set(g.selected)))
