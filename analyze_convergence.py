"""Análise a posteriori (f): Hill Climbing nas classes sem limite de orçamento.

Compara as buscas levadas à convergência (results/convergence, de run_convergence.py) com a análise
principal (results/tasks), que continua sendo a do protocolo congelado. O que é reportado está em
REGISTRO_ANALISE_F.md: ΔMCC por projeto (média das cinco sementes), ganho interno, passos e número de
atributos finais; Wilcoxon pareado unilateral para ganho, com Holm sobre os três classificadores;
correlação bisserial de postos pareada; Hodges-Lehmann com intervalo de confiança exato de 95%.

Uso: python analyze_convergence.py [results] [results/convergence]
Saídas em results/analysis/convergence_*.csv.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parent
ART = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results'
CONV = Path(sys.argv[2]) if len(sys.argv) > 2 else ART / 'convergence'
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


new = tasks(CONV, 'classes', MODELS)
art = {k: v for k, v in tasks(ART, 'classes', MODELS).items() if k in new}
assert new and set(new) == set(art), 'tarefas de convergência sem correspondente na análise principal'
MA = [m for m in MODELS if any(r['model'] == m for r in new.values())]

# 1. Conferência: tudo o que a análise principal avaliou reaparece com o mesmo valor; trajetórias que
#    já convergiam terminam no mesmo ponto; linha de base idêntica nos modelos determinísticos.
chk = []
for n, r in new.items():
    o = art[n]; ev = {e['mask']: e['mcc_validation'] for e in r['trace']}
    chk.append(dict(task=n, model=r['model'], original_evaluations=len(o['trace']), new_evaluations=len(r['trace']),
                    missing=sum(e['mask'] not in ev for e in o['trace']),
                    max_abs_diff=max((abs(ev[e['mask']] - e['mcc_validation']) for e in o['trace'] if e['mask'] in ev), default=0.0),
                    converged_trajectories_same_end=all(a['end'] == b['end'] and a['steps'] == b['steps']
                                                        for a, b in zip(o['trajectories'], r['trajectories'])
                                                        if a['stop'] == 'local_optimum'),
                    steps_not_fewer=all(b['steps'] >= a['steps'] for a, b in zip(o['trajectories'], r['trajectories'])),
                    baseline_abs_diff=abs(o['full']['mcc'] - r['full']['mcc']),
                    all_local_optimum=all(t['stop'] == 'local_optimum' for t in r['trajectories']),
                    n_warnings=r['n_warnings']))
chk = pd.DataFrame(chk); chk.round(12).to_csv(OUT / 'convergence_check.csv', index=False)

# 2. Trajetórias: passos, atributos finais e paradas, por classificador e inicialização
tr = pd.concat([trajectories(art, 'principal'), trajectories(new, 'convergida')])
tr.to_csv(OUT / 'convergence_trajectories_raw.csv', index=False)
rows = []
for (lab, mo, ini), g in tr.groupby(['analysis', 'model', 'init']):
    rows.append(dict(analysis=lab, model=mo, init=ini, trajectories=len(g), stopped_by_budget=int((g.stop == 'budget').sum()),
                     steps_median=g.steps.median(), steps_max=int(g.steps.max()), end_size_median=g.end_size.median(),
                     end_size_min=int(g.end_size.min()), end_size_max=int(g.end_size.max()),
                     new_evaluations_mean=g.new_evaluations.mean()))
pd.DataFrame(rows).round(4).to_csv(OUT / 'convergence_trajectories.csv', index=False)

# 3. Subconjuntos escolhidos: tamanho, origem, ganho interno e ganho no teste
sel = pd.concat([selections(art).assign(analysis='principal'), selections(new).assign(analysis='convergida')])
sel.to_csv(OUT / 'convergence_selections_raw.csv', index=False)
rows = []
for (lab, mo), g in sel.groupby(['analysis', 'model']):
    rows.append(dict(analysis=lab, model=mo, searches=len(g), size_mean=g.n_selected.mean(), size_min=int(g.n_selected.min()),
                     size_max=int(g.n_selected.max()), reduction_pct=100 * (1 - g.n_selected / g.n_available).mean(),
                     from_random_start=int((g.origin == 'aleatória').sum()), validation_gain=g.validation_gain.mean(),
                     test_gain=g.test_gain.mean(), evaluations_mean=g.unique_evaluations.mean(),
                     search_hours=g.search_seconds.sum() / 3600, max_steps=int(g.max_steps.max())))
pd.DataFrame(rows).round(6).to_csv(OUT / 'convergence_selections.csv', index=False)
a = sel[sel.analysis == 'principal'].set_index(['model', 'fold', 'seed']); c = sel[sel.analysis == 'convergida'].set_index(['model', 'fold', 'seed'])
changed = (a.selected_mask != c.selected_mask.reindex(a.index)).groupby('model').sum()

# 4. Frequência de seleção por atributo
freq = []
for (lab, mo), g in sel.groupby(['analysis', 'model']):
    allc = ' '.join(g.selected).split()
    for col in sorted(set(' '.join(sel.selected).split())):
        freq.append(dict(analysis=lab, model=mo, attribute=col, frequency=allc.count(col) / len(g)))
pd.DataFrame(freq).round(4).to_csv(OUT / 'convergence_attribute_frequency.csv', index=False)

# 5. ΔMCC por projeto e testes
pu_new = per_unit_diff(new, 'classes'); pu_old = per_unit_diff(art, 'classes')
m = pu_old[['model', 'unit', 'mcc_full', 'mcc_hc', 'd_mcc']].merge(
    pu_new[['model', 'unit', 'mcc_full', 'mcc_hc', 'd_mcc', 'd_f1', 'd_auc']], on=['model', 'unit'],
    suffixes=('_principal', '_convergida'))
m['convergida_menos_principal'] = m.mcc_hc_convergida - m.mcc_hc_principal
m.round(6).to_csv(OUT / 'convergence_by_project.csv', index=False)
t_new = tests(pu_new, MA, 'convergida'); t_old = tests(pu_old, MA, 'principal (Holm só entre estes modelos)')
pd.concat([t_old, t_new]).round(6).to_csv(OUT / 'convergence_tests.csv', index=False)
cmp = []
for mo in MA:
    d = m[m.model == mo].convergida_menos_principal.to_numpy(); nz = d[np.abs(d) > EPS]
    cmp.append(dict(model=mo, n_positive=int((d > EPS).sum()), n_negative=int((d < -EPS).sum()),
                    n_zero=int((np.abs(d) <= EPS).sum()), median=float(np.median(d)), mean=float(d.mean()),
                    p_two_sided=1.0 if len(nz) == 0 else float(wilcoxon(d, zero_method='wilcox').pvalue),
                    masks_changed=int(changed[mo])))
pd.DataFrame(cmp).round(6).to_csv(OUT / 'convergence_vs_main.csv', index=False)

print('Conferência:', dict(tarefas=len(chk), ausentes=int(chk.missing.sum()), dif_max=float(chk.max_abs_diff.max()),
                           mesmas_convergidas=bool(chk.converged_trajectories_same_end.all()),
                           passos_nao_menores=bool(chk.steps_not_fewer.all()),
                           todas_em_otimo_local=bool(chk.all_local_optimum.all()), avisos=int(chk.n_warnings.sum())))
print(pd.read_csv(OUT / 'convergence_selections.csv').to_string(index=False))
print(pd.concat([t_old, t_new])[['analysis', 'model', 'n_positive', 'n_negative', 'median_dmcc', 'p_greater', 'p_holm',
                                 'p_two_sided', 'p_less', 'hodges_lehmann', 'ci_low', 'ci_high']].round(4).to_string(index=False))
print(pd.DataFrame(cmp).round(4).to_string(index=False))
