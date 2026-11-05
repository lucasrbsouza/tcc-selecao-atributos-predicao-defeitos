"""Análise complementar: Hill Climbing (HC) × filtro F × seleção embutida L1, sob o mesmo protocolo.

Unidades e agregação seguem a QP2: projetos (classes) ou famílias (commits); média das cinco sementes.
Para cada método: ΔMCC = MCC(método) − MCC(todos os atributos), Wilcoxon pareado unilateral (deslocamento
positivo) e correção de Holm sobre as seis combinações do método. A comparação direta entre métodos de seleção
(método − HC) é descritiva (mediana e contagem de sinais), para não multiplicar testes confirmatórios.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from analyze import rank_biserial, holm, MODELS

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'; ALT = RES / 'alternatives'; OUT = ALT / 'analysis'; OUT.mkdir(parents=True, exist_ok=True)
METHODS = {'hc': 'Hill Climbing', 'filter': 'Filtro F', 'l1': 'L1'}

def unit_rows(scenario, fold, block):
    if scenario == 'classes':
        return {fold: block['mcc']}
    return {u: m['mcc'] for u, m in block['by_group']['family'].items()}

rows, sel, overall = [], [], []
for f in sorted((ALT / 'tasks').glob('*.json')):
    a = json.loads(f.read_text()); o = json.loads((RES / 'tasks' / f.name).read_text())
    base = dict(scenario=a['scenario'], fold=a['fold'], model=a['model'], seed=a['seed'])
    blocks = {'full': o['full'], 'hc': o['hc'], 'filter': a['filter']['test'], 'l1': a['l1']['test']}
    assert abs(blocks['full']['n'] - a['filter']['test']['n']) == 0
    for meth, b in blocks.items():
        overall.append({**base, 'method': meth, **{k: b[k] for k in ['mcc', 'f1', 'auc', 'precision', 'recall', 'n_features'] if k in b}})
        for u, v in unit_rows(a['scenario'], a['fold'], b).items():
            rows.append({**base, 'method': meth, 'unit': u, 'mcc': v})
    sel.append({**base, 'method': 'hc', 'n_selected': len(o['selected']), 'n_available': o['n_available'],
                'selected': ' '.join(o['selected']), 'validation_gain': o['validation_mcc_selected'] - o['validation_mcc_full'],
                'search_seconds': o['search_seconds']})
    for meth in ['filter', 'l1']:
        sel.append({**base, 'method': meth, 'n_selected': len(a[meth]['selected']), 'n_available': a['n_available'],
                    'selected': ' '.join(a[meth]['selected']), 'param': a[meth]['param'],
                    'validation_gain': a[meth]['validation_mcc'] - o['validation_mcc_full'],
                    'search_seconds': a[meth]['search_seconds']})

U = pd.DataFrame(rows); S = pd.DataFrame(sel); O = pd.DataFrame(overall)
U.to_csv(OUT / 'runs_by_unit.csv', index=False); S.to_csv(OUT / 'selections.csv', index=False); O.to_csv(OUT / 'runs_overall.csv', index=False)
pu = U.groupby(['scenario', 'model', 'unit', 'method']).mcc.mean().unstack('method').reset_index()
for m in METHODS:
    pu[f'd_{m}'] = pu[m] - pu['full']
pu['filter_minus_hc'] = pu['filter'] - pu['hc']; pu['l1_minus_hc'] = pu['l1'] - pu['hc']
pu.to_csv(OUT / 'differences_by_unit.csv', index=False)

tests = []
for meth in METHODS:
    block = []
    for s in ['classes', 'commits']:
        for mo in MODELS:
            d = pu[(pu.scenario == s) & (pu.model == mo)][f'd_{meth}'].to_numpy()
            nz = d[np.abs(d) > 1e-12]
            p = 1.0 if len(nz) == 0 else float(wilcoxon(d, alternative='greater').pvalue)
            block.append(dict(method=meth, scenario=s, model=mo, n_units=len(d), n_positive=int((d > 1e-12).sum()),
                              n_negative=int((d < -1e-12).sum()), median_dmcc=float(np.median(d)), p=p, rank_biserial=rank_biserial(d)))
    b = pd.DataFrame(block); b['p_holm'] = holm(b.p.to_numpy()); tests.append(b)
T = pd.concat(tests); T.to_csv(OUT / 'wilcoxon_vs_full.csv', index=False)

vs = []
for s in ['classes', 'commits']:
    for mo in MODELS:
        g = pu[(pu.scenario == s) & (pu.model == mo)]
        for col in ['filter_minus_hc', 'l1_minus_hc']:
            d = g[col].to_numpy()
            vs.append(dict(scenario=s, model=mo, comparison=col, median=float(np.median(d)),
                           n_positive=int((d > 1e-12).sum()), n_negative=int((d < -1e-12).sum())))
pd.DataFrame(vs).to_csv(OUT / 'methods_vs_hc.csv', index=False)

S['reduction'] = 1 - S.n_selected / S.n_available
red = S.groupby(['method', 'scenario', 'model']).agg(selected_mean=('n_selected', 'mean'), reduction_mean=('reduction', 'mean'),
                                                     validation_gain_mean=('validation_gain', 'mean'),
                                                     search_hours=('search_seconds', lambda x: x.sum() / 3600)).reset_index()
tg = O.pivot_table(index=['scenario', 'fold', 'model', 'seed'], columns='method', values='mcc').reset_index()
for m in METHODS:
    tg[f'tg_{m}'] = tg[m] - tg['full']
tgm = tg.groupby(['scenario', 'model'])[[f'tg_{m}' for m in METHODS]].mean().reset_index().melt(['scenario', 'model'], var_name='method', value_name='test_gain_mean')
tgm['method'] = tgm.method.str.replace('tg_', '')
red = red.merge(tgm, on=['method', 'scenario', 'model'])
red.to_csv(OUT / 'reduction_cost_optimism.csv', index=False)
agg = O[O.scenario == 'commits'].groupby(['model', 'method']).mcc.mean().unstack('method')
agg.to_csv(OUT / 'commits_aggregate_mcc.csv')
cl = U[U.scenario == 'classes'].groupby(['model', 'unit', 'method']).mcc.mean().groupby(['model', 'method']).mean().unstack('method')
cl.to_csv(OUT / 'classes_mean_mcc.csv')
print(T.round(3).to_string()); print(pd.DataFrame(vs).round(3).to_string()); print(red.round(3).to_string()); print(agg.round(3)); print(cl.round(3))
