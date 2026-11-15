"""Análises a posteriori (seção 3.8 do artigo).

Não fazem parte do protocolo congelado antes da execução definitiva. Usam apenas os registros e as
previsões já gravados em results/ (nenhum modelo é reajustado):

1. Trajetórias: motivo de término, passos e tamanho final, por cenário, modelo e inicialização.
2. Origem da máscara escolhida (inicialização completa ou aleatória) e distância ao ponto de partida.
3. Ganho de MCC nas validações internas (máscara escolhida − conjunto completo): mínimo e média.
4. Wilcoxon bilateral exploratório sobre as diferenças por unidade (complemento do teste unilateral da QP2).
5. Teste de commits sem o identificador apache/hadoop (sem positivos na base): MCC, F1, precisão e
   sensibilidade a partir das matrizes de confusão por identificador; AUC-ROC a partir das previsões,
   se os dados estiverem em data/ (python download_data.py).
6. Estimador de Hodges-Lehmann (mediana das médias de Walsh) e intervalo de confiança exato, baseado
   na distribuição nula da estatística de Wilcoxon, para o ΔMCC por unidade de análise.

Uso: python analyze_posthoc.py [results] [data]
Saída: results/analysis/posthoc_*.csv
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results'
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'data'
OUT = RES / 'analysis'; OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['LR', 'RF', 'SVM']; SCEN = ['classes', 'commits']
INIT = {0: 'completa', 1: 'aleatoria_1', 2: 'aleatoria_2'}
HADOOP = 'hadoop'   # identificador apache/hadoop, como gravado em by_group['project_id']


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def mcc_from_counts(tp, fp, tn, fn):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


tasks = [json.loads(f.read_text()) for f in sorted((RES / 'tasks').glob('*.json'))]

# 1. Trajetórias ---------------------------------------------------------------------------------
traj = pd.DataFrame([dict(scenario=r['scenario'], model=r['model'], init=INIT[t['trajectory']],
                          stop=t['stop'], steps=t['steps'], n_start=t['start'].count('1'),
                          n_end=t['end'].count('1'))
                     for r in tasks for t in r['trajectories']])
t1 = traj.groupby(['scenario', 'model', 'init']).agg(
    trajetorias=('stop', 'size'), encerradas_orcamento=('stop', lambda x: int((x == 'budget').sum())),
    otimo_local=('stop', lambda x: int((x == 'local_optimum').sum())),
    passos_media=('steps', 'mean'), passos_max=('steps', 'max'),
    atributos_inicio=('n_start', 'mean'), atributos_fim=('n_end', 'mean'), atributos_fim_min=('n_end', 'min')
).round(2).reset_index()
t1.to_csv(OUT / 'posthoc_trajectories.csv', index=False)

# 2. Origem da máscara escolhida -----------------------------------------------------------------
rows = []
for r in tasks:
    best = r['selected_mask']
    origin = next(t['trajectory'] for t in r['trace'] if t['mask'] == best)   # 1ª trajetória que a avaliou
    start = r['trajectories'][origin]['start']
    rows.append(dict(scenario=r['scenario'], fold=r['fold'], model=r['model'], seed=r['seed'],
                     origin=INIT[origin], n_selected=best.count('1'), n_start=start.count('1'),
                     hamming_start=hamming(best, start),
                     start_1=r['trajectories'][1]['start'], start_2=r['trajectories'][2]['start'],
                     gain_validation=r['validation_mcc_selected'] - r['validation_mcc_full']))
sel = pd.DataFrame(rows)
t2 = sel.assign(completa=sel.origin.eq('completa'), aleatoria=~sel.origin.eq('completa')).groupby(
    ['scenario', 'model']).agg(buscas=('seed', 'size'), origem_completa=('completa', 'sum'),
                               origem_aleatoria=('aleatoria', 'sum'), selecionados_media=('n_selected', 'mean'),
                               inicio_media=('n_start', 'mean'), distancia_media=('hamming_start', 'mean'),
                               distancia_max=('hamming_start', 'max')).round(2).reset_index()
t2.to_csv(OUT / 'posthoc_mask_origin.csv', index=False)
distinct = {s: len(set(sel[sel.scenario == s].start_1) | set(sel[sel.scenario == s].start_2)) for s in SCEN}

# 3. Ganho interno -------------------------------------------------------------------------------
t3 = sel.groupby(['scenario', 'model']).gain_validation.agg(['min', 'mean', 'max']).round(6).reset_index()
t3.to_csv(OUT / 'posthoc_validation_gain.csv', index=False)

# 4. Wilcoxon bilateral exploratório -------------------------------------------------------------
d = pd.read_csv(OUT / 'qp2_differences_by_unit.csv')
t4 = pd.DataFrame([dict(scenario=s, model=m, n=len(v),
                        p_bilateral=wilcoxon(v, alternative='two-sided').pvalue,
                        p_unilateral_piora=wilcoxon(v, alternative='less').pvalue)
                   for s in SCEN for m in MODELS
                   for v in [d[(d.scenario == s) & (d.model == m)].d_mcc.to_numpy()]]).round(6)
t4.to_csv(OUT / 'posthoc_wilcoxon_two_sided.csv', index=False)


# 6. Hodges-Lehmann e intervalo de confiança exato ------------------------------------------------
def signed_rank_cdf(n):
    """P(T <= k) para a soma de postos de Wilcoxon sob H0, sem empates (enumeração exata)."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for r in range(1, n + 1):
        shifted = np.zeros_like(counts)
        shifted[r:] = counts[:-r]
        counts = counts + shifted
    return np.cumsum(counts / counts.sum())


def hodges_lehmann(v, conf=0.95):
    v = np.asarray(v, dtype=float)
    n = len(v)
    i, j = np.triu_indices(n)
    walsh = np.sort((v[i] + v[j]) / 2)
    cdf = signed_rank_cdf(n)
    k = int(np.searchsorted(cdf, (1 - conf) / 2, side='right')) - 1   # maior k com P(T <= k) <= alfa/2
    if k < 0:
        return np.median(walsh), walsh[0], walsh[-1], 1.0
    achieved = 1 - 2 * cdf[k]                                         # k médias descartadas em cada cauda
    return np.median(walsh), walsh[k], walsh[len(walsh) - 1 - k], achieved


rows6 = []
for s in SCEN:
    for m in MODELS:
        v = d[(d.scenario == s) & (d.model == m)].d_mcc.to_numpy()
        hl, lo, hi, conf = hodges_lehmann(v)
        rows6.append(dict(scenario=s, model=m, n=len(v), mediana=np.median(v), hodges_lehmann=hl,
                          ic_inferior=lo, ic_superior=hi, confianca=conf))
t6 = pd.DataFrame(rows6).round(6)
t6.to_csv(OUT / 'posthoc_hodges_lehmann.csv', index=False)

# 5. Commits sem apache/hadoop -------------------------------------------------------------------
keep = None
csv = DATA / 'commits' / 'apachejit_total.csv'
if csv.exists():
    j = pd.read_csv(csv)
    te = j[j.year >= 2017].reset_index(drop=True)
    keep = (te.project != 'apache/hadoop').to_numpy(); y_ref = te.buggy.astype(int).to_numpy()
rows = []
for r in tasks:
    if r['scenario'] != 'commits':
        continue
    z = np.load(RES / 'predictions' / f"commits__{r['fold']}__{r['model']}__{r['seed']}.npz") if keep is not None else None
    for tr in ['full', 'hc']:
        b = r[tr]; h = b['by_group']['project_id'][HADOOP]
        tp, fp, tn, fn = (b[k] - h[k] for k in ['tp', 'fp', 'tn', 'fn'])
        row = dict(model=r['model'], seed=r['seed'], treatment=tr, n=tp + fp + tn + fn, positives=tp + fn,
                   mcc=mcc_from_counts(tp, fp, tn, fn), precision=tp / (tp + fp), recall=tp / (tp + fn),
                   f1=2 * tp / (2 * tp + fp + fn), fp_hadoop=h['fp'], mcc_all=b['mcc'])
        if z is not None:
            from sklearn.metrics import roc_auc_score
            assert (z['y'] == y_ref).all(), 'ordem do teste difere das previsões gravadas'
            row['auc'] = roc_auc_score(z['y'][keep], z[tr + '_score'][keep])
        rows.append(row)
t5 = pd.DataFrame(rows).groupby(['model', 'treatment']).mean(numeric_only=True).drop(columns='seed').round(6).reset_index()
t5.to_csv(OUT / 'posthoc_commits_without_hadoop.csv', index=False)

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
for name, t in [('Trajetórias', t1), ('Origem da máscara escolhida', t2), ('Ganho interno', t3),
                ('Wilcoxon bilateral (exploratório)', t4), ('Commits sem apache/hadoop', t5)]:
    print(f'\n== {name}\n{t.to_string(index=False)}')
print('\nMáscaras aleatórias distintas por cenário:', distinct)
print('Tabelas em', OUT)
