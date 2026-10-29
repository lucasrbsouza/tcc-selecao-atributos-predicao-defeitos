"""Figuras do artigo (PNG 300 dpi, fundo branco, legíveis em impressão em tons de cinza).

Figura A: ΔMCC (seleção − todos os atributos) por unidade de análise, por modelo e cenário.
Figura B: frequência de seleção dos atributos por modelo e cenário.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results'
A = RES / 'analysis'; FIG = RES / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from run_experiment import CF, JF

MODELS = ['LR', 'RF', 'SVM']
LABEL = {'LR': 'Regressão logística', 'RF': 'Floresta aleatória', 'SVM': 'SVM linear'}
COLOR = {'LR': '#2a78d6', 'RF': '#eb6834', 'SVM': '#1baf7a'}   # slots 1-3 da paleta de referência
MARK = {'LR': 'o', 'RF': 's', 'SVM': '^'}                        # forma redundante à cor
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SCEN = {'classes': 'Classes (10 projetos)', 'commits': 'Commits (13 famílias)'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})

def fig_delta():
    d = pd.read_csv(A / 'qp2_differences_by_unit.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    rng = np.random.default_rng(0)
    for ax, s in zip(axes, ['classes', 'commits']):
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
        for i, m in enumerate(MODELS):
            v = d[(d.scenario == s) & (d.model == m)].d_mcc.to_numpy()
            if len(v) == 0:
                continue
            y = i + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(v, y, s=26, marker=MARK[m], color=COLOR[m], edgecolor='white', linewidth=0.8, zorder=3)
            med = np.median(v)
            ax.plot([med, med], [i - 0.28, i + 0.28], color=INK, lw=2, zorder=4)
        ax.set_yticks(range(3)); ax.set_yticklabels([LABEL[m] for m in MODELS])
        ax.set_ylim(-0.6, 2.6); ax.invert_yaxis()
        ax.set_title(SCEN[s], fontsize=9.5, color=INK, loc='left')
        ax.set_xlabel('ΔMCC (seleção − todos os atributos)')
        lim = max(0.02, np.nanmax(np.abs(d[d.scenario == s].d_mcc)) * 1.15) if (d.scenario == s).any() else 0.05
        ax.set_xlim(-lim, lim)
    axes[1].tick_params(labelleft=True)
    fig.text(0.99, 0.01, 'Traço preto: mediana. Cada ponto: média das 5 sementes em uma unidade.',
             ha='right', va='bottom', fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / 'fig_delta_mcc.png', dpi=300, facecolor='white'); plt.close(fig)

def fig_freq():
    f = pd.read_csv(A / 'qp3_attribute_frequency.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={'width_ratios': [1, 1]})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#f3f7fd', '#2a78d6', '#0d2f5c'])
    for ax, s, cols in zip(axes, ['classes', 'commits'], [CF, JF]):
        g = f[f.scenario == s].pivot(index='attribute', columns='model', values='frequency').reindex(index=cols, columns=MODELS)
        im = ax.imshow(g.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(3)); ax.set_xticklabels(MODELS)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
        for (r, c), v in np.ndenumerate(g.to_numpy()):
            if np.isnan(v):
                continue
            ax.text(c, r, f'{100*v:.0f}', ha='center', va='center', fontsize=7,
                    color='white' if v > 0.55 else INK)
        ax.set_title(SCEN[s].split(' (')[0], fontsize=9.5, color=INK, loc='left')
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True); ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)
        ax.grid(which='minor', color='white', lw=2); ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('% das execuções em que o atributo foi selecionado', fontsize=8)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(['0', '50', '100']); cb.outline.set_visible(False)
    fig.savefig(FIG / 'fig_attribute_frequency.png', dpi=300, facecolor='white', bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    fig_delta(); fig_freq(); print('figuras em', FIG)
