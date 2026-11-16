"""Figuras do artigo (PNG 300 dpi, fundo branco, legíveis em impressão em tons de cinza).

Figura A: ΔMCC (seleção − todos os atributos) por unidade de análise, por modelo e cenário
          (os dois painéis usam a mesma escala no eixo x).
Figura B: frequência de seleção dos atributos por modelo e cenário.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

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
    # Mesma escala nos dois painéis (sharex): a dispersão das classes e a dos commits ficam comparáveis.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    lim = max(0.02, float(np.nanmax(np.abs(d.d_mcc))) * 1.15)
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
        ax.set_title(SCEN[s], fontsize=9.5, color=INK, loc='left')
        ax.set_xlim(-lim, lim)
        ax.set_xticks([-0.10, -0.05, 0, 0.05, 0.10])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f'{v:.2f}'.replace('.', ',').replace('-', '−')))
    axes[0].set_yticks(range(3)); axes[0].set_yticklabels([LABEL[m] for m in MODELS])
    axes[0].set_ylim(-0.6, 2.6); axes[0].invert_yaxis()
    fig.supxlabel('ΔMCC (seleção − todos os atributos)', fontsize=9, color=INK, y=0.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=2.0)
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

def fig_protocol():
    """Figura 1: fluxo do protocolo nos dois cenários (desenho, sem dados)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    def box(x, y, w, h, lines, title=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc='#F3F3F3', ec=INK2, lw=0.8))
        step = 5.4 if title else 4.9
        y0 = y + h / 2 + step * (len(lines) - 1) / 2
        for i, t in enumerate(lines):
            ax.text(x + w / 2, y0 - i * step, t, ha='center', va='center', color=INK,
                    fontsize=9.5 if title else 8.6)

    def arrow(x, y1, y2):
        ax.annotate('', xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.9, mutation_scale=9))

    xl, xr, w = 2, 52.5, 45.5
    box(xl, 86, w, 12.5, ['CLASSES', '10 projetos • uma versão por projeto'], title=True)
    box(xr, 86, w, 12.5, ['COMMITS', 'Estratos publicados do ApacheJIT'], title=True)
    box(xl, 69, w, 11.5, ['Rodada externa: 9 projetos para desenvolver', '1 projeto reservado para teste'])
    box(xr, 69, w, 11.5, ['Desenvolvimento: year até 2016', 'Teste reservado: year de 2017 a 2019'])
    box(xl, 52, w, 11.5, ['Validação interna: GroupKFold com 3 divisões', 'Projetos de treino e validação separados'])
    box(xr, 52, w, 11.5, ['3 janelas internas com treino crescente', 'Validações nos estratos 2014, 2015 e 2016'])
    xc, wc = 17.5, 65
    box(xc, 30, wc, 16.5, ['Linha de base completa e seleção por Hill Climbing', 'Transformações ajustadas em cada treino interno',
                          'Busca orientada somente pela validação'])
    box(xc, 17, wc, 7.5, ['Reajuste no desenvolvimento com a máscara escolhida'])
    box(xc, 1, wc, 10.5, ['Avaliação externa uma vez por configuração e semente', 'F1 • AUC-ROC • MCC • número de atributos • tempo'])
    for x in (xl + w / 2, xr + w / 2):
        arrow(x, 86, 80.5); arrow(x, 69, 63.5); arrow(x, 52, 46.5)
    arrow(50, 30, 24.5); arrow(50, 17, 11.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(FIG / 'fig_protocol.png', dpi=300, facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    fig_protocol(); fig_delta(); fig_freq(); print('figuras em', FIG)
