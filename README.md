# Seleção de atributos por Hill Climbing na predição de defeitos de software

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24295861.svg)](https://doi.org/10.5281/zenodo.24295861)

Código e resultados do Trabalho de Conclusão de Curso de **José Lucas Silva Souza**
(Engenharia de Software, Instituto de Ensino Superior iCEV), orientado pelo Prof. Me. Mauro José Araújo de Melo.

O estudo avalia o efeito da seleção de atributos por Hill Climbing sobre o desempenho de classificadores de
defeitos em duas granularidades:

- **classe**: 10 projetos Java da coleção de Jureczko e Madeyski, avaliados entre projetos (leave-one-project-out);
- **commit**: 106.674 commits do ApacheJIT, avaliados por período (desenvolvimento até 2016, teste de 2017 a 2019).

Foram comparados regressão logística, floresta aleatória e SVM linear, com todos os atributos e com o subconjunto
escolhido pelo Hill Climbing (MCC médio em três validações internas; teste externo isolado das decisões).
São 165 buscas (5 sementes) e 330 avaliações externas.

## Resultado principal

A seleção reduziu, em média, de 22% a 50% dos atributos, mas nenhuma das seis comparações entre seleção e linha de
base mostrou ganho significativo após a correção de Holm. Na floresta aleatória com classes, a seleção piorou o MCC
em sete dos dez projetos. Os ganhos obtidos nas validações internas foram maiores que os observados no teste. Um filtro
univariado e uma seleção embutida L1, avaliados sob o mesmo protocolo, também não produziram ganho significativo.

Na análise a posteriori (f), registrada em `REGISTRO_ANALISE_F.md` antes da execução, as 150 buscas do cenário de
classes foram refeitas sem limite de orçamento, até o ótimo local. Nenhum dos três classificadores passou a ter ganho:
o ganho na validação interna aumentou e o ganho no teste diminuiu nos três, e, na floresta aleatória, o intervalo de
confiança da diferença de MCC ficou inteiramente abaixo de zero.

Na análise a posteriori (g), descrita em `ANALISE_G.md`, o cenário de commits foi refeito sem o identificador
`apache/hadoop`, que não tem nenhum commit rotulado como indutor de defeito, retirado do desenvolvimento e do teste.
Essa análise não foi registrada antes da execução. O MCC com todos os atributos subiu para 0,355 a 0,463, e a seleção
continuou sem ganho significativo nos três classificadores.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `run_experiment.py` | Experimento completo. Grava um JSON por tarefa e um cache por avaliação; pode ser interrompido e retomado. |
| `analyze.py` | Análise das questões de pesquisa: QP1, QP2 (Wilcoxon pareado e Holm) e QP3, além da análise de sensibilidade. |
| `figures.py` | Figuras do artigo (Figura 1, fluxo do protocolo; Figuras 2 e 3, resultados). |
| `run_alternatives.py` | Experimento complementar: filtro univariado (teste F) e seleção embutida (regressão logística L1), sob o mesmo protocolo. |
| `analyze_alternatives.py` | Comparação entre Hill Climbing, filtro e L1. |
| `analyze_posthoc.py` | Análises a posteriori feitas só com os registros gravados: trajetórias e paradas por orçamento, origem da máscara escolhida, ganho interno, Wilcoxon bilateral, estimativa de Hodges-Lehmann com intervalo de confiança e desempenho dos commits sem `apache/hadoop` (saídas em `results/analysis/posthoc_*.csv`). |
| `REGISTRO_ANALISE_F.md` | Registro da análise a posteriori (f), feito antes da execução, e o relato da execução. |
| `run_convergence.py` | Análise (f): as 150 buscas do cenário de classes sem limite de orçamento (parada apenas em ótimo local). |
| `analyze_convergence.py` | Comparação da análise (f) com a análise principal (saídas em `results/analysis/convergence_*.csv`). |
| `ANALISE_G.md` | Descrição da análise a posteriori (g), com o histórico da execução e a conferência. |
| `run_nohadoop.py` | Análise (g): o cenário de commits sem o identificador `apache/hadoop`, retirado do desenvolvimento e do teste. |
| `analyze_nohadoop.py` | Comparação da análise (g) com a análise principal e com o recálculo (c) (saídas em `results/analysis/nohadoop_*.csv`). |
| `compare_tasks.py` | Compara duas execuções tarefa por tarefa (JSON, exceto tempos, e previsões). |
| `download_data.py`, `data_manifest.json` | Obtenção dos dados nas fontes originais, com conferência de SHA-256. |
| `results/tasks/` | Um JSON por (cenário, rodada, modelo, semente): subconjunto escolhido, trajetórias, rastro da busca e métricas externas. |
| `results/eval_cache/` | Cada avaliação de candidato: MCC em cada divisão interna e tempo. |
| `results/predictions/` | Previsões e escores no teste externo, com e sem seleção. |
| `results/analysis/` | Tabelas usadas no artigo. |
| `results/figures/` | Figuras. |
| `results/partitions.csv` | Partições internas e externas. |
| `results/environment.json`, `results/run.log` | Ambiente e registro da execução. |
| `results/alternatives/` | Resultados do experimento complementar (tarefas, cache das avaliações e análise). |
| `results/convergence/` | Resultados da análise (f): tarefas, cache das avaliações e previsões. |
| `results/nohadoop/` | Resultados da análise (g): tarefas, cache das avaliações e previsões; em `conferencia/`, o ambiente e a comparação da reexecução. |

## Como reproduzir

Requer Python 3.12 ou 3.13.

```bash
pip install -r requirements.txt
python download_data.py          # baixa os dados e confere os hashes
python run_experiment.py --workers 2   # cerca de 12 horas de CPU; reaproveita o que já estiver em results/
python analyze.py
python figures.py
python run_alternatives.py --workers 2   # experimento complementar (cerca de 1 hora)
python analyze_alternatives.py
python analyze_posthoc.py
python run_convergence.py --workers 2   # análise a posteriori (f), cerca de 3 horas
python analyze_convergence.py
python run_nohadoop.py --workers 2   # análise a posteriori (g), cerca de 2 horas
python analyze_nohadoop.py
```

Os resultados da execução original já estão em `results/`. Para recalcular tudo do zero, apague `results/tasks/` e
`results/eval_cache/` antes de rodar `run_experiment.py`.

## Dados

Os arquivos de dados **não** são redistribuídos aqui; `download_data.py` os obtém das fontes:

- Classes: coleção de Jureczko e Madeyski, pelo espelho [klainfo/DefectData](https://github.com/klainfo/DefectData)
  (revisão fixada no manifesto).
- Commits: ApacheJIT v2 (Keshavarz e Nagappan, 2022; [Zenodo 5907847](https://zenodo.org/records/5907847)), pelo
  repositório dos autores [hosseinkshvrz/apachejit](https://github.com/hosseinkshvrz/apachejit). Os arquivos são
  idênticos, por SHA-256, aos do pacote do Zenodo.

Ao usar os dados, cite os autores originais:

- JURECZKO, M.; MADEYSKI, L. Towards identifying software project clusters with regard to defect prediction. PROMISE 2010.
- KESHAVARZ, H.; NAGAPPAN, M. ApacheJIT: a large dataset for just-in-time defect prediction. MSR 2022.

## Como citar

Versão 1.2.0, usada no artigo: [10.5281/zenodo.24295861](https://doi.org/10.5281/zenodo.24295861).
Para citar sem fixar uma versão, use o DOI conceito [10.5281/zenodo.24258042](https://doi.org/10.5281/zenodo.24258042),
que sempre aponta para a versão mais recente.

- SOUZA, J. L. S. Seleção de atributos por Hill Climbing na predição de defeitos de software. Versão 1.2.0. Zenodo,
  2026. DOI: 10.5281/zenodo.24295861.

Versões anteriores: 1.1.0 ([10.5281/zenodo.24295160](https://doi.org/10.5281/zenodo.24295160)), que acrescentou a
análise a posteriori (f), e 1.0.0 ([10.5281/zenodo.24258043](https://doi.org/10.5281/zenodo.24258043)). A versão
1.2.0 acrescentou a análise a posteriori (g).

## Licença

O código deste repositório é distribuído sob a [GNU General Public License v3.0](LICENSE). Copyright (C) 2026
José Lucas Silva Souza. Os dados das bases não fazem parte do repositório e seguem as condições de uso de seus autores.
