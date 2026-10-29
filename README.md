# Seleção de atributos por Hill Climbing na predição de defeitos de software

Código e resultados do Trabalho de Conclusão de Curso de **José Lucas Silva Souza**
(Engenharia de Software, Instituto de Ensino Superior iCEV), orientado pelo Prof. Me. Mauro José Araújo de Melo.

O estudo avalia se a seleção de atributos por Hill Climbing preserva ou melhora o desempenho de classificadores
de defeitos em duas granularidades:

- **classe**: 10 projetos Java da coleção de Jureczko e Madeyski, avaliados entre projetos (leave-one-project-out);
- **commit**: 106.674 commits do ApacheJIT, avaliados por período (desenvolvimento até 2016, teste de 2017 a 2019).

Foram comparados regressão logística, floresta aleatória e SVM linear, com todos os atributos e com o subconjunto
escolhido pelo Hill Climbing (MCC médio em três validações internas; teste externo isolado das decisões).
São 165 buscas (5 sementes) e 330 avaliações externas.

## Resultado principal

A seleção reduziu, em média, de 22% a 50% dos atributos, mas nenhuma das seis comparações entre seleção e linha de
base mostrou ganho significativo após a correção de Holm. Na floresta aleatória com classes, a seleção piorou o MCC
em sete dos dez projetos. Os ganhos obtidos nas validações internas foram maiores que os observados no teste.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `run_experiment.py` | Experimento completo. Grava um JSON por tarefa e um cache por avaliação; pode ser interrompido e retomado. |
| `analyze.py` | Análise das questões de pesquisa: QP1, QP2 (Wilcoxon pareado e Holm) e QP3, além da análise de sensibilidade. |
| `figures.py` | Figuras do artigo. |
| `download_data.py`, `data_manifest.json` | Obtenção dos dados nas fontes originais, com conferência de SHA-256. |
| `results/tasks/` | Um JSON por (cenário, rodada, modelo, semente): subconjunto escolhido, trajetórias, rastro da busca e métricas externas. |
| `results/eval_cache/` | Cada avaliação de candidato: MCC em cada divisão interna e tempo. |
| `results/predictions/` | Previsões e escores no teste externo, com e sem seleção. |
| `results/analysis/` | Tabelas usadas no artigo. |
| `results/figures/` | Figuras. |
| `results/partitions.csv` | Partições internas e externas. |
| `results/environment.json`, `results/run.log` | Ambiente e registro da execução. |

## Como reproduzir

Requer Python 3.12 ou 3.13.

```bash
pip install -r requirements.txt
python download_data.py          # baixa os dados e confere os hashes
python run_experiment.py --workers 2   # cerca de 12 horas de CPU; reaproveita o que já estiver em results/
python analyze.py
python figures.py
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
