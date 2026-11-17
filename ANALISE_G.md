# Análise a posteriori (g): commits sem o identificador `apache/hadoop`

## Situação desta análise

- **Não foi registrada antes da execução.** Diferentemente da análise (f) (ver `REGISTRO_ANALISE_F.md`), a análise
  (g) foi executada primeiro, em caráter exploratório, e só depois se decidiu incluí-la no artigo, com os
  resultados já conhecidos.
- A remoção de dados que ela examina foi decidida depois da execução principal, com os resultados do teste já
  conhecidos: o problema do identificador tinha aparecido na análise (c).
- Ela não substitui a análise principal, que continua sendo a do protocolo congelado. No artigo, é apresentada como
  análise a posteriori e exploratória (seções 3.8 e 4.7).

Para que possa ser conferida, este repositório arquiva o código, os resultados da execução original, sem nenhuma
alteração, e o resultado de uma reexecução de conferência.

## Motivação

O identificador `apache/hadoop` não tem nenhum commit rotulado como indutor de defeito: 0 positivos em 6.543
commits de desenvolvimento e em 5.421 de teste (18,0% do teste). A análise (c), em `analyze_posthoc.py`, retirou o
identificador apenas do teste, recalculando as métricas com as previsões já gravadas. A análise (g) retira-o também
do desenvolvimento e refaz todo o cenário de commits: busca, reajuste e avaliação no teste.

## O que foi executado

- As 15 buscas do cenário de commits: 3 classificadores (regressão logística, floresta aleatória e SVM linear) × 5
  sementes (101, 202, 303, 404 e 505).
- Código: `run_experiment.py`, sem alterações desde a versão 1.0.0, chamado por `run_nohadoop.py`.
- Única diferença em relação ao experimento principal: depois do carregamento e das verificações de integridade dos
  dados, os registros com `project == 'apache/hadoop'` são retirados do desenvolvimento e do teste, antes de
  qualquer partição.
- Tudo o mais é igual: atributos, janelas internas por ano (validação em 2014, 2015 e 2016), configuração dos
  classificadores, MCC médio de validação como critério, regra de desempate, orçamento de 100 candidatos novos por
  trajetória, três inicializações por busca, reajuste no desenvolvimento e avaliação única no teste externo.

Consequências do filtro (`results/analysis/nohadoop_datasets.csv`):

| Conjunto | Experimento principal | Sem `apache/hadoop` |
|---|---|---|
| Desenvolvimento (até 2016) | 76.563 registros, 29,28% positivos | 70.020 registros, 32,02% positivos |
| Teste (2017 a 2019) | 30.111 registros, 19,32% positivos | 24.690 registros, 23,56% positivos |

Como mudam o desenvolvimento e sua prevalência, mudam também os pesos de classe e as três validações internas. Por
isso, a comparação com o experimento principal mistura o efeito do identificador com o dessas mudanças.

## Execução original

- **Data:** 16/11/2026, das 0h30 às 2h53 (horário de Brasília), no mesmo processo da execução exploratória das buscas
  lineares da análise (f).
- **Código:** um script auxiliar, `run_extra.py`, mantido fora do repositório, que aplicava o mesmo filtro, com a
  mesma conferência de 24.690 registros de teste, às mesmas 15 tarefas e com o mesmo orçamento.
  `run_nohadoop.py` é a versão arquivada desse procedimento.
- **Ambiente:** nuvem, processadores Intel Xeon virtualizados, duas CPUs lógicas, dois processos com uma thread cada;
  Python 3.13.18, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.7.2 e SciPy 1.16.3, as mesmas versões registradas para
  a análise (f). A execução original não gravou um `environment.json`.
- **Conferência do ambiente:** imediatamente antes, no mesmo ambiente, quatro tarefas do experimento principal
  (classes: Ant com regressão logística, semente 101, e Xerces com SVM, semente 505; commits: regressão logística,
  semente 101, e SVM, semente 303) foram refeitas e coincidiram com as originais, exceto nos tempos.
- **Resultados:** `results/nohadoop/` (tarefas, cache das avaliações e previsões), copiados sem alteração; só a
  subpasta `conferencia/` foi acrescentada depois.

## Reexecução de conferência

Em 16/11/2026, das 21h24 às 22h10, `run_nohadoop.py` refez, em outra pasta e no mesmo ambiente, as dez buscas dos
modelos lineares e a floresta aleatória da semente 101. `compare_tasks.py` comparou cada tarefa com a execução
original: os JSON são idênticos, exceto nos campos de tempo (rastro da busca, máscaras, MCC de validação, trajetórias e
métricas no teste), e os arrays das previsões no teste também são idênticos. As outras quatro florestas não foram
refeitas, pelo custo (cerca de 45 minutos cada).

```
commits__temporal__LR__101.json: idêntica (exceto tempos)
commits__temporal__LR__202.json: idêntica (exceto tempos)
commits__temporal__LR__303.json: idêntica (exceto tempos)
commits__temporal__LR__404.json: idêntica (exceto tempos)
commits__temporal__LR__505.json: idêntica (exceto tempos)
commits__temporal__RF__101.json: idêntica (exceto tempos)
commits__temporal__SVM__101.json: idêntica (exceto tempos)
commits__temporal__SVM__202.json: idêntica (exceto tempos)
commits__temporal__SVM__303.json: idêntica (exceto tempos)
commits__temporal__SVM__404.json: idêntica (exceto tempos)
commits__temporal__SVM__505.json: idêntica (exceto tempos)
11 tarefas comparadas; 0 com diferença
```

O ambiente da reexecução está em `results/nohadoop/conferencia/environment.json`, e a saída acima, em
`results/nohadoop/conferencia/comparacao.txt`.

## Resultados

Calculados por `analyze_nohadoop.py` (`results/analysis/nohadoop_*.csv`), com as mesmas regras de `analyze.py` e
`analyze_posthoc.py`, e reportados no artigo (seção 4.7):

| Modelo | MCC no teste completo | `apache/hadoop` fora do teste (c) | Retreinado sem `apache/hadoop` (g) |
|---|---|---|---|
| Regressão logística | 0,307 → 0,323 | 0,355 → 0,368 | 0,371 → 0,384 |
| Floresta aleatória | 0,361 → 0,362 | 0,415 → 0,419 | 0,463 → 0,458 |
| SVM linear | 0,298 → 0,308 | 0,345 → 0,348 | 0,355 → 0,356 |

MCC com todos os atributos → com seleção, média das cinco sementes.

- Teste de Wilcoxon pareado unilateral para ganho nas 13 famílias, com correção de Holm sobre os três
  classificadores: `p` ajustado de 0,366 (regressão logística) e 1,000 (floresta e SVM). Estimativa de
  Hodges-Lehmann: 0,018 [−0,011; 0,039], −0,002 [−0,011; 0,008] e −0,007 [−0,045; 0,024].
- As 45 trajetórias terminaram em ótimo local, com no máximo sete passos; nenhum ajuste emitiu aviso.
- Subconjuntos: a regressão logística escolheu os mesmos oito atributos nas cinco sementes (os mesmos do experimento
  principal); a SVM, os mesmos cinco (`la`, `ld`, `ns`, `nuc` e `aexp`); a floresta, cinco subconjuntos diferentes,
  de 8 a 11 atributos.
- Ganho médio na validação interna: 0,040, 0,006 e 0,039 (regressão logística, floresta e SVM); no teste: 0,014,
  −0,004 e 0,002.
- Tempo de busca somado: 0,04 h na regressão logística, 3,80 h na floresta aleatória e 0,05 h na SVM linear.

## Como refazer

Sem `--out`, `run_nohadoop.py` usa `results/nohadoop/` e reaproveita as tarefas já gravadas, sem recalcular nada. Para
refazer a análise do zero e compará-la com a arquivada:

```bash
python run_nohadoop.py --out /tmp/g --workers 2   # cerca de 2 horas
python compare_tasks.py results/nohadoop /tmp/g
python analyze_nohadoop.py                        # tabelas de results/analysis/nohadoop_*.csv
```

Para uma conferência rápida, use `--model LR` ou `--model SVM` (cerca de 3 minutos cada).
