# Registro da análise a posteriori (f): Hill Climbing nas classes até a convergência

Registrado em 16/11/2026, antes da execução descrita abaixo. Este arquivo fixa o que será executado e o que
será reportado, para que o resultado entre no artigo seja qual for.

## Motivação

No experimento principal, o orçamento de 100 candidatos novos por trajetória encerrou 339 das 450 trajetórias do
cenário de classes antes de um ótimo local (138 na regressão logística, 136 na SVM linear e 65 na floresta
aleatória), e nenhuma passou de cinco passos. A análise (f) verifica se a ausência de ganho e a redução de
atributos observadas nesse cenário dependem dessa interrupção.

## O que será executado

- As 150 buscas do cenário de classes: 10 rodadas (um projeto de teste por vez) × 3 classificadores (regressão
  logística, floresta aleatória e SVM linear) × 5 sementes (101, 202, 303, 404 e 505).
- Código: `run_experiment.py`, sem alterações, chamado por `run_convergence.py`.
- Única diferença em relação ao experimento principal: não há limite de candidatos por trajetória. Cada
  trajetória termina apenas em um ótimo local, isto é, quando nenhum vizinho supera a solução atual por mais
  de 1e-12.
- Tudo o mais é igual: dados, partições externas e internas (GroupKFold com três divisões), as três
  inicializações por busca (conjunto completo e as duas máscaras aleatórias da semente), configuração dos
  classificadores, MCC médio de validação como critério, regra de desempate, reajuste no desenvolvimento e
  avaliação única no teste externo.
- O cenário de commits não é reexecutado: nele, as 45 trajetórias do experimento principal já terminaram em
  ótimo local.

## O que será reportado

Calculado por `analyze_convergence.py`, com as mesmas regras de `analyze.py` e `analyze_posthoc.py`:

1. ΔMCC por projeto (seleção convergida menos linha de base, média das cinco sementes), com número de projetos
   com diferença positiva e negativa, mediana e intervalo interquartil.
2. Teste de Wilcoxon pareado unilateral para ganho, com correção de Holm sobre os três classificadores;
   correlação bisserial de postos pareada; estimativa de Hodges-Lehmann com intervalo de confiança de 95%.
3. Ganho interno (MCC médio de validação da máscara escolhida menos o do conjunto completo) e ganho no teste.
4. Passos por trajetória e número de atributos finais; tamanho e origem (inicialização completa ou aleatória)
   da máscara escolhida; frequência de seleção por atributo.
5. Diferença entre a seleção convergida e a seleção do experimento principal, por projeto.
6. Conferência: cada avaliação do experimento principal deve reaparecer com o mesmo valor, e as trajetórias
   que já convergiam devem terminar no mesmo ponto.

Medianas de ΔF1 e ΔAUC-ROC são descritivas, como no experimento principal.

## Papel no artigo

A análise principal continua sendo a do protocolo congelado (orçamento de 100 candidatos por trajetória). A
análise (f) é a posteriori e exploratória, e será apresentada como tal, ao lado das demais análises a
posteriori da seção 3.8.

## Histórico de execução

- **Regressão logística e SVM linear:** essas 100 buscas já foram executadas uma vez em 16/11/2026, antes
  deste registro, em caráter exploratório, e o autor conhecia seus resultados ao escrevê-lo. Elas serão
  reexecutadas depois deste commit. Como os dois modelos e a busca são determinísticos, espera-se o mesmo
  resultado; qualquer diferença será relatada.
- **Floresta aleatória:** as 50 buscas não foram executadas antes deste registro.

## Compromisso

Os resultados das 150 buscas serão reportados no artigo e arquivados neste repositório (em
`results/convergence/` e `results/analysis/convergence_*.csv`), quaisquer que sejam.
