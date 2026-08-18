# Experimentos: Fonte dos Documentos no Prompt de Resposta Final

## Visão geral

Quatro execuções sobre o mesmo mini dataset de validação do 2WikiMultihopQA (100 perguntas,
`data/mini_2wiki100`), combinando duas estratégias de decodificação (`best_of_n`, `dynamic_chain`)
com duas fontes de documentos para o prompt de resposta final (`dataset`, `chain`).

| | Fonte = dataset | Fonte = chain |
|---|---|---|
| **best_of_n** | Execução 1 | Execução 3 |
| **dynamic_chain** | Execução 2 | Execução 4 |

Parâmetros comuns às quatro execuções: `n=4` (número de cadeias candidatas amostradas por
pergunta), `max_path_length (L)=6` (tamanho máximo de cada cadeia), `num_contexts=5` (documentos
selecionados para o prompt final).

## Execução 1 — `best_of_n`, fonte = dataset

**Fluxo:**
1. Para cada pergunta, amostram-se `n=4` cadeias candidatas via `sample_path` — a 1ª com
   `temperature=0` (greedy), as demais com `temperature=0.7`. Cada cadeia gera até `L=6` hops
   (subquery → retrieval → subanswer).
2. Escolhe-se **uma única** cadeia entre as 4, via `selecionar_por_penalizacao`: conta, em cada
   candidata, quantos hops responderam "no relevant information found" e escolhe a que tiver
   menos (empate resolvido pela 1ª amostra, a greedy).
3. **Montagem do prompt final:** os documentos vêm de `context_doc_ids`, campo já presente no
   dataset `corag/multihopqa` — um retrieval feito **uma única vez**, antes de qualquer execução,
   com a pergunta original (não com as subqueries). Corta-se nos 5 primeiros desses ids.
4. O prompt final é montado com 4 blocos: `Documents` (os 5 documentos estáticos), `Intermediate
   queries and answers` (só da cadeia escolhida no passo 2), `Task description`, `Main query`.

## Execução 2 — `dynamic_chain`, fonte = dataset, 5 documentos

**Fluxo:**
1. Mesma amostragem de `n=4` cadeias candidatas do passo 1 acima.
2. **Sem seleção de uma única cadeia** — em vez disso, funde os hops úteis (que não retornaram
   "no relevant information found") de **todas as 4 cadeias** numa única chain mesclada
   (`montar_chain_final`).
3. Documentos do prompt final: mesma fonte estática da Execução 1 — `context_doc_ids` do dataset,
   5 primeiros.
4. Prompt final: `Documents` (5 documentos estáticos, idênticos aos da Execução 1) +
   `Intermediate queries and answers` (todos os hops úteis mesclados das 4 cadeias) + `Task
   description` + `Main query`.

## Execução 3 — `best_of_n`, fonte = chain

**Fluxo:**
1. Mesma amostragem e seleção da Execução 1 (`n=4` candidatas, escolhe 1 via
   `selecionar_por_penalizacao`).
2. **Diferença:** os documentos do prompt final não vêm do dataset — vêm dos `doc_ids` que a
   própria cadeia escolhida recuperou em cada um dos seus hops, fundidos por **Reciprocal Rank
   Fusion (RRF)** numa lista única ordenada por relevância agregada, cortada nos 5 primeiros.
3. Prompt final: `Documents` (5 documentos vindos da busca feita pela própria chain) +
   `Intermediate queries and answers` (cadeia escolhida) + `Task description` + `Main query`.

## Execução 4 — `dynamic_chain`, fonte = chain

**Fluxo:**
1. Mesma amostragem e fusão de cadeias da Execução 2 (`n=4` candidatas, funde os hops úteis de
   todas).
2. **Diferença:** os documentos do prompt final vêm da fusão RRF dos `doc_ids` de todos os hops
   úteis da chain mesclada, cortados nos 5 primeiros.
3. Prompt final: `Documents` (5 documentos da fusão RRF) + `Intermediate queries and answers`
   (mesclados das 4 cadeias) + `Task description` + `Main query`.

## Diferença entre fonte = dataset e fonte = chain

- **`dataset`**: os documentos são um retrieval **estático e único**, feito com a pergunta
  original, pré-computado e salvo no dataset `corag/multihopqa` antes de qualquer execução. Não
  tem relação nenhuma com as subqueries geradas durante a chain — é o mesmo conjunto de 100
  candidatos para qualquer estratégia, qualquer execução, qualquer subquery gerada.
- **`chain`**: os documentos vêm da busca (`search_by_http`) feita durante os próprios hops da
  chain, com as subqueries geradas naquela execução específica. Como cada hop retorna uma lista
  já ordenada por score do retriever, a fusão por RRF combina essas listas somando `1/(k+rank)`
  por documento, favorecendo documentos que aparecem bem rankeados em múltiplos hops. Reflete o
  que o modelo efetivamente encontrou ao decompor a pergunta, em vez de um retrieval feito antes
  de qualquer decomposição.