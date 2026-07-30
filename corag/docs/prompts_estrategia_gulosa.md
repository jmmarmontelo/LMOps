# Prompts da estratégia gulosa (greedy) no CoRAG

Este documento detalha, prompt a prompt, o que é enviado ao LLM durante a execução da
estratégia **greedy** (`CoRagAgent.sample_path`, `src/agent/corag_agent.py:39-96`) e da
geração da resposta final (`CoRagAgent.generate_final_answer`,
`src/agent/corag_agent.py:98-112`), como executado por `src/reproducao/teste.py`.
Complementa `docs/estrategias_inferencia.md` (que descreve o fluxo geral das 3
estratégias) focando especificamente na **composição dos prompts**.

## Visão geral do loop guloso

A cada iteração do `while` em `sample_path` (`corag_agent.py:54-89`), até 2 chamadas ao
LLM são feitas: uma para gerar a **subquery**, outra para gerar a **subresposta** (com
uma busca de retrieval entre as duas). Isso se repete até `max_path_length` passos serem
atingidos. No final de todo o loop, uma terceira chamada — separada, feita depois que
`sample_path` retorna — gera a **resposta final**.

```
para cada hop (até max_path_length):
    [Prompt 1] gerar subquery  ──►  LLM
                                     │
                              busca E5 (top-k documentos)
                                     │
    [Prompt 2] gerar subresposta ──►  LLM  (usa os docs recuperados)

(depois do loop, uma vez por pergunta)
    [Prompt 3] gerar resposta final ──►  LLM  (usa subqueries+subrespostas + docs do dataset)
```

---

## Prompt 1 — Geração da subquery

Função: `get_generate_subquery_prompt` (`src/prompts.py:4-28`), chamada em
`sample_path` (`corag_agent.py:56-61`).

**Template exato:**
```
You are using a search engine to answer the main query by iteratively searching the web. Given the following intermediate queries and answers, generate a new simple follow-up question that can help answer the main query. You may rephrase or decompose the main query when previous answers are not helpful. Ask simple follow-up questions only as the search engine may not understand complex questions.

## Previous intermediate queries and answers
{past}

## Task description
{task_desc}

## Main query to answer
{query}

Respond with a simple follow-up question that will help answer the main query, do not explain yourself or output anything else.
```

**De onde vem cada variável:**
- `{query}` — a pergunta original do exemplo (`exemplo["query"]`).
- `{task_desc}` — string fixa `"answer multi-hop questions"`, adicionada artificialmente
  em `criar_dataset_benchmark` (`teste.py:41`) — não faz parte do dataset original.
- `{past}` — concatenação de todas as subqueries/subrespostas já geradas nos hops
  anteriores, no formato:
  ```
  Intermediate query 1: <subquery 1>
  Intermediate answer 1: <subanswer 1>
  Intermediate query 2: <subquery 2>
  Intermediate answer 2: <subanswer 2>
  ```
  Se ainda não há histórico (primeiro hop), vira o literal `"Nothing yet"`.

**Parâmetros da chamada** (`corag_agent.py:64`): `temperature=0.` (greedy, vindo de
`teste.py:110`: `temperature=0. if amostra == 0 else temperature`), `max_tokens=64`
(`teste.py:111`).

**Pós-processamento da resposta**: `_normalize_subquery` (`corag_agent.py:19-26`) remove
aspas envolventes e o prefixo `"Intermediate query N: "` caso o modelo o repita por engano.

**Deduplicação**: se a subquery normalizada já estiver em `past_subqueries`, ela é
descartada, a temperatura é elevada para `max(subquery_temp, 0.7)` e o loop tenta gerar
outra (até um teto de `4 × max_path_length` tentativas no total, ver
`docs/estrategias_inferencia.md`).

**Exemplo real** (do log da sessão, pergunta sobre o gênero *Pinanga*):
```
Main query to answer: What are some examples of plants from the Pinanga genus?
→ subquery gerada (hop 2): "What are some examples of plants from the Alopecurus genus?"
```

---

## Retrieval intermediário (entre os prompts 1 e 2)

Não é um prompt ao LLM, mas é o que alimenta o Prompt 2. Em
`_get_subanswer_and_doc_ids` (`corag_agent.py:128-142`):
```python
retriever_results = search_by_http(query=subquery)          # busca no índice E5 (mini-corpus)
doc_ids = [res['doc_id'] for res in retriever_results]        # top-k (TOP_K do servidor, default 5)
documents = [format_input_context(self.corpus[int(doc_id)]) for doc_id in doc_ids][::-1]
```
- `format_input_context` (`data_utils.py:32-38`) formata cada documento como
  `"{title}\n{contents}"` (removendo repetição do título no início do texto).
- A ordem é **invertida** (`[::-1]`): assumindo que a busca retorna do mais para o menos
  relevante, a inversão põe o mais relevante **por último** — mais perto da pergunta no
  prompt seguinte.

---

## Prompt 2 — Geração da subresposta

Função: `get_generate_intermediate_answer_prompt` (`src/prompts.py:31-49`), chamada em
`_get_subanswer_and_doc_ids` (`corag_agent.py:135-138`).

**Template exato:**
```
Given the following documents, generate an appropriate answer for the query. DO NOT hallucinate any information, only use the provided documents to generate the answer. Respond "No relevant information found" if the documents do not contain useful information.

## Documents
{context}

## Query
{subquery}

Respond with a concise answer only, do not explain yourself or output anything else.
```

**De onde vem cada variável:**
- `{context}` — os documentos recuperados na etapa anterior (já invertidos), concatenados
  com uma linha em branco entre cada um.
- `{subquery}` — a subquery recém-gerada no Prompt 1 (não a pergunta original).

**Parâmetros da chamada** (`corag_agent.py:141`): `temperature=0.` (fixo, **sempre**
determinístico, independente da estratégia/temperatura da subquery), `max_tokens=128`.

**Truncamento**: `self._truncate_long_messages(messages, max_length=max_message_length)`
— como `teste.py:106-112` não passa `max_message_length` explicitamente, usa o default de
`sample_path`: **4096**. A truncagem só age se o conteúdo da mensagem passar de
`2 × 4096 = 8192` caracteres (não tokens), cortando pelo meio do texto
(`truncate_from_middle=True`, via `batch_truncate`). Não há limite individual por
documento aqui (diferente do Prompt 3, ver abaixo).

**Exemplo real** (mesmo log):
```
Query: What are some examples of plants from the Pinanga genus found in temperate regions?
→ subanswer: "No relevant information found."
```
(esse caso ilustra a instrução explícita do prompt para admitir falta de informação em
vez de alucinar — o mini-corpus provavelmente não continha um documento de *Pinanga* em
regiões temperadas dentre os poucos milhares indexados.)

---

## Condição de parada do loop

O `while` em `sample_path` (`corag_agent.py:54`) para quando:
- `len(past_subqueries) >= max_path_length` (**3**, valor usado por `teste.py:109`), ou
- `num_llm_calls >= max_num_llm_calls` (`4 × max_path_length` = **12**, teto de
  segurança contra loops de subqueries repetidas).

Não há critério de parada antecipada por "resposta já suficiente" nessa estratégia — ela
sempre tenta completar os `max_path_length` hops (a menos que bata no teto de tentativas).

---

## Prompt 3 — Geração da resposta final

Função: `get_generate_final_answer_prompt` (`src/prompts.py:52-88`), chamada em
`CoRagAgent.generate_final_answer` (`corag_agent.py:98-112`), a partir de
`teste.py:113-120` (uma vez por candidato de resposta, depois que o `RagPath` do loop
guloso já está pronto).

**Template exato:**
```
Given the following intermediate queries and answers, generate a final answer for the main query by combining relevant information. Note that intermediate answers are generated by an LLM and may not always be accurate.

## Documents
{context}

## Intermediate queries and answers
{past}

## Task description
{task_desc}

## Main query
{query}

Respond with an appropriate answer only, do not explain yourself or output anything else.
```

**De onde vem cada variável:**
- `{past}` — **todas** as subqueries/subrespostas acumuladas nos `max_path_length` hops
  do loop guloso (mesmo formato do Prompt 1).
- `{task_desc}` e `{query}` — os mesmos da pergunta original.
- `{context}` — **não são os documentos recuperados durante o loop!** Vêm de
  `format_documents_for_final_answer` (`data_utils.py:76-90`), chamado uma vez por
  pergunta em `teste.py:95-100`:
  ```python
  args = SimpleNamespace(num_contexts=5, max_len=3072, context_placement="backward")
  documentos = format_documents_for_final_answer(
      args=args, context_doc_ids=exemplo["context_doc_ids"], tokenizer=..., corpus=corpus,
  )
  ```
  - Usa só os **5 primeiros** (`num_contexts=5`) dos 100 `context_doc_ids` do exemplo do
    dataset (remapeados pros ids locais do mini-corpus na nossa construção) — um conjunto
    **fixo por pergunta**, independente do que foi buscado durante os hops.
  - Cada um dos 5 documentos é truncado individualmente para
    `int(3072/5 × 1.2) ≈ 737 tokens` (por tokenizer real, via `batch_truncate`).
  - A ordem dos 5 é invertida (`context_placement="backward"`) — mesma lógica de "mais
    relevante por último".
  - Esse mesmo conjunto de 5 documentos é reaproveitado **igual para todos os N
    caminhos**, no caso de best-of-n — só o histórico de subqueries/subrespostas muda
    entre os candidatos.

**Parâmetros da chamada** (`teste.py:113-120`): `temperature=0.` (sempre determinística),
`max_tokens=128`, `max_message_length=3072`.

---

## Resumo comparativo

| | Prompt 1 (subquery) | Prompt 2 (subanswer) | Prompt 3 (resposta final) |
|---|---|---|---|
| Função | `get_generate_subquery_prompt` | `get_generate_intermediate_answer_prompt` | `get_generate_final_answer_prompt` |
| Frequência | 1× por hop | 1× por hop (se subquery não repetida) | 1× por pergunta (× N no best-of-n) |
| Fonte dos documentos | — (sem docs) | Busca E5 ao vivo (top-k do mini-índice) | `context_doc_ids` fixos do dataset (top 5) |
| Ordem dos docs | — | Invertida (`[::-1]`) | Invertida (`context_placement="backward"`) |
| Truncamento | — | ~8192 caracteres no total (solto) | ~737 tokens por documento (rígido) |
| Temperatura | 0. (greedy) / 0.7 (retry ou best-of-n) | sempre 0. | sempre 0. |
| `max_tokens` | 64 | 128 | 128 |

## Parâmetros efetivamente usados nesta sessão (`teste.py`)

- `max_path_length=3`
- `task_desc="answer multi-hop questions"` (fixo, sintético)
- `num_contexts=5`, `max_len=3072`, `context_placement="backward"` (resposta final)
- `TOP_K` do servidor E5 (retrieval por hop): default 5, conforme
  `start_e5_server_main.py`
- Mini-corpus (`data/mini/`): índice reduzido construído por
  `construir_mini_corpus.py`, contendo os documentos de contexto das 30 perguntas de
  cada benchmark + ~1000 distratores aleatórios por benchmark
