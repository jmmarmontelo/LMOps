# `dynamic_chain.py` — como funciona

Script de reprodução do CoRAG com uma estratégia diferente da do `best_of_n.py`: em vez de
escolher **uma** cadeia entre N candidatas amostradas, ele **mescla os hops úteis de todas as N
cadeias** numa única "cadeia dinâmica" — daí o nome. Roda as 4 tasks de `TASK_SPLITS`
(hotpotqa, 2wikimultihopqa, musique, bamboogle), calcula métricas EM/F1 por task.

## Pré-requisitos pra rodar

1. **`.env`** em `src/reproducao/.env` com `BASE_URL`, `API_KEY`, `MODEL_NAME` do backend LLM.
2. **`data/mini/`** presente (corpus, índice E5 e perguntas) — gerado por
   `construir_mini_corpus.py` ou copiado de outra máquina.
3. **Servidor E5**: diferente do `best_of_n.py`, este script **sobe o servidor sozinho**
   (`iniciar_servidor_e5`) se a porta 8090 ainda não estiver ocupada — não precisa subir à parte.

## Visão geral do fluxo

```
__main__
 ├─ iniciar_servidor_e5(...)        → sobe o servidor E5 (mini-índice) se ainda não estiver no ar
 ├─ carregar_mini_corpus()          → corpus local (uma vez só)
 ├─ _montar_agente(...)             → monta o CoRagAgent
 └─ para cada task em TASK_SPLITS:
     ├─ carregar_mini_perguntas(task)
     └─ para cada pergunta:
         └─ executar_pipeline_pergunta(corag_agent, exemplo, n, max_path_length)
             ├─ format_documents_for_final_answer(...)  → top-5 context_doc_ids do exemplo (mesmo de best_of_n.py)
             ├─ executar_dynamic_chain(...)     → amostra N cadeias, SEM escolher uma
             ├─ montar_chain_final(paths)       → concatena as N cadeias + filtra hops sem resposta
             └─ gerar_resposta_final(..., documents=documentos)  → UMA resposta final: cadeia mesclada + documentos
     └─ calcular_metricas(resultados)           → EM/F1 da task
 └─ imprime resumo final de métricas por task
```

## Funções

### Setup / infraestrutura

- **`carregar_mini_corpus()`** — `Dataset.load_from_disk(MINI_CORPUS_DIR)`, carregado uma vez
  fora do loop de tasks. É necessário mesmo com o servidor E5 rodando, porque o E5 só devolve
  `doc_id`/`score` (ver `E5Searcher.batch_search`, `verbose=False` por padrão) — quem traduz
  `doc_id` pra texto de fato é o `CoRagAgent`, usando essa cópia local do corpus.
- **`carregar_mini_perguntas(task)`** — seta `MINI_DATASET_DIR` e delega pra
  `criar_dataset_benchmark` (importada de `best_of_n.py`), que aí carrega direto do disco em vez
  de baixar/amostrar do HF.
- **`_porta_aberta(host, port)`** — checa se uma porta TCP já está aceitando conexões
  (`socket.connect_ex`).
- **`iniciar_servidor_e5(index_dir, corpus_dir, ...)`** — se a porta 8090 já estiver aberta, não
  faz nada (assume que já tem um servidor rodando). Senão, sobe
  `uvicorn src.search.start_e5_server_main:app` como subprocesso, com `INDEX_DIR`/`CORPUS_DIR`
  apontando pro mini-índice/mini-corpus, e espera (poll a cada 2s, timeout 600s) até a porta
  abrir.
- **`_montar_agente(model, api_key, base_url, corpus)`** — monta `VllmClient` +
  `CoRagAgent`, com os mesmos dois contornos do `best_of_n.py`: troca o client OpenAI interno por
  um apontando pra `base_url` completa (https), e substitui `get_vllm_model_id` por um valor fixo
  (`tokenizer_name_or_path`) pra evitar a consulta a `localhost:8000/v1/models` de um vLLM real.
- **`_imprimir_path(path)`** — imprime uma `RagPath` (query + cada hop: subquery, subanswer,
  doc_ids) no console, usado como debug em vários pontos do pipeline.

### Amostragem e montagem da cadeia dinâmica

- **`executar_dynamic_chain(corag_agent, query, task_desc, n, max_path_length)`** — amostra `n`
  cadeias independentes via `corag_agent.sample_path` (1ª amostra greedy/`temperature=0.`, as
  demais com `temperature=0.7` pra diversidade), cada uma com até `max_path_length` hops.
  **Não pontua nem escolhe uma cadeia** — o comentário no código explica por quê: o `best_of_n`
  original pontuaria via logprob de "No relevant information found"
  (`extra_body={"prompt_logprobs": 1}`), recurso exclusivo de vLLM real que o backend usado aqui
  não suporta. Em vez de substituir esse scoring por um critério de escolha única (como fizemos
  em `best_of_n.py` com `selecionar_por_penalizacao`), a estratégia aqui é usar todas as N
  cadeias. Imprime cada candidata (`--- Candidato i/n ---`) e retorna a lista completa.

- **`selecionar_subperguntas_respondidas(path)`** — dado uma cadeia (potencialmente longa, fruto
  da concatenação de várias), descarta um hop **só** se a subresposta for
  `"no relevant information found"` (comparação exata, case-insensitive). Esse é hoje o **único**
  critério de descarte — uma deduplicação por subquery repetida existia antes e foi removida a
  pedido do usuário, então subperguntas iguais respondidas em candidatas diferentes são todas
  mantidas na cadeia final.

- **`montar_chain_final(paths)`** — concatena `past_subqueries`/`past_subanswers`/`past_doc_ids`
  de **todas** as N cadeias amostradas numa `RagPath` única "bruta", e aplica
  `selecionar_subperguntas_respondidas` nela — resultando numa cadeia mesclada só com os hops que
  encontraram informação relevante, de qualquer uma das N candidatas.

### Resposta final

- **`montar_prompt_final(path, task_desc, documents)`** — monta as mensagens de prompt
  (`get_generate_final_answer_prompt`) a partir da cadeia final.
- **`gerar_resposta_final(corag_agent, path, task_desc, documents)`** — chama
  `corag_agent.vllm_client.call_chat(...)` **diretamente** (não passa por
  `CoRagAgent.generate_final_answer`, embora o efeito seja equivalente), com `temperature=0.` e
  `max_tokens=128`.

- **`executar_pipeline_pergunta(corag_agent, exemplo, n, max_path_length)`** — orquestra o
  pipeline pra uma pergunta: primeiro monta `documentos` via `format_documents_for_final_answer`
  (`DOC_ARGS = SimpleNamespace(num_contexts=5, max_len=3072, context_placement="backward")`,
  mesmos parâmetros de `best_of_n.py`, usando `corag_agent.tokenizer`/`corag_agent.corpus`);
  depois `executar_dynamic_chain` → `montar_chain_final` → `gerar_resposta_final(..., documents=documentos)`.
  Isso equaliza a comparação com `best_of_n.py`: as duas estratégias agora recebem os mesmos
  documentos de contexto (`context_doc_ids` do exemplo) na geração da resposta final — só a forma
  de montar a cadeia de subperguntas muda entre elas. Retorna o dicionário de resultado (`query`,
  `answers`, `subqueries`, `subanswers`, `doc_ids`, `prediction`) que alimenta `calcular_metricas`.

### Função não utilizada

- **`executar_teste(corag_agent, query, task_desc)`** — faz um `sample_path` único (greedy, sem
  best_of_n/merge) e imprime o resultado. Não é chamada em nenhum lugar do arquivo nem importada
  em outro módulo — ficou como resquício de uma versão anterior do pipeline.

### `__main__`

```python
n = 2                # nº de cadeias amostradas em executar_dynamic_chain
max_path_length = 6  # L: tamanho máximo de cada cadeia amostrada
```

Sobe o servidor E5, monta o corpus/agente uma vez, itera pelas 4 tasks chamando
`executar_pipeline_pergunta` pergunta a pergunta, e no fim imprime métricas por task + resumo
final.

## Diferenças-chave em relação a `best_of_n.py`

| | `best_of_n.py` | `dynamic_chain.py` |
|---|---|---|
| Estratégia de seleção | escolhe **uma** cadeia entre N (`selecionar_por_penalizacao`) | **mescla** hops úteis de todas as N cadeias |
| Modo "greedy" (1 amostra) | sim (`estrategia="greedy"`) | não — sempre amostra `n` cadeias |
| Servidor E5 | sobe sozinho (`iniciar_servidor_e5`) | sobe sozinho (`iniciar_servidor_e5`, mesma função) |
| Documentos de contexto (`context_doc_ids`) na resposta final | sim (`format_documents_for_final_answer`) | sim (`format_documents_for_final_answer`, mesmos parâmetros) |
| Geração da resposta final | `CoRagAgent.generate_final_answer` | `CoRagAgent.generate_final_answer` (via `gerar_resposta_final`) |
