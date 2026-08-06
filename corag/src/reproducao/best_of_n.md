# `best_of_n.py` — como funciona

Script de reprodução do CoRAG (Chain-of-Retrieval Augmented Generation) usando um backend
OpenAI-compatible qualquer (não exige vLLM real), rodando contra o mini-corpus/mini-índice
local. Roda as 4 tasks de `TASK_SPLITS` (hotpotqa, 2wikimultihopqa, musique, bamboogle),
30 perguntas cada, e calcula métricas EM/F1 por task.

## Pré-requisitos pra rodar

1. **`.env`** em `src/reproducao/.env` com `BASE_URL`, `API_KEY`, `MODEL_NAME` do backend LLM.
2. **`data/mini/`** presente (corpus, índice E5 e perguntas) — gerado por
   `construir_mini_corpus.py` ou copiado de outra máquina onde já foi gerado.
3. **Servidor E5**: o próprio script sobe o servidor sozinho (`iniciar_servidor_e5`, mesma
   função de `dynamic_chain.py`) se a porta 8090 ainda não estiver ocupada — não precisa subir
   à parte.

## Visão geral do fluxo

```
__main__
 ├─ iniciar_servidor_e5(...)        → sobe o servidor E5 (mini-índice) se ainda não estiver no ar
 └─ para cada task em TASK_SPLITS (hotpotqa, 2wikimultihopqa, musique, bamboogle):
     ├─ carregar_perguntas(task)      → carrega as 30 perguntas do mini-dataset
     ├─ executar_rag(dataset, corpus, ...)
     │   └─ para cada pergunta:
     │       ├─ amostra N cadeias (ou 1, se estrategia="greedy")
     │       ├─ escolhe a melhor cadeia (só se estrategia="best_of_n")
     │       ├─ gera UMA resposta final a partir da cadeia escolhida
     │       └─ grava o resultado em data/rag_log_{task}.jsonl
     └─ calcular_metricas(resultados)  → EM/F1 da task
 └─ imprime resumo final de métricas por task
```

O `corpus` (mini-corpus, `data/mini/corpus`) é carregado **uma única vez**, antes do loop de
tasks, e reaproveitado entre elas — carregá-lo a cada task repetiria um custo de I/O
desnecessário.

## Funções

### Preparação de dados

- **`carregar_perguntas(task)`** — carrega direto do disco (`Dataset.load_from_disk`) as
  perguntas pré-amostradas em `data/mini/questions/{task}` (geradas previamente por
  `construir_mini_corpus.py`). Não baixa nem amostra nada em tempo de execução.
- **`criar_dataset_benchmark(task, n, split, aleatorio, seed)`** / **`criar_dataset_hotpotqa`** —
  função mais genérica que baixa do HuggingFace (`corag/multihopqa`) e amostra `n` exemplos
  aleatórios; usada por `construir_mini_corpus.py` e por `dynamic_chain.py` (via
  `MINI_DATASET_DIR`), mas **não** é chamada pelo `__main__` deste arquivo (que usa
  `carregar_perguntas`).
- **`calcular_metricas(resultados)`** — calcula EM/F1 (`compute_metrics_dict`, modo
  `em_and_f1`) comparando `resultados[i]["answers"]` (gabarito) com
  `resultados[i]["prediction"]` (resposta gerada).

### Seleção de cadeia sem logprob (substituto do `best_of_n` original)

O `best_of_n` original do CoRAG (`CoRagAgent.best_of_n`, em `agent/corag_agent.py`) escolhe entre
N cadeias amostradas calculando a logprob que o modelo atribuiria à resposta fixa
`"No relevant information found"` — recurso exclusivo de um servidor vLLM real
(`extra_body={"prompt_logprobs": 1}`). O backend usado aqui não suporta isso, então:

- **`contar_hops_sem_resposta(path)`** — conta quantos hops de uma cadeia (`RagPath`) tiveram
  subresposta igual a `"no relevant information found"` (comparação exata, case-insensitive).
- **`selecionar_por_penalizacao(caminhos)`** — dado uma lista de cadeias candidatas, retorna o
  índice da que tem **menos** hops sem informação relevante (empate resolvido pela primeira
  ocorrência, ou seja, a amostra greedy/`temperature=0`). É a métrica-substituta: penaliza
  cadeias que "não acharam nada" em vez de pontuar por logprob.

### `executar_rag(...)` — o laço principal

Parâmetros relevantes: `estrategia` (`"greedy"` ou `"best_of_n"`), `n` (nº de candidatos,
só usado no `best_of_n`), `max_path_length` (L, tamanho máximo da cadeia — hoje 6, ver
`__main__`), `temperature` (usada nas amostras não-gulosas do `best_of_n`).

Monta o `VllmClient`/`CoRagAgent` (com alguns contornos — ver seção "Detalhes técnicos" abaixo)
e, para cada pergunta do dataset:

1. **Monta os documentos de contexto** (`format_documents_for_final_answer`) a partir dos
   `context_doc_ids` do gabarito da pergunta — esses documentos alimentam a geração da resposta
   final, além da cadeia de subperguntas/subrespostas.
2. **Amostra cadeias**: `num_amostras = n` se `estrategia == "best_of_n"`, senão `1` (greedy). A
   1ª amostra é sempre `temperature=0.` (determinística); as demais usam `temperature` (default
   0.7) pra gerar diversidade entre candidatas. Cada amostra é uma chamada a
   `corag_agent.sample_path(...)`, que gera até `max_path_length` hops de
   (subpergunta → busca no E5 → subresposta).
3. **Escolhe uma cadeia**: se `best_of_n`, usa `selecionar_por_penalizacao`; se `greedy`, é
   sempre a única cadeia amostrada (índice 0).
4. **Gera a resposta final**: só **uma** chamada a `corag_agent.generate_final_answer(...)`, a
   partir da cadeia escolhida — diferente de uma implementação ingênua que geraria N respostas
   finais (uma por candidata) e votasse entre elas; aqui a escolha acontece **antes** da geração
   final, economizando N-1 chamadas de LLM por pergunta quando `estrategia="best_of_n"`.
5. **Loga e grava**: imprime cada hop (subpergunta/subresposta) da cadeia escolhida, grava um
   JSON por pergunta em `data/rag_log_{task}.jsonl` (`query`, `answers`, `subqueries`,
   `subanswers`, `doc_ids`, `penalizacoes` — penalização de cada candidata amostrada, útil pra
   depuração — e `prediction`, a resposta final).

### `__main__`

Variáveis simples no topo (sem argparse — editar o arquivo diretamente pra mudar):

```python
estrategia = "best_of_n"   # ou "greedy"
n = 2                      # nº de candidatos, só usado no best_of_n
max_path_length = 6        # L: tamanho máximo da cadeia de subperguntas
log_dir = "data"
```

Carrega o mini-corpus uma vez, itera pelas 4 tasks chamando `executar_rag` e `calcular_metricas`,
e no fim imprime um resumo `task → métricas`.

## Detalhes técnicos (contornos no `executar_rag`)

- **`vllm_client.client = OpenAI(base_url=base_url, ...)`** — o `VllmClient` original monta a
  `base_url` fixando `http://host:port`; como o backend remoto usa `https`, o client OpenAI
  interno é substituído por um apontando pra `base_url` completa.
- **`corag_agent_module.get_vllm_model_id = lambda *a, **kw: tokenizer_name_or_path`** —
  `CoRagAgent.__init__` normalmente descobre o id do modelo consultando
  `http://localhost:8000/v1/models` (API local de um vLLM real) só pra carregar o tokenizer
  certo. Esse contorno substitui essa chamada por um valor fixo
  (`corag/CoRAG-Llama3.1-8B-MultihopQA`), já que `model` aqui é só o apelido usado nas chamadas
  de chat, não o repositório real do tokenizer no HF Hub.

## Saídas

- `data/rag_log_{task}.jsonl` — um JSON por pergunta, sobrescrito a cada execução.
- Métricas impressas no console por task e um resumo final `task: {em, f1}`.
