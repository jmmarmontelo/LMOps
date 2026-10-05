# Mudanças no fork — rodar `best_of_n` e `dynamic_chain`

Documento único consolidando **tudo que foi alterado em relação ao CoRAG original** para
conseguir rodar as estratégias **`best_of_n`** e **`dynamic_chain`** neste ambiente.
Os arquivos `best_of_n.md` e `dynamic_chain.md` descrevem cada script isoladamente; aqui o
foco é **o que mudou e por quê**.

Base: `git diff daf5134..HEAD` (`daf5134` = último commit upstream, "Merge pull request #438").
Autor de tudo abaixo: Joao Marcos Marmontelo.

---

## 1. Visão geral

O CoRAG original é **código só de inferência**, feito para rodar em 8×A100 40 GB, com:

- um **vLLM local** servindo o modelo em `localhost:8000` (OpenAI-compatível, com suporte a
  `extra_body={"prompt_logprobs": 1}`);
- um **servidor E5** em `localhost:8090` carregando o **corpus KILT inteiro** (~36 M
  passagens) e ~71 GB de embeddings pré-computados;
- execução via `scripts/*.sh` + `torchrun`, configurada por `src/config.py`.

Este ambiente é diferente em três pontos, e cada um forçou uma adaptação:

| Aspecto | CoRAG original | Aqui |
| --- | --- | --- |
| Backend LLM | vLLM local (`localhost:8000`, http) | backend remoto OpenAI-compatível (`https://amora.dac.ufla.br/v1`) — **não** retorna `prompt_logprobs` |
| Retrieval | corpus KILT inteiro + 71 GB de embeddings | **mini-corpus local** por benchmark, ou corpus completo via **índice FAISS** |
| Execução | `scripts/eval_multihopqa.sh` + `torchrun` + `src/config.py` | **drivers Python** em `src/reproducao/`, config por `.env` + variáveis de ambiente |
| Hardware alvo | 8×A100 | i5-4440, GTX 1650 4 GB, ~10 GB RAM (via WSL2) |

### Arquivos novos (todos em `src/reproducao/`, exceto onde indicado)

- `best_of_n.py` — driver da estratégia `best_of_n` **e biblioteca compartilhada** (todo o
  resto importa daqui).
- `dynamic_chain.py` — driver da estratégia `dynamic_chain`.
- `construir_mini_corpus.py` — gera o mini-corpus/índice/perguntas locais.
- `rodar_split_em_chunks.py` — roda um split inteiro em chunks resumíveis.
- `teste_paralelo.py` — smoke test paralelo (100 perguntas de 2wikimultihopqa).
- `best_of_n.md`, `dynamic_chain.md` — docs por script.
- `.env` / `.env.example` — config do backend (`.env` está no `.gitignore`).
- `src/search/construir_indice_faiss.py` — builder do índice FAISS do corpus completo.
- `docs/estrategias_inferencia.md`, `docs/prompts_estrategia_gulosa.md`,
  `docs/experimentos_fonte_documentos.md`, `docs/plano_indice_faiss_corpus_completo.md`,
  `docs/fluxo_estrategia_*.pdf` — notas de projeto.

### Arquivos do core que foram tocados

`src/agent/corag_agent.py` (só prints de debug), `src/data_utils.py`,
`src/search/e5_searcher.py`, `src/search/simple_encoder.py`,
`src/search/start_e5_server_main.py`, `requirements.txt`, `.gitignore`.

**Não** foram alterados: `src/config.py`, `src/inference/`, `scripts/`. As estratégias
foram **reimplementadas fora do pipeline oficial**, não plugadas no `decode_strategy` do
`src/config.py`.

### Modelo servido pelo backend (`corag-8b`)

O `MODEL_NAME=corag-8b` do `.env` (§8) **não** é o modelo original em `safetensors` — é uma
**quantização GGUF Q4_K_M** do `corag/CoRAG-Llama3.1-8B-MultihopQA`, feita com `llama.cpp`
para caber no hardware alvo. Isto descreve o *setup do backend*, não uma alteração no código
deste repositório.

**Pipeline:** HF `safetensors` → GGUF F16 → GGUF Q4_K_M.

1. **Toolchain** — `llama.cpp` compilado com CMake / GCC 13.3.0 (Linux x86_64), gerando o
   binário `llama-quantize`.
2. **Conversão HF → GGUF** (`hf-to-gguf`): lê os 4 shards
   `model-0000X-of-00004.safetensors` (`bfloat16`, `LlamaForCausalLM`, 8B params, 32
   camadas, embedding 4096, FFN 14336, context length 131072, RoPE base 500000). Pesos
   `bfloat16` → **F16** (tensores de peso 2D) e `bfloat16` → **F32** (norms); tokenizer
   (`gpt2`/`llama-bpe`, vocab 128256) e chat template Jinja preservados. Saída
   `corag-8b-f16.gguf`: 292 tensores, ~15 GB (16.00 BPW).
3. **Quantização** (`llama-quantize corag-8b-f16.gguf corag-8b-q4_k_m.gguf Q4_K_M`, 32
   threads, ~34 s). Estratégia mista por tensor (preset Q4_K_M):
   - `output.weight`, `attn_v` e `ffn_down` de **metade** das camadas → **Q6_K** (tensores
     mais sensíveis, maior precisão);
   - `token_embd`, `attn_q`, `attn_k`, `attn_output`, `ffn_gate`, `ffn_up` e os demais
     `attn_v`/`ffn_down` → **Q4_K**;
   - norms mantidos em **F32**.

**Resultado:**

| Versão | Tamanho | BPW |
| --- | --- | --- |
| F16 (base) | 15317 MiB (~15 G) | 16.00 |
| Q4_K_M | 4685.30 MiB (~4.6 G) | 4.89 |

Redução de ~3.27× no tamanho, mantendo os tensores críticos (embedding de saída, projeção de
valor, down-projection) em precisão maior (Q6_K) e as camadas de maior volume
(Q/K/proj/FFN gate-up) em Q4_K — o trade-off padrão do método K-quants do `llama.cpp`.

> O **tokenizer** usado pelo pipeline continua o do repo HF original
> (`corag/CoRAG-Llama3.1-8B-MultihopQA`, carregado via o monkeypatch de `get_vllm_model_id`
> descrito na §2). Só os **pesos** servidos pelo backend estão quantizados.

---

## 2. Estratégia `best_of_n` — `src/reproducao/best_of_n.py`

### O que é no paper / upstream

`CoRagAgent.best_of_n` ([src/agent/corag_agent.py:172-192](../agent/corag_agent.py#L172)):
amostra N cadeias, pontua cada uma e devolve a de **menor** score. O score
(`_eval_single_path`) é a **prompt-logprob** que o modelo atribuiria à string fixa
`"No relevant information found"`, obtida com `extra_body={"prompt_logprobs": 1}`.

### Por que não dá para usar aqui

O backend remoto OpenAI-compatível **não retorna `prompt_logprobs`** — recurso exclusivo de
um servidor vLLM de verdade. Isso está documentado em `docs/estrategias_inferencia.md`
(`tree_search` e `best_of_n` originais dependem disso).

### Substituição: penalização por hops sem resposta

Em vez do logprob, a escolha usa uma heurística textual
([best_of_n.py:153-166](best_of_n.py#L153)):

- `contar_hops_sem_resposta(path)` — conta quantos subanswers da cadeia são exatamente
  `"no relevant information found"` (constante `SEM_RESPOSTA`, comparação `strip().lower()`).
- `selecionar_por_penalizacao(caminhos)` — devolve o índice da cadeia com **menos** desses
  hops. Empate → primeira ocorrência, que é sempre a amostra greedy (`temperature=0`).

A mesma convenção de texto (`SEM_RESPOSTA`) é a que a instrução do prompt de subresposta
usa (`src/prompts.py`).

### Fluxo por pergunta (`_processar_exemplo`, [best_of_n.py:169-233](best_of_n.py#L169))

1. `num_amostras = n` (para `best_of_n`; para `"greedy"` seria 1).
2. Para cada amostra, `corag_agent.sample_path(...)` com `max_path_length=L`,
   `temperature=0.` na 1ª e `0.7` nas demais, `max_tokens=64`.
3. `selecionar_por_penalizacao(caminhos)` escolhe **uma** cadeia.
4. `format_documents_for_final_answer(...)` monta os documentos do prompt final. Fonte:
   `context_doc_ids` do dataset (`DOC_SOURCE=dataset`, default) **ou** fusão RRF dos doc_ids
   que a cadeia recuperou (`DOC_SOURCE=chain`, via `fundir_doc_ids_por_rrf`).
5. **Uma** chamada a `corag_agent.generate_final_answer(...)`
   (`max_message_length=3072`, `temperature=0.`, `max_tokens=128`).
6. Registra no JSONL: subqueries, subanswers, doc_ids, `penalizacoes` (contagem por
   candidato), `doc_source`, `prediction`.

### Parâmetros (`__main__`, [best_of_n.py:282-321](best_of_n.py#L282))

`estrategia = "best_of_n"`, `n = 4`, `max_path_length = 6` (o "L" definido no commit
`4459bb7`). Tasks: as 4 de `TASK_SPLITS` (`hotpotqa`, `2wikimultihopqa`, `musique` em
`validation`; `bamboogle` em `test`), sobrescrevível por `TASKS`.

> Nota: `best_of_n.md` ainda cita `n = 2` — está desatualizado; o código atual usa `n = 4`.

### Contornos do backend remoto (`executar_rag`, [best_of_n.py:236-256](best_of_n.py#L236))

- `vllm_client.client = OpenAI(base_url=base_url, api_key=api_key)` — o `VllmClient` monta a
  URL fixando o esquema `http://host:port`; o endpoint remoto é `https`, então o client
  OpenAI interno é substituído por um com a `base_url` completa.
- `corag_agent_module.get_vllm_model_id = lambda *a, **kw: tokenizer_name_or_path` —
  monkeypatch que evita a consulta a `localhost:8000/v1/models` feita dentro de
  `CoRagAgent.__init__`. O `model` passado nas chamadas é só o apelido (`corag-8b`); o
  **tokenizer** é carregado do HF Hub (`corag/CoRAG-Llama3.1-8B-MultihopQA`).

### Paralelismo

`executar_rag` usa `ThreadPoolExecutor(max_workers=NUM_THREADS)` sobre `enumerate(dataset)`
(`.map` preserva a ordem). O único ponto que precisa de trava é o tokenizer — passado como
`lock=corag_agent.lock` para `format_documents_for_final_answer`. O JSONL só é escrito depois
de coletar todos os resultados, para não intercalar linhas de threads diferentes.
`NUM_THREADS` default 2 (teste local); no servidor, exportar um valor maior (ex. 32).

### Evolução

`teste.py` (`891e0a3`, estratégia `best_of_n_self_consistency` = voto majoritário sobre N
respostas **finais**, gerava N respostas) → renomeado `reproducao.py` (`09d9ba0`) →
renomeado `best_of_n.py` + troca da self-consistency por `selecionar_por_penalizacao` + uma
resposta final só (`dcdabba`) → `L = 6` e `n` parametrizável (`4459bb7`) → E5 sobe pelo
código (`53d424d`) → paralelismo + RRF/`DOC_SOURCE` + `MINI_BASE_DIR` (`1a31950`).

---

## 3. Estratégia `dynamic_chain` — `src/reproducao/dynamic_chain.py`

### Ideia

Em vez de escolher **uma** cadeia entre as N amostradas, **mescla os hops úteis de todas**
numa única "cadeia dinâmica".

> **Não** é adaptação do número de hops nem *early-stop*. Cada cadeia amostrada roda sempre
> até `L` hops (o loop de `CoRagAgent.sample_path` não tem parada antecipada). "Dinâmica" se
> refere à cadeia final, montada por concatenação + filtro **depois** da amostragem.

### Funções-chave

- `executar_dynamic_chain(...)` ([dynamic_chain.py:198-222](dynamic_chain.py#L198)) —
  amostra N cadeias (1ª `temperature=0.`, resto `0.7`), **sem pontuar nem escolher**. O
  comentário no código explica: o `best_of_n` original pontuaria por logprob (indisponível
  aqui), e a decisão foi usar todas as N em vez de trocar por um critério de escolha única.
- `montar_chain_final(paths)` ([dynamic_chain.py:146-162](dynamic_chain.py#L146)) —
  concatena `past_subqueries` / `past_subanswers` / `past_doc_ids` das N cadeias numa
  `RagPath` "bruta" e aplica o filtro abaixo.
- `selecionar_subperguntas_respondidas(path)`
  ([dynamic_chain.py:125-143](dynamic_chain.py#L125)) — descarta um hop **só** se o subanswer
  for exatamente `"no relevant information found"`. **Esse é o único critério de descarte.**
  Uma deduplicação por subquery repetida existiu antes e **foi removida a pedido do
  usuário** (ver `dynamic_chain.md`), então subperguntas iguais respondidas em candidatas
  diferentes são todas mantidas.

### Resposta final

`gerar_resposta_final(...)` ([dynamic_chain.py:165-182](dynamic_chain.py#L165)) roteia por
`corag_agent.generate_final_answer` (para herdar o truncamento de mensagem longa — a cadeia
mesclada pode ter até `N × L` hops antes do filtro). Recebe os mesmos documentos de contexto
que o `best_of_n` (`context_doc_ids` do dataset via `format_documents_for_final_answer`,
`DOC_ARGS = num_contexts=5, max_len=3072, context_placement="backward"`), para deixar a
comparação entre as duas estratégias justa — só a forma de montar a cadeia muda.

### Parâmetros (`__main__`, [dynamic_chain.py:294-329](dynamic_chain.py#L294))

`n = 4`, `max_path_length = 6`. (`dynamic_chain.md` ainda diz `n = 2` — desatualizado.)

### Reúso de `best_of_n.py`

Importa `TASK_SPLITS`, `criar_dataset_benchmark`, `calcular_metricas`,
`fundir_doc_ids_por_rrf`, `DOC_SOURCE`, `NUM_CONTEXTS`, `RRF_K`, `NUM_THREADS`. Tem cópia
própria de `_porta_aberta` / `iniciar_servidor_e5` e o helper `_montar_agente` (mesmos dois
contornos de backend do `executar_rag`).

`executar_dynamic_chain_dataset(...)`
([dynamic_chain.py:265-291](dynamic_chain.py#L265)) roda o pipeline por dataset com
`ThreadPoolExecutor` — é o que `rodar_split_em_chunks.py` chama.
`executar_teste(...)` é resíduo de versão antiga, não é chamado.

### Evolução

Criado em `dcdabba` (`max_path_length=3` fixo, sem documentos de contexto no prompt final) →
`4459bb7` (`L` parametrizável = 6; `DOC_ARGS`; resposta final passa por
`format_documents_for_final_answer` + `generate_final_answer`; `montar_prompt_final`
removido) → `1a31950` (`executar_dynamic_chain_dataset` paralelo, RRF/`DOC_SOURCE`).

---

## 4. Servidor E5 iniciado pelo código

**Antes:** `bash scripts/start_e5_server.sh` rodado à parte (o script continua no repo,
inalterado).

**Agora:** `iniciar_servidor_e5(...)` ([best_of_n.py:63-103](best_of_n.py#L63)):

- `_porta_aberta(host, port)` — probe TCP em `localhost:8090` (`socket.connect_ex`).
- Se a porta já está aberta → assume que já há um servidor, imprime aviso e retorna `None`.
- Senão → `subprocess.Popen(["uvicorn", "src.search.start_e5_server_main:app", "--port",
  "8090"], cwd=REPO_ROOT, env=...)` com `INDEX_DIR`, `CORPUS_DIR` (só quando um corpus local
  é passado — ver abaixo) e `PYTHONPATH=<repo>/src`. Log em `e5_server_mini.log`. Faz poll da
  porta a cada 2 s; levanta `RuntimeError` se o processo morre antes de subir, `TimeoutError`
  após 600 s.
- `parar_servidor_e5(processo)` ([best_of_n.py:106-117](best_of_n.py#L106)) — `terminate()`,
  depois `kill()` se não sair; **no-op se `processo is None`** (servidor externo, não é nosso
  para derrubar).

`CORPUS_DIR` **só** é setado quando `corpus_dir` é passado (`f337b38`). Sem ele,
`load_corpus()` cai no default de baixar o corpus KILT completo do HuggingFace — usado no
modo FAISS.

**Chamadores:** `best_of_n.py`, `dynamic_chain.py`, `teste_paralelo.py`,
`rodar_split_em_chunks.py` (um servidor por chunk, ou um só no modo `CORPUS_COMPLETO`).

### Suporte no app do servidor (`src/search/start_e5_server_main.py`, `891e0a3`)

- `@app.on_event("startup")` (deprecado no Starlette) → `lifespan` (`asynccontextmanager`
  passado a `Starlette(..., lifespan=lifespan)`).
- Nova env `MAX_SHARDS` → `E5Searcher(max_shards=...)` (carrega só os N primeiros shards, útil
  para teste rápido).

### Bug já corrigido (ver memória do projeto)

`iniciar_servidor_e5` **só** checa se a porta está aberta — pode reusar um servidor que está
servindo o **corpus errado**. Mitigações:

- `rodar_split_em_chunks.py` adiciona `_matar_processo_na_porta(8090)` (via `psutil`) antes
  de subir o servidor de cada chunk, para matar listeners órfãos de execuções anteriores.
- `f337b38` garante `parar_servidor_e5(...)` num `finally` no modo `CORPUS_COMPLETO`, para o
  servidor único não vazar se um chunk lançar exceção.

Se o retrieval parecer quebrado, **checar isto primeiro**.

---

## 5. Mini-corpus local — `src/reproducao/construir_mini_corpus.py`

### Por quê

Rodar sem baixar as ~36 M passagens do KILT e os ~71 GB de embeddings. Ressalva: com poucos
milhares de documentos candidatos o retrieval fica fácil demais e as métricas **não são
comparáveis** com o paper — ver `docs/plano_indice_faiss_corpus_completo.md` e
`docs/experimentos_fonte_documentos.md`.

### Como

1. União dos `context_doc_ids` *gold* de todas as perguntas selecionadas +
   `N_DISTRATORES_POR_BENCHMARK = 1000` distratores aleatórios por benchmark (`SEED = 42`).
2. Remapeia os doc ids escolhidos para `0..N-1` (`construir_mapa_ids`).
3. `construir_tabela_shards` + `extrair_embeddings` leem os shards `*.pt` originais com
   `mmap=True` e puxam **só as linhas necessárias** para um tensor `(N, 1024) float16`.
4. Salva em `data/mini/` (ou `MINI_BASE_DIR`): `corpus/` (Arrow, com coluna `orig_doc_id`),
   `e5-large-index/e5-large-shard-0.pt`, `questions/{task}` (com `context_doc_ids`
   remapeados), `id_map.json`.

CLI: `--tasks`, `--n-perguntas` (30), `--n-distratores` (1000), `--seed` (42),
`--output-dir` (`data/mini`). `fatiar_dataset_sequencial` (fatia contígua não-aleatória de um
split) é reusada pelo runner em chunks.

### Suporte no core

`src/data_utils.py` (`80bdd25`): `load_corpus()` → `load_corpus(corpus_dir: Optional[str] =
None)`. Se `corpus_dir` (ou a env `CORPUS_DIR`) estiver setado, carrega
`Dataset.load_from_disk(corpus_dir)` em vez de baixar `corag/kilt-corpus` do HF.

`criar_dataset_benchmark` (em `best_of_n.py`): carrega as perguntas do disco quando
`MINI_DATASET_DIR` está setado; senão baixa e amostra `corag/multihopqa`.

---

## 6. Índice FAISS para o corpus completo — `src/search/construir_indice_faiss.py`

### Por quê (`f337b38` + `docs/plano_indice_faiss_corpus_completo.md`)

Rodar contra o corpus KILT inteiro (mais fiel ao paper) **sem** carregar os ~71 GB de
embeddings na RAM.

### Builder (arquivo novo, ~100 linhas)

`IndexIVFPQ` — quantizer `IndexFlatIP` (produto interno, mesma métrica do `torch.mm`
original), `DIM=1024`, `NLIST=4096`, `PQ_M=256`, `PQ_NBITS=8` (defaults, via env). Treina
numa amostra (`TRAIN_PER_SHARD`, `TRAIN_SHARD_STRIDE` — 1 a cada N shards), depois
`index.add` shard a shard em lotes de `BATCH_SIZE`, convertendo float16→float32. Grava
`data/e5-large-index-faiss/index.faiss`. O `add` é sequencial, então os ids internos do FAISS
== `doc_id` (índice da linha no corpus) — nada de `add_with_ids`.

Uso: `PYTHONPATH=src python src/search/construir_indice_faiss.py`.

> `PQ_M=256` (~74 % de recall@10 vs. busca exata) foi escolhido depois de medir: `m=32` dava
> só ~10 % de recall, o que causava muitos `"no relevant information found"` no pipeline.
> O `index.faiss` presente em disco (~1.44 GB, de 19/ago) é de um `m` menor, **anterior** ao
> default atual.

### E5Searcher (`src/search/e5_searcher.py`, `f337b38`)

- `_get_faiss_index_path(index_dir)` — procura um único `*.faiss` no `INDEX_DIR`.
- `__init__`: se achou → `self.usa_faiss = True`, `faiss.read_index(...)`,
  `self.faiss_index.nprobe = int(os.getenv('FAISS_NPROBE', '32'))`.
- `_compute_topk`: se `usa_faiss` → `self.faiss_index.search(query_embed.float().numpy(), k)`
  e retorna. Senão → caminho *brute-force* `torch.mm` + `torch.topk` original.
- `batch_search`: só faz o cast do embedding da query para o dtype dos shards no caminho
  não-FAISS.

**Não há flag** — o backend FAISS é ativado apenas pela **presença** de um `*.faiss` no
diretório apontado por `INDEX_DIR`. Ajuste de recall/velocidade: env `FAISS_NPROBE`.

O mesmo commit `891e0a3` que precede isto já tinha adicionado ao `E5Searcher`: **fallback
para CPU** (`self.devices = ['cpu']` quando não há GPU) e o parâmetro `max_shards`.

### Pendência

`faiss-cpu` **não** foi adicionado a `requirements.txt` (foi instalado à mão no `.venv`). O
`docs/plano_indice_faiss_corpus_completo.md` lista isso como item a fazer.

---

## 7. Runner em chunks — `src/reproducao/rodar_split_em_chunks.py`

### Por quê

Rodar um split inteiro (ex.: `2wikimultihopqa` validation, ~12 k perguntas) em máquina
modesta, de forma **resumível**.

### Como

Fatia o split em chunks de `CHUNK_SIZE = 1000`. Para cada chunk (`processar_chunk`):

1. `fatiar_dataset_sequencial` → coleta ids de contexto → amostra distratores → constrói
   mini-corpus + índice + perguntas remapeadas (reusa `construir_mini_corpus.py`).
2. `_matar_processo_na_porta(8090)` → `iniciar_servidor_e5(index_dir, corpus_dir)`.
3. Roda a estratégia (`ESTRATEGIA`): `executar_rag(..., estrategia=ESTRATEGIA)` para
   `greedy`/`best_of_n`, ou `_montar_agente` + `executar_dynamic_chain_dataset` para
   `dynamic_chain`.
4. Grava `data/chunks/{TASK}/{ESTRATEGIA}/rag_log_{ini}-{fim}.jsonl` (**pula o chunk se esse
   arquivo já existe**).
5. `finally: parar_servidor_e5(...)`; opcionalmente `rmtree` do mini-corpus do chunk.

`MAX_CHUNKS = 1` → cada invocação processa **1 chunk novo** e para; rode o script de novo
para o próximo. Métricas agregadas em `metricas_completo.json` (split todo coberto) ou
`metricas_parcial.json`, com `tempo_execucao_segundos`, `chunks_concluidos`, etc.

### Config por variável de ambiente

`TASK` (`2wikimultihopqa`), `CHUNK_SIZE` (1000), `ESTRATEGIA`
(`greedy`|`best_of_n`|`dynamic_chain`), `N_CHAINS` (4), `MAX_PATH_LENGTH` (6), `NUM_THREADS`
(2), `MAX_CHUNKS` (1), `N_DISTRATORES_POR_CHUNK` (1000), `LIMPAR_MINI_CORPUS_APOS_CHUNK` (0),
`CORPUS_COMPLETO` (0), `FAISS_INDEX_DIR` (`data/e5-large-index-faiss`).

`CORPUS_COMPLETO=1` (`f337b38`): pula o mini-corpus por chunk, sobe **um** servidor E5 no
índice FAISS **antes** do loop (`iniciar_servidor_e5(FAISS_INDEX_DIR, None)`), usa
`processar_chunk_corpus_completo` (sem remapeamento de ids), e derruba o servidor no
`finally` do `main`.

### Smoke test — `teste_paralelo.py`

Roda `best_of_n` ou `dynamic_chain` sobre `data/mini_2wiki100` (100 perguntas de
2wikimultihopqa). CLI: `--estrategia {best_of_n,dynamic_chain}` (default `best_of_n`),
`--n-perguntas`, `--n-chains` (4), `--max-path-length` (6). Imprime wall-time, média por
pergunta, extrapolação para 100 e EM/F1.

---

## 8. Mudanças no core (resumo) e config via `.env`

| Arquivo | Mudança | Commit |
| --- | --- | --- |
| `src/agent/corag_agent.py` | **Só** `print('[CoRAG greedy] ...')` de debug em `sample_path` (passo, subquery, "subquery repetida", subanswer, doc_ids, dump final). Lógica **inalterada** — sem early-stop. | `891e0a3` |
| `src/data_utils.py` | `load_corpus(corpus_dir=None)` aceita arg/env `CORPUS_DIR` e carrega do disco (§5). | `80bdd25` |
| `src/search/simple_encoder.py` | `torch_dtype = float16 if cuda else float32`. | `891e0a3` |
| `src/search/start_e5_server_main.py` | `lifespan` no lugar de `@app.on_event("startup")`; env `MAX_SHARDS` (§4). | `891e0a3` |
| `src/search/e5_searcher.py` | Fallback CPU + `max_shards` (`891e0a3`); backend FAISS (`f337b38`) (§4, §6). | `891e0a3`, `f337b38` |
| `requirements.txt` | Remove `flash-attn` e `vllm==0.6.0` (não há vLLM local aqui); adiciona `python-dotenv` e `psutil`. | `891e0a3`, `1a31950` |
| `.gitignore` | Novo: `data/`, `tmp/`, `*.log`, `__pycache__/`, `*.pyc`, `.env`. | `891e0a3` |

### `.env` (em `src/reproducao/`, git-ignorado)

```
BASE_URL=https://amora.dac.ufla.br/v1
API_KEY=<sua-chave>
MODEL_NAME=corag-8b
```

Lido com `load_dotenv()` no `__main__` de cada driver. `.env.example` tem o template.
`MODEL_NAME=corag-8b` é a build GGUF Q4_K_M descrita na §1 ("Modelo servido pelo backend").

---

## 9. Como rodar

### Pré-requisitos

1. `src/reproducao/.env` preenchido (`BASE_URL`, `API_KEY`, `MODEL_NAME`).
2. Backend LLM remoto no ar.
3. Retrieval — um dos dois:
   - **mini-corpus**: `PYTHONPATH=src python src/reproducao/construir_mini_corpus.py`
     (gera `data/mini/`); ou
   - **corpus completo**: `bash scripts/download_embeddings.sh` (baixa
     `data/e5-large-index/`) e depois
     `PYTHONPATH=src python src/search/construir_indice_faiss.py`.

Os drivers sobem o servidor E5 sozinhos — não precisa rodar `scripts/start_e5_server.sh`.

### `best_of_n`

```bash
PYTHONPATH=src python src/reproducao/best_of_n.py
# env úteis: TASKS=2wikimultihopqa  NUM_THREADS=32  DOC_SOURCE=dataset  MINI_BASE_DIR=data/mini
```

### `dynamic_chain`

```bash
PYTHONPATH=src python src/reproducao/dynamic_chain.py
```

### Split inteiro em chunks (resumível)

```bash
PYTHONPATH=src ESTRATEGIA=best_of_n TASK=2wikimultihopqa \
  python src/reproducao/rodar_split_em_chunks.py
# rode de novo até cobrir o split todo; CORPUS_COMPLETO=1 usa o KILT via FAISS
```

### Saídas

- `data/rag_log_{task}.jsonl`, `data/rag_log_dynamic_chain_{task}.jsonl`
- `data/chunks/{TASK}/{ESTRATEGIA}/rag_log_{ini}-{fim}.jsonl`,
  `data/chunks/{TASK}/{ESTRATEGIA}/metricas_*.json`
- EM/F1 por task no stdout (`calcular_metricas` →
  `compute_metrics_dict(..., eval_metrics="em_and_f1")`).

---

## 10. Resumo commit a commit

| Commit | Tema |
| --- | --- |
| `891e0a3` | Prints de debug no CoRAG (`sample_path`); primeiro `teste.py`; `.env`/`.env.example`; `load_corpus` ainda sem `corpus_dir`; E5Searcher ganha fallback CPU + `max_shards`; servidor E5 migra para `lifespan` + `MAX_SHARDS`; `requirements.txt` sem `flash-attn`/`vllm`; `.gitignore`. |
| `fb96a80` | `teste.py`: estratégia default → `greedy`. |
| `80bdd25` | `construir_mini_corpus.py`; `load_corpus(corpus_dir=)` + env `CORPUS_DIR`; `criar_dataset_benchmark` genérico com `MINI_DATASET_DIR`. |
| `09d9ba0` | `teste.py` → `reproducao.py`; CLI `--estrategia/--n/--log-path`; notebook `reproducao.ipynb`; diagramas `docs/fluxo_estrategia_*.pdf`. |
| `dcdabba` | `reproducao.py` → `best_of_n.py`; troca self-consistency por `contar_hops_sem_resposta` + `selecionar_por_penalizacao`; uma resposta final só; **cria `dynamic_chain.py`** (`selecionar_subperguntas_respondidas` / `montar_chain_final`). |
| `4459bb7` | Define `L = max_path_length = 6` e torna `n`/`L` parametrizáveis nas duas estratégias; `dynamic_chain` passa a alimentar `context_doc_ids` no prompt final via `generate_final_answer`; cria `best_of_n.md` e `dynamic_chain.md`. |
| `53d424d` | `iniciar_servidor_e5` / `parar_servidor_e5` — o servidor E5 sobe por `subprocess` a partir do código (`_porta_aberta`, poll, log em `e5_server_mini.log`). |
| `1a31950` | `rodar_split_em_chunks.py` e `teste_paralelo.py`; `ThreadPoolExecutor` em `executar_rag` / `executar_dynamic_chain_dataset`; `fundir_doc_ids_por_rrf` + `DOC_SOURCE`/`RRF_K`/`NUM_CONTEXTS`/`NUM_THREADS`; `MINI_BASE_DIR`; `psutil` + `_matar_processo_na_porta`; `docs/plano_indice_faiss_corpus_completo.md`, `docs/experimentos_fonte_documentos.md`. |
| `f337b38` | Backend FAISS no `E5Searcher` (`_get_faiss_index_path`, `FAISS_NPROBE`); `src/search/construir_indice_faiss.py`; modo `CORPUS_COMPLETO=1` em `rodar_split_em_chunks.py` (servidor E5 único, `iniciar_servidor_e5(..., None)`); `finally` que derruba o E5 e evita vazamento em falha. |
