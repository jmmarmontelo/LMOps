# Plano: índice FAISS para rodar o corpus KILT completo com pouca memória

> Documento standalone — escrito pra ser lido do zero (sem contexto de
> conversa anterior) na máquina onde a execução vai continuar.

## Objetivo

Rodar `src/reproducao/rodar_split_em_chunks.py` contra o **corpus KILT
completo** (~36M passagens) em vez do mini-corpus por chunk que ele usa
hoje, pra poder comparar os resultados com os números reportados no paper
do CoRAG — sem precisar carregar os ~70GB de embeddings inteiros na
memória.

## Por que o mini-corpus atual não serve pra comparar com o paper

Hoje, `rodar_split_em_chunks.py` constrói um mini-corpus por chunk (`data/
mini_chunks/{TASK}/{chunk}/`): só os documentos "gold" das perguntas daquele
chunk + alguns distratores aleatórios (função `construir_mini_corpus_e_
indice`, `src/reproducao/construir_mini_corpus.py`). Isso deixa a busca por
sub-pergunta dentro da chain muito mais fácil do que no paper (poucos
milhares de documentos candidatos, não os ~36M do KILT completo) — os
números não são diretamente comparáveis.

## Decisões já tomadas pra maximizar comparabilidade

Além de trocar pro corpus completo (este documento), pra comparar com o
paper de forma justa:

- **`DOC_SOURCE=dataset`** (env var lida em `src/reproducao/best_of_n.py`,
  linha ~44) — os documentos usados no prompt da resposta final devem vir
  de `context_doc_ids` (pré-computados pelos autores do paper, distribuídos
  no dataset `corag/multihopqa`), **não** de `DOC_SOURCE=chain` (ablação
  própria deste repo, que usa os doc_ids que a própria chain recuperou).
  Confirmado em `src/inference/run_inference.py:63-68` (pipeline oficial):
  ele sempre usa `ex['context_doc_ids']`, independente da estratégia de
  decodificação.
- **Não rodar a seleção de `best_of_n`** — a implementação local
  (`selecionar_por_penalizacao`, `best_of_n.py:140-149`) substitui a
  pontuação por logprob real do vLLM (`prompt_logprobs`, exclusiva de um
  servidor vLLM de verdade, indisponível no endpoint remoto usado aqui) por
  uma contagem de hops "sem informação relevante" — diverge do método do
  paper. Usar `ESTRATEGIA=greedy` evita esse problema por completo (decode
  guloso não precisa escolher entre candidatos).
- **Backend/modelo**: confirmado que o `BASE_URL` usado serve o checkpoint
  oficial `corag/CoRAG-Llama3.1-8B-MultihopQA`.

Com `DOC_SOURCE=dataset`, a troca de mini-corpus pra corpus completo (via
FAISS) **não afeta diretamente a métrica final** — ela só afeta a busca por
sub-pergunta dentro da chain (`CoRagAgent._get_subanswer_and_doc_ids`,
`src/agent/corag_agent.py:139-153`), que influencia indiretamente a
qualidade do raciocínio intermediário.

## Hardware alvo (a máquina onde isso vai rodar)

- CPU: Intel i5-4440 (4 núcleos, geração 2013)
- GPU: GTX 1650, 4GB VRAM
- RAM: 12GB físicos no host — **mas a execução é via WSL2 (Windows)**,
  então a RAM efetivamente disponível pro Linux/Claude Code é menor, ~6GB
  (ver seção "Rodando via WSL2" abaixo antes de assumir que só tem 6GB —
  provavelmente dá pra aumentar isso).
- Disco: SSD Kingston A400

**Análise de viabilidade:**
- GPU: só carrega o encoder E5 (`intfloat/e5-large-v2`, ~335M parâmetros,
  já é o que `SimpleEncoder`/`E5Searcher` fazem por padrão) — sobra margem
  nos 4GB.
- RAM: o índice FAISS comprimido (IVF+PQ) mira ficar em ~2-3GB. Com os
  ~6GB do WSL2 (default), fica mais apertado que o inicialmente estimado —
  ver sizing conservador na seção WSL2.
- SSD: ler os ~70GB de shards de embeddings (só na construção do índice,
  uma vez) deve levar poucos minutos, não é gargalo.
- CPU: é a etapa mais lenta — treinar o índice (k-means do IVF + codebook
  do PQ) e adicionar os 36M vetores é trabalho de CPU, esperado na faixa de
  dezenas de minutos (não horas) nesse processador. É um custo **único**
  (a construção do índice não se repete a cada execução). A busca em tempo
  de execução (por sub-pergunta) deve ser rápida mesmo nesse CPU, já que
  IVF+PQ evita comparar contra os 36M vetores a cada query — bem diferente
  da busca brute-force atual.

### Rodando via WSL2 (Windows) — ler antes de começar

A máquina alvo roda Windows; a execução (Claude Code, Python, FAISS) é
dentro do WSL2. Isso muda três coisas:

1. **RAM do WSL2 é limitada por padrão a ~50% da RAM do host.** Com 12GB
   físicos, isso dá os ~6GB observados — não é um limite de hardware, é uma
   configuração. Pra aumentar: no lado **Windows** (não dentro do WSL),
   criar/editar `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   memory=10GB
   ```
   e depois, no PowerShell: `wsl --shutdown` (fecha o WSL) e reabrir o
   terminal WSL de novo. Isso libera bem mais margem (10GB em vez de 6GB)
   sem precisar de mais hardware.
2. **GPU (GTX 1650) precisa de passthrough CUDA-on-WSL**: instalar o
   driver NVIDIA com suporte a WSL **no Windows** (não instalar nenhum
   driver de GPU separado dentro do WSL — isso quebra o passthrough) e o
   toolkit CUDA dentro do WSL. Confirmar que funcionou rodando `nvidia-smi`
   dentro do WSL — se reconhecer a GTX 1650, está OK.
3. **Sizing mais conservador do índice FAISS**, caso não dê pra ajustar o
   `.wslconfig` (ou mesmo depois de ajustar, como margem de segurança):
   - Amostra de treino do IVF/PQ: usar ~200-300k vetores em vez de ~500k-1M
     (500k vetores em float32 já usa ~2GB só pra essa etapa transitória).
   - `m=32` em vez de `m=64` no `IndexIVFPQ` (32 bytes/vetor em vez de 64
     → 36M vetores ≈ 1.15GB em vez de ~2.3GB) — troca uma fração de recall
     por bem mais margem de memória durante a construção do índice, que é
     o pico de memória mais apertado de todo o processo.

## O que implementar

### 1. Script de construção do índice: `src/search/construir_indice_faiss.py` (novo)

Reaproveitar `_get_all_shards_path` (`src/search/e5_searcher.py:12-21`) pra
iterar os 40 arquivos `data/e5-large-index/*-shard-*.pt` na ordem correta —
a mesma ordem que hoje define `doc_id` = índice da linha no corpus (via
`idx_offset` em `E5Searcher._compute_topk`, `e5_searcher.py:90-99`).

Passos:

1. Carregar cada shard com `torch.load(path, mmap=True, weights_only=True,
   map_location="cpu")` — mesmo padrão já usado em
   `src/reproducao/construir_mini_corpus.py:65` (`extrair_embeddings`).
   Nunca materializar os 70GB de uma vez.
2. Amostrar um subconjunto de treino e treinar:
   ```python
   import faiss
   dim = 1024  # dimensao do e5-large-v2
   quantizer = faiss.IndexFlatIP(dim)  # produto interno = mesma metrica do torch.mm atual
   index = faiss.IndexIVFPQ(quantizer, dim, 4096, 64, 8)  # nlist=4096, m=64, nbits=8
   index.train(amostra_treino)  # amostra_treino: np.ndarray float32, shape (N, 1024)
   ```
   `m=64, nbits=8` → 64 bytes/vetor comprimido → 36M vetores ≈ 2.3GB. **Se
   estiver rodando com a RAM limitada do WSL2 (ver seção "Rodando via
   WSL2" acima)**, usar parâmetros mais conservadores: amostra de treino
   de ~200-300k vetores (em vez de ~500k-1M — 500k em float32 já usa ~2GB
   só nessa etapa transitória) e `m=32` em vez de `m=64` (32 bytes/vetor →
   36M vetores ≈ 1.15GB, mais folga ao custo de um pouco de recall).
3. Adicionar os vetores **shard por shard**, na mesma ordem de
   `_get_all_shards_path` (convertendo cada shard de float16 pra float32 só
   no momento do `index.add`, um shard por vez):
   ```python
   for shard_path in _get_all_shards_path(index_dir):
       shard = torch.load(shard_path, mmap=True, weights_only=True, map_location="cpu")
       index.add(shard.float().numpy())
       del shard
   ```
   Isso garante que os ids internos do FAISS (sequenciais, 0, 1, 2, ...)
   caiam exatamente nos mesmos `doc_id` que o código atual já usa — não
   precisa de `add_with_ids`.
4. Salvar: `faiss.write_index(index, "data/e5-large-index-faiss/index.faiss")`.

Testar primeiro com poucos shards (reaproveitar o padrão de `MAX_SHARDS` já
existente em `src/search/start_e5_server_main.py:30-34`) antes de rodar nos
40 completos.

### 2. Adaptar `src/search/e5_searcher.py`

`E5Searcher` passa a aceitar um índice FAISS como alternativa ao caminho
brute-force atual: se `INDEX_DIR` apontar pra uma pasta com um arquivo
`*.faiss` (em vez de `*-shard-*.pt`), carregar via `faiss.read_index(...)`
e usar `index.search(query_embed_np, k)` no lugar do `torch.mm` + `topk`
atual (`_compute_topk`, `e5_searcher.py:86-107`). `index.nprobe` (quantos
clusters IVF são vasculhados por busca — trade-off recall/velocidade) deve
ficar configurável por env var (ex. `FAISS_NPROBE`, default ~32-64). O
encoder (`SimpleEncoder`, roda na GPU) não muda — só converter a saída pra
numpy float32 antes de chamar `index.search`. `iniciar_servidor_e5`/
`INDEX_DIR` (em `src/reproducao/best_of_n.py`) não precisam mudar — só
passam a apontar pra essa nova pasta.

### 3. `requirements.txt`

Adicionar `faiss-cpu` (não `faiss-gpu` — 4GB de VRAM não compensa a
complexidade de compatibilidade de versão CUDA só pra guardar o índice, que
já cabe confortavelmente na RAM).

### 4. Ajustar `src/reproducao/rodar_split_em_chunks.py` pra corpus completo

Com o índice FAISS pronto, não é mais necessário construir mini-corpus por
chunk. Ajustes no fluxo de `processar_chunk`:
- Pular a construção de mini-corpus (`fatiar_dataset_sequencial` +
  `coletar_ids_contexto` + `amostrar_distratores` + `construir_mapa_ids` +
  `construir_mini_corpus_e_indice` + `remapear_e_salvar_datasets`) quando
  um modo "corpus completo" estiver ativado (ex. nova env var
  `CORPUS_COMPLETO=1`).
- Servidor E5 deve apontar direto pra `data/e5-large-index-faiss/` (índice)
  e pro corpus completo (`CORPUS_DIR` sem valor = usa `corag/kilt-corpus`
  do HuggingFace direto, comportamento default de `load_corpus()` em
  `src/data_utils.py:16-24`).
- Iniciar o servidor E5 **uma única vez**, fora do loop de chunks em
  `main()` (não dentro de `processar_chunk`) — já que o corpus/índice não
  muda mais entre chunks. Reaproveitar `iniciar_servidor_e5`/
  `_matar_processo_na_porta` (já existente em `rodar_split_em_chunks.py`),
  chamados uma vez no início de `main()`.

## Portabilidade — não precisa copiar os 70GB manualmente

Os shards brutos (`data/e5-large-index/`) só são necessários na etapa de
**construção** do índice FAISS. Na máquina nova, rodar:
```bash
bash scripts/download_embeddings.sh
```
baixa eles direto do HuggingFace (`corag/kilt-corpus-embeddings`) — evita
transferir 70GB via pendrive/rede. Depois de construído o índice FAISS
(poucos GB), os shards brutos podem até ser apagados; só o `index.faiss` é
necessário pra rodar a busca depois.

## Checklist pra continuar na máquina nova

1. Copiar/clonar a pasta do projeto (`corag/`) — este documento vem junto.
2. `pip install -r requirements.txt` (já vai incluir `faiss-cpu` depois do
   passo 3 do plano acima).
3. Configurar `.env` na raiz com `BASE_URL`/`API_KEY` (mesmo backend já
   validado, modelo oficial `corag/CoRAG-Llama3.1-8B-MultihopQA`).
4. `bash scripts/download_embeddings.sh` — baixa os shards de embeddings.
5. Implementar os itens 1-4 da seção "O que implementar" acima (pedir pro
   Claude Code seguir este documento).
6. Construir o índice FAISS (passo 1) — validar primeiro com poucos shards
   (`MAX_SHARDS`), depois rodar completo.
7. Rodar `rodar_split_em_chunks.py` com `CORPUS_COMPLETO=1 DOC_SOURCE=dataset
   ESTRATEGIA=greedy` num chunk pequeno de teste e conferir que as buscas
   por sub-pergunta retornam documentos plausíveis (não "no relevant
   information found" na maioria das vezes).

## Verificação

1. Construir o índice numa amostra pequena primeiro (`MAX_SHARDS=2`) pra
   validar o pipeline de ponta a ponta rápido, antes dos 40 shards
   completos.
2. Comparar o top-k retornado pelo índice FAISS com o top-k exato
   (busca brute-force atual) pra um punhado de queries de teste — checar
   que a sobreposição (recall) está alta (ex. >90%) com o `nprobe`
   escolhido.
3. Medir o tempo de construção do índice completo (40 shards) nessa
   máquina, pra confirmar que fica na faixa esperada (dezenas de minutos).
4. Rodar `rodar_split_em_chunks.py` num chunk pequeno apontando pro índice
   FAISS + corpus completo e conferir que as respostas fazem sentido.

## Referência: bug já corrigido nesta base de código (não repetir)

`iniciar_servidor_e5` (`best_of_n.py`) só verifica se a porta 8090 já está
aberta — se estiver, reaproveita o que já está rodando **sem checar se
aponta pro corpus/índice certos**. Isso já causou um bug real (servidor
órfão de um script anterior servindo o corpus errado, gerando "no relevant
information found" pra quase todas as sub-perguntas). Já corrigido em
`rodar_split_em_chunks.py` via `_matar_processo_na_porta()` (mata qualquer
processo na porta antes de subir um servidor novo) — ao ajustar o fluxo
pro item 4 acima (servidor único por execução, fora do loop de chunks),
manter essa chamada no início de `main()`, antes do único
`iniciar_servidor_e5()`.
