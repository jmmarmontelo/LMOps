# `spmc_no_placeholder.py` explicado

Este documento explica, parte por parte, como
[`spmc_no_placeholder.py`](spmc_no_placeholder.py) implementa a variante **sem
placeholders** do SPMC-RAG, e como uma pergunta percorre o código do início ao fim.
É o equivalente, para esta variante, do que o [`CODIGO.md`](CODIGO.md) é para o
`spmc.py`.

Resumo em uma frase: um LLM (gpt-oss:120b via Ollama) gera um **plano** com N cadeias
alternativas de L passos, cada passo descrito só por um texto guia (`guide_text`); as N
cadeias executam **em paralelo e isoladas**, e em cada passo geram uma subpergunta,
recuperam top-k chunks do mini-corpus com E5 e geram uma subresposta. No fim, uma
**resposta final** é produzida a partir das subrespostas de todas as cadeias e dos 20
chunks (dentre os recuperados pelas cadeias) mais relevantes para a pergunta original.

---

## Sumário

0. [Visão geral do pipeline](#0-visão-geral-do-pipeline)
1. [Diferenças em relação ao `spmc.py`](#1-diferenças-em-relação-ao-spmcpy)
2. [Configuração e constantes](#2-configuração-e-constantes)
3. [Os prompts](#3-os-prompts)
4. [Schema Pydantic do plano](#4-schema-pydantic-do-plano)
5. [Chamada ao LLM](#5-chamada-ao-llm)
6. [Retrieval E5 in-process sobre o mini-corpus](#6-retrieval-e5-in-process-sobre-o-mini-corpus)
7. [Subgrafo de uma cadeia](#7-subgrafo-de-uma-cadeia)
8. [Grafo de nível superior](#8-grafo-de-nível-superior)
9. [Fluxo de execução passo a passo](#9-fluxo-de-execução-passo-a-passo)
10. [Avaliação no mini dataset](#10-avaliação-no-mini-dataset)
11. [Limitações e pontos de atenção](#11-limitações-e-pontos-de-atenção)

---

## 0. Visão geral do pipeline

O código usa **dois grafos LangGraph**, um dentro do outro.

**Grafo de nível superior** (`construir_grafo`, estado `PlanoState`):

```
                         ┌──► executar_cadeia (cadeia 1) ──┐
                         │                                  │
START ──► gerar_plano ───┼──► executar_cadeia (cadeia 2) ──┼──► gerar_resposta_final ──► END
                         │            ...                   │
                         └──► executar_cadeia (cadeia N) ──┘
               fan-out com Send (N cópias)        fan-in: reducer operator.add em cadeias_exec
```

Antes de chamar o LLM, `gerar_resposta_final` junta os chunks recuperados em todos os
passos de todas as cadeias e seleciona os `FINAL_DOCS` (20) mais similares à pergunta
original, que entram no prompt junto com as cadeias.

**Subgrafo de uma cadeia** (`construir_grafo_cadeia`, estado `CadeiaState`), rodado
por cada `executar_cadeia`:

```
START ──► gerar_subpergunta ──► retrieve ──► gerar_subresposta ──┬──► END (passo_atual == L)
                ▲                                                │
                └────────────────────────────────────────────────┘
                              (passo_atual < L)
```

Cada volta do laço é **um passo (hop)** da cadeia; o laço dá exatamente L voltas.

### O que cada nó faz

| Grafo | Nó | Chama LLM? | Teto de tokens | `retry_policy` |
|---|---|---|---|---|
| topo | `gerar_plano` | sim (`json_mode`) | `PLAN_MAX_TOKENS` (8192) | `RETRY_LLM` |
| topo | `executar_cadeia` | não (invoca o subgrafo) | — | nenhuma |
| topo | `gerar_resposta_final` | sim (+ seleção E5 dos chunks) | `FINAL_MAX_TOKENS` (2048) | `RETRY_LLM` |
| cadeia | `gerar_subpergunta` | sim | `SUBQ_MAX_TOKENS` (2048) | `RETRY_LLM` |
| cadeia | `retrieve` | não (E5 local) | — | nenhuma |
| cadeia | `gerar_subresposta` | sim | `SUBA_MAX_TOKENS` (2048) | `RETRY_LLM` |

Todas as chamadas de LLM usam o **mesmo modelo** (`LLM_MODEL`).

---

## 1. Diferenças em relação ao `spmc.py`

| Aspecto | `spmc.py` | `spmc_no_placeholder.py` |
|---|---|---|
| Passo do plano | triplas com placeholders (`#1`, `#2`...) resolvidos em runtime | só um `guide_text` em linguagem natural |
| Como um passo usa passos anteriores | `resolver_placeholder` substitui o placeholder pelo valor achado | o LLM de subpergunta lê o histórico da cadeia e reescreve a referência ("o diretor achado no passo 1") com a subresposta |
| Modelos | Gemini (plano) + CoRAG-8B (resto) | gpt-oss:120b via Ollama para tudo |
| Extração de evidência | nó `evidence_extractor` | não existe; a subresposta sai direto dos chunks |
| Retry de subpergunta | reformula a subpergunta se não achou resposta | não existe; o passo segue com `SEM_RESPOSTA` |
| Retrieval | mini-corpus local + reserva HTTP | só mini-corpus local |
| Prompt da resposta final | só os passos respondidos das cadeias (subpergunta, subresposta, tripla) | passos respondidos das cadeias + os 20 chunks mais relevantes à pergunta |
| Driver de avaliação | `rodar_hotpotqa30.py` à parte | `rodar_mini_dataset` no próprio arquivo, com as 4 tarefas |
| Dependências | — | arquivo **standalone**: não importa nada de `spmc.py` |

A ideia da variante: tirar a camada de placeholders (que exigia que o plano acertasse a
estrutura de triplas e que o resolvedor achasse o valor certo) e deixar o LLM resolver
as referências entre passos em linguagem natural, com o histórico da cadeia no prompt.

---

## 2. Configuração e constantes

[`spmc_no_placeholder.py:34-70`](spmc_no_placeholder.py#L34-L70). Quase tudo pode ser
mudado por variável de ambiente, sem editar o código.

**LLM**

- `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`): endpoint OpenAI-compatível
  do daemon Ollama local. A chamada usa a lib `openai`; o Ollama local não exige chave
  (passa-se `api_key="ollama"` só porque o cliente exige algum valor).
- `SPMC_MODEL` → `LLM_MODEL` (default `gpt-oss:120b-cloud`). A tag `-cloud` faz o daemon
  local encaminhar a chamada para a nuvem do Ollama, então o 120B não roda na sua GPU.

**Tamanho do plano**

- `N_DEFAULT = 4`: cadeias por plano. Pode ser trocado por `--N` ou `SPMC_N`.
- `L_DEFAULT = 6`: passos por cadeia, o mesmo `max_path_length = 6` do CoRAG original.
  Pode ser trocado por `--L` ou `SPMC_L`.

**Tetos de tokens**, um por tipo de chamada:

- `PLAN_MAX_TOKENS = 8192`, `SUBQ_MAX_TOKENS = 2048`, `SUBA_MAX_TOKENS = 2048`,
  `FINAL_MAX_TOKENS = 2048`.
- São folgados porque o gpt-oss é um modelo de raciocínio: gasta tokens pensando
  **antes** de escrever a resposta, e esses tokens contam no `max_tokens`. Com teto
  baixo a resposta sai vazia ou truncada, mesmo que ela seja uma única palavra.

**Plano**

- `PLAN_JSON_RETRIES = 2`: quantas vezes **regenerar** o plano quando o JSON vem
  inválido ou fora do formato N×L. São 1 + 2 = 3 tentativas no total.

**Retrieval**

- `MINI_BASE_DIR` (default `data/mini`): onde estão `corpus/`, `e5-large-index/` e
  `questions/<task>`, gerados por `src/reproducao/construir_mini_corpus.py`.
- `E5_MODEL = "intfloat/e5-large-v2"`: tem que ser o mesmo modelo que gerou os
  embeddings do mini-corpus; se não for, os vetores da query e dos documentos ficam em
  espaços diferentes.
- `SPMC_TOP_K` → `TOP_K` (default 5): chunks recuperados por subpergunta.
- `CHUNK_CHARS = 600`: cada chunk é cortado nesse tamanho no prompt da subresposta e
  no da resposta final.
- `SPMC_FINAL_DOCS` → `FINAL_DOCS` (default 20): quantos chunks, dentre os recuperados
  pelas cadeias, entram no prompt da resposta final (ver seção 8). Com `0`, nenhum
  chunk é enviado e o bloco de passages aparece como `(none)`. O valor 20 é o mesmo
  `num_contexts` que o CoRAG original usa na resposta final.
- `SEM_RESPOSTA = "No relevant information found"`: a string que marca "não achei". É
  a mesma que o CoRAG original usa, e aparece nos prompts, nos fallbacks e no filtro da
  resposta final.

**Execução**

- `TASKS`: as 4 tarefas do mini dataset (`hotpotqa`, `2wikimultihopqa`, `musique`,
  `bamboogle`).
- `RUNS_DIR = data/spmc_runs/no_placeholder/`: onde vão os logs e as métricas.

**`RETRY_LLM`**

```python
RETRY_LLM = RetryPolicy(max_attempts=5, initial_interval=2.0,
                        retry_on=(APIConnectionError, RateLimitError, InternalServerError))
```

É a política de retry do LangGraph aplicada aos nós que chamam o LLM. Ela só cobre
**erros transitórios**: rede, 429 e 5xx. Com os defaults do LangGraph (backoff ×2 e
jitter), as esperas ficam em torno de 2 s, 4 s, 8 s e 16 s, somando até 5 tentativas.
Qualquer outra exceção (ex. `BadRequestError`, `ValueError`) sobe direto.

---

## 3. Os prompts

Os quatro prompts são de sistema e estão em inglês. O que varia por chamada vai na
mensagem `user`, montada pelos nós.

### `PLAN_PROMPT` ([l.73-209](spmc_no_placeholder.py#L73-L209))

Transforma o LLM no "Plan Generator". Suas partes:

- **Key concepts**: define *Step* (texto guia de 1-2 frases que diz o que descobrir,
  cita as entidades como estão na query, deixa clara a relação e o tipo do valor, e
  referencia passos anteriores **pelo número do passo**, nunca pelo valor) e *Chain*
  (passos em ordem de dependência).
- **Procedure**: identificar entidades/relações/tipo de resposta → fatos-ponte → montar
  N cadeias que chegam à mesma resposta por **rotas diferentes** (outra entidade-ponte,
  outra ordem, relação reformulada) → conferir as regras. Duas cadeias que pedem as
  mesmas coisas na mesma ordem são tratadas como duplicatas.
- **Rules**:
  - *Structure*: exatamente N cadeias e L passos, cada passo só com `k` e `guide_text`.
  - *Values and dependencies*: **nunca** escrever no texto guia um valor que deveria
    ser recuperado, mesmo que o modelo o conheça. É a regra central da variante: se o
    plano já traz o valor, o retrieval é pulado na prática e um erro do plano se
    propaga pela cadeia.
  - *Number of hops*: se a pergunta precisa de menos de L hops, os passos extras servem
    para verificar ou desambiguar, **antes** do último; nunca repetir passos.
  - *Final step*: o passo L produz a resposta final; em perguntas de comparação ou
    sim/não, os passos anteriores buscam cada atributo e o último compara.
- **Output**: JSON puro com `overview`, `entities`, `relations`, `answer_type` e
  `chains` (cada uma com `id`, `description` e `steps`).
- **Exemplo** (Titanic, 2 cadeias × 3 passos): mostra o formato e que o nome do
  diretor nunca aparece no plano.

O prompt é preenchido com `PLAN_PROMPT.format(N=..., L=...)`. Por isso as chaves
literais do JSON de exemplo estão escritas como `{{ }}`; o `.format` as converte em
`{ }`.

### `SUBQ_PROMPT` ([l.211-231](spmc_no_placeholder.py#L211-L231))

Gera **uma** subpergunta para o passo atual. Regras principais:

- pedir só o que o texto guia pede (um fato), sem antecipar passos seguintes;
- ser **autocontida**: trocar "o diretor identificado no passo 1" pela subresposta do
  passo 1 que está no contexto. É isso que substitui o mecanismo de placeholders;
- se a subresposta referenciada for `SEM_RESPOSTA`, formular a subpergunta com o que a
  pergunta original diz;
- não responder a subpergunta.

### `SUBA_PROMPT` ([l.233-241](spmc_no_placeholder.py#L233-L241))

Responde a subpergunta **usando só os passages recuperados**, com uma frase curta ou um
nome de entidade. Se os passages não contêm a resposta, devolve exatamente
`SEM_RESPOSTA`. É uma f-string, então `{SEM_RESPOSTA}` é interpolado quando o módulo é
importado.

### `FINAL_PROMPT` ([l.243-268](spmc_no_placeholder.py#L243-L268))

Produz a resposta final a partir de **duas fontes de evidência**:

1. **Retrieved passages**: os chunks mais relevantes para a pergunta original, dentre
   todos os recuperados durante a execução das cadeias (seleção na seção 8);
2. **Reasoning chains**: as cadeias, só com os passos que acharam algo.

O prompt define o papel de cada fonte:

- as cadeias guiam o **raciocínio multi-hop** (qual entidade leva a qual);
- os passages servem para **confirmar, corrigir ou completar** esse raciocínio;
- as subrespostas foram geradas por um LLM e podem estar erradas: se uma subresposta
  contradiz os passages, valem os passages. Um passage também pode trazer um fato que
  nenhuma cadeia extraiu. O aviso segue o prompt final do CoRAG ("intermediate answers
  ... may not always be accurate");
- continua proibido usar conhecimento externo; a resposta é curta, ou `SEM_RESPOSTA`.

Também é uma f-string.

Os três prompts de execução pedem **resposta curta, sem explicação**. Isso importa
porque o EM/F1 compara a predição literal com o gabarito.

---

## 4. Schema Pydantic do plano

[l.274-333](spmc_no_placeholder.py#L274-L333). O JSON do plano é validado contra três
modelos aninhados:

```
Plano
├── overview: str
├── entities: list[str]
├── relations: list[str]
├── answer_type: str = ""        # opcional: o prompt pede, mas não quebra se faltar
└── chains: list[Cadeia]
        ├── id: int
        ├── description: str
        └── steps: list[Passo]
                ├── id: int      # aceita "id" OU "k" (AliasChoices)
                └── guide_text: str (min_length=1)
```

- **`Passo`**: o prompt pede `"k"`, mas o código usa `id`. O
  `validation_alias=AliasChoices("id", "k")` aceita as duas chaves, e o
  `populate_by_name=True` permite construir o objeto por `id`. Ao serializar com
  `model_dump()` (no log), a chave sai como `id`. `guide_text` vazio é rejeitado.
- **`Cadeia`** e **`Plano`**: campos simples. O Pydantic rejeita campos obrigatórios
  ausentes e tipos errados. Campos extras são ignorados (o default do Pydantic), então
  um LLM que devolva triplas a mais não quebra a validação.

### `_checar_formato(plano, N, L)` ([l.324](spmc_no_placeholder.py#L324))

N e L só são conhecidos em tempo de execução, então essa verificação fica fora do
schema. A função confere que:

1. há exatamente `N` cadeias;
2. em cada cadeia, os ids dos passos são **exatamente** `[1, 2, ..., L]`, em ordem.
   Isso pega passo faltando, passo sobrando, id repetido e ordem trocada.

Devolve uma string descrevendo o problema (que vai para o log de erro e para a
exceção final), ou `None` se o plano estiver certo.

---

## 5. Chamada ao LLM

### `_extrair_json(texto)` ([l.341](spmc_no_placeholder.py#L341))

Mesmo com `response_format=json_object`, o modelo às vezes embrulha o JSON em cercas
```` ``` ```` ou põe texto antes/depois. A função:

1. recorta do **primeiro** `{` até o **último** `}`;
2. remove vírgulas sobrando antes de `}` ou `]` (`{"a": 1,}` → `{"a": 1}`), um erro
   comum de LLMs que o JSON estrito não aceita;
3. levanta `ValueError` se não houver `{...}` na resposta.

### `_chamar_llm(system, user, *, max_tokens, json_mode=False)` ([l.350](spmc_no_placeholder.py#L350))

É a única função do arquivo que fala com o LLM.

- Cria um cliente `OpenAI` apontando para o Ollama, com **`max_retries=0`**. O retry
  interno da lib `openai` é desligado de propósito: quem retenta é o `RetryPolicy` do
  nó LangGraph, então os dois mecanismos não se multiplicam.
- Manda duas mensagens (`system` + `user`) com **`temperature=0`**, para ter o
  comportamento mais determinístico possível.
- `json_mode=True` (só no plano) adiciona `response_format={"type": "json_object"}`.
- Se `finish_reason == "length"`, **só avisa** no stderr que a resposta foi truncada e
  que é preciso aumentar o teto; não levanta erro.
- Devolve o conteúdo com `strip()`, ou `""` se vier `None`. Cada nó decide o que fazer
  com string vazia (ver os fallbacks nas seções 7 e 8).

---

## 6. Retrieval E5 in-process sobre o mini-corpus

[l.376-438](spmc_no_placeholder.py#L376-L438). O retrieval não usa o servidor HTTP
(`start_e5_server.sh`): o encoder E5 e os embeddings do mini-corpus são carregados
**dentro do próprio processo**.

### Cache global `_MINI` + `_MINI_LOCK`

```python
_MINI = {"corpus": None, "emb": None, "encode": None}
_MINI_LOCK = threading.Lock()
```

O carregamento é **lazy**: só acontece na primeira chamada a `_retrieve`. Como as N
cadeias rodam em threads paralelas, várias podem chegar ao primeiro retrieve ao mesmo
tempo. O lock garante que só uma carrega e as outras esperam; depois disso, o
`if _MINI["corpus"] is not None: return` faz as chamadas seguintes saírem na hora.

### `_carregar_mini()` ([l.382](spmc_no_placeholder.py#L382))

1. Confere se `data/mini/corpus` e `data/mini/e5-large-index/e5-large-shard-0.pt`
   existem. Se não existem, levanta `RuntimeError` com o comando que gera o
   mini-corpus.
2. Carrega o tokenizer e o modelo `intfloat/e5-large-v2` em **CPU, float32**, em modo
   `eval`.
3. Define a função `encode` com a **mesma convenção** usada para gerar o índice:
   - prefixo `"query: "` em cada texto (o E5 foi treinado com prefixos `query:` e
     `passage:`);
   - tokenização com truncagem em 512 tokens;
   - **mean pooling mascarado**: a média dos hidden states só sobre os tokens reais
     (a máscara zera o padding; o `clamp` evita divisão por zero);
   - normalização **L2**, para que o produto interno vire similaridade de cosseno.
4. Carrega a matriz de embeddings do mini-corpus (`torch.load`, `weights_only=True`)
   e o `Dataset` do corpus (`load_from_disk`).

### `_retrieve(subpergunta, k)` ([l.418](spmc_no_placeholder.py#L418))

1. Codifica a subpergunta → vetor `[1, 1024]`.
2. Faz `vetor @ emb.T` → um score por documento do mini-corpus (**busca exaustiva**; o
   mini-corpus é pequeno, então não precisa de FAISS).
3. `torch.topk` pega os `k` maiores (ou menos, se o corpus tiver menos de `k` docs).
4. Para cada índice devolve `{idx, doc_id, score, title, contents}`:
   - `idx` é o índice **local** no mini-corpus, a linha em `emb` e em `corpus`. É ele
     que o histórico guarda para, na resposta final, reaproveitar texto e embedding do
     chunk sem refazer o retrieval;
   - `doc_id` vem de `orig_doc_id`, o id do documento no corpus KILT completo, para
     rastrear de onde o chunk veio; se o campo não existir, usa o índice local.

---

## 7. Subgrafo de uma cadeia

[l.445-585](spmc_no_placeholder.py#L445-L585). Cada cadeia do plano roda neste
subgrafo, com um **estado próprio**, então o que uma cadeia descobre não vaza para as
outras.

### `CadeiaState` ([l.445](spmc_no_placeholder.py#L445))

| Campo | Tipo | Quem escreve | Observação |
|---|---|---|---|
| `questao` | `str` | entrada (Send) | pergunta original |
| `plano` | `Plano` | entrada (Send) | usado para overview/entidades/relações |
| `cadeia` | `Cadeia` | entrada (Send) | só a cadeia desta execução |
| `passo_atual` | `int` | `no_executar_cadeia` (0), `gerar_subresposta` (+1) | índice 0..L-1 em `cadeia.steps` |
| `subpergunta_atual` | `str` | `gerar_subpergunta` | sobrescrito a cada passo |
| `chunks_atual` | `list[dict]` | `retrieve` | sobrescrito a cada passo |
| `historico` | `Annotated[list[dict], operator.add]` | `gerar_subresposta` | **acumula**: cada nó devolve `[registro]` e o reducer concatena |

A diferença entre `passo_atual` e `historico` vem do reducer. Sem reducer, o LangGraph
**sobrescreve** o valor com o que o nó devolveu. Com `operator.add`, ele **concatena**
a lista nova à existente. Por isso `gerar_subresposta` devolve só o registro novo
(`{"historico": [registro]}`) e não a lista inteira.

### `_contexto_acumulado(plano, historico)` ([l.471](spmc_no_placeholder.py#L471))

Monta o bloco de contexto (em inglês, porque vai para o LLM):

```
Plan overview: <plano.overview>
Key entities: <e1>; <e2>
Relations: <r1>; <r2>
Previous steps in this chain:
Step 1:
  sub-question: ...
  sub-answer: ...
Step 2:
  ...
```

No primeiro passo, "Previous steps" é `(none yet)`. Só entram subpergunta e
subresposta dos passos anteriores; os chunks recuperados **não** entram, o que mantém
o prompt pequeno.

### Nó `gerar_subpergunta` ([l.493](spmc_no_placeholder.py#L493))

- Pega o passo atual: `state["cadeia"].steps[state["passo_atual"]]`.
- Mensagem `user` = pergunta original + contexto acumulado + `guide_text` do passo.
- Chama o LLM com `SUBQ_PROMPT`.
- **Fallback**: se o LLM devolver vazio (ex. gastou todo o teto raciocinando), usa o
  próprio `guide_text` como subpergunta, para não buscar com uma query vazia.
- Saída: `{"subpergunta_atual": ...}`.

### Nó `retrieve` ([l.519](spmc_no_placeholder.py#L519))

Chama `_retrieve(subpergunta_atual, TOP_K)`. Saída: `{"chunks_atual": [...]}`. Não
tem `retry_policy`, porque é local e não sofre de erro transitório de rede.

### Nó `gerar_subresposta` ([l.533](spmc_no_placeholder.py#L533))

- Formata os chunks como passages numerados, cada um cortado em `CHUNK_CHARS`:
  ```
  [1] <title>: <contents[:600]>

  [2] <title>: <contents[:600]>
  ...
  ```
- Mensagem `user` = passages + subpergunta. Note que a pergunta original e o histórico
  **não** entram aqui: a subresposta depende só dos chunks, o que força a resposta a
  vir do retrieval.
- Chama o LLM com `SUBA_PROMPT`; se vier vazio, usa `SEM_RESPOSTA`.
- Monta o registro do passo:
  ```python
  {"id": passo.id, "guide_text": ..., "subpergunta": ...,
   "chunks": [títulos dos chunks], "docs": [idx dos chunks], "subresposta": ...}
  ```
  Só os **títulos** e os **índices** (`docs`) dos chunks são guardados, não o texto,
  para o log não ficar enorme. Os `docs` alimentam a seleção de chunks da resposta
  final (seção 8).
- Saída: `{"historico": [registro], "passo_atual": passo_atual + 1}`. É aqui que o
  passo avança.

### Roteador `_proximo_passo` ([l.564](spmc_no_placeholder.py#L564))

Depois de `gerar_subresposta`: se `passo_atual < L`, volta para `gerar_subpergunta`;
senão vai para `END`. Como `passo_atual` começa em 0 e sobe 1 por volta, o laço dá
exatamente L voltas.

### `construir_grafo_cadeia()` e `_GRAFO_CADEIA` ([l.568-585](spmc_no_placeholder.py#L568-L585))

Liga os nós conforme o diagrama da seção 0 e compila. O subgrafo é compilado **uma
única vez**, quando o módulo é importado (`_GRAFO_CADEIA`), e reusado por todas as
cadeias e perguntas. Um grafo compilado não guarda estado entre invocações, então esse
reuso entre threads é seguro.

---

## 8. Grafo de nível superior

[l.592-787](spmc_no_placeholder.py#L592-L787).

### `PlanoState` ([l.592](spmc_no_placeholder.py#L592))

| Campo | Quem escreve | Observação |
|---|---|---|
| `questao`, `N`, `L` | entrada do `invoke` | |
| `plano_bruto` | `gerar_plano` | JSON cru da tentativa aceita |
| `plano` | `gerar_plano` | objeto `Plano` validado |
| `cadeias_exec` | cada `executar_cadeia` | `Annotated[list[dict], operator.add]`: cada cadeia contribui com 1 item |
| `docs_finais` | `gerar_resposta_final` | chunks que entraram no prompt final (`doc_id`, `title`, `score`), em ordem de relevância |
| `resposta_final` | `gerar_resposta_final` | |

### Nó `gerar_plano` ([l.621](spmc_no_placeholder.py#L621))

```
para tentativa em 1..(PLAN_JSON_RETRIES + 1):
    texto = LLM(PLAN_PROMPT(N, L), questao, json_mode=True)
    tenta: plano = Plano.model_validate_json(_extrair_json(texto))
           erro  = _checar_formato(plano, N, L)
    se deu ValidationError/ValueError: erro = descrição
    se erro é None: devolve {plano_bruto, plano}
    senão: loga "invalid plan on attempt i" e tenta de novo
loga o JSON cru da última tentativa; levanta ValueError
```

A mensagem `user` do plano é **só a pergunta**; tudo o mais está no prompt de sistema.

Aqui há **dois níveis de retry**, independentes:

1. **Loop interno** (`PLAN_JSON_RETRIES`): cobre **conteúdo** ruim (JSON malformado,
   schema errado, N×L errado). Regenera com o mesmo prompt.
2. **`RETRY_LLM`** no nó: cobre **erros transitórios** de rede/API. Se
   `_chamar_llm` levanta `APIConnectionError` no meio do loop, a exceção sai do nó e o
   LangGraph **reexecuta o nó inteiro**, começando o loop de novo do zero.

O `ValueError` final ("plano inválido após 3 tentativas") **não** está em
`retry_on`, então não é retentado pelo LangGraph: sobe até `rodar_mini_dataset`, que
marca a pergunta como falha. No pior caso, o plano consome 5 × 3 = 15 chamadas ao LLM.

### `_distribuir_cadeias` ([l.653](spmc_no_placeholder.py#L653)): o fan-out

```python
[Send("executar_cadeia", {"questao": ..., "plano": ..., "cadeia": c}) for c in plano.chains]
```

É usado como **aresta condicional** saindo de `gerar_plano`. Cada `Send` cria uma
execução separada do nó `executar_cadeia`, com um **payload próprio** em vez do
`PlanoState`. O LangGraph roda todas as `Send` do mesmo superstep **em paralelo**
(num pool de threads), então as N cadeias executam ao mesmo tempo. O payload leva só o
necessário: pergunta, plano completo e a cadeia daquela execução.

### Nó `executar_cadeia` ([l.662](spmc_no_placeholder.py#L662))

1. Recebe o payload da `Send` (um `dict`, não o `PlanoState`).
2. Invoca o subgrafo `_GRAFO_CADEIA` com o estado inicial
   `{questao, plano, cadeia, passo_atual: 0, historico: []}`.
3. Passa `recursion_limit = 3·L + 5`: cada passo são 3 supersteps (3 nós), mais uma
   folga. Versões antigas do LangGraph tinham default 25, que estouraria com L ≥ 8; a
   instalada no `.venv` usa 10007 (`langgraph/_internal/_config.py`). Calcular o limite a
   partir de L funciona nas duas e serve de trava: se o laço não parar por algum bug
   no roteador, o subgrafo falha com `GraphRecursionError` logo depois do passo L, em vez
   de girar milhares de vezes.
4. Devolve `{"cadeias_exec": [{"id", "description", "historico"}]}`. Graças ao reducer
   `operator.add`, os N resultados paralelos são **concatenados** em
   `PlanoState["cadeias_exec"]`, na ordem em que as cadeias terminam (por isso quem lê
   depois ordena por `id`).

Este nó não tem `retry_policy` própria: os retries acontecem dentro do subgrafo, nó a
nó.

### Barreira: quando a resposta final dispara

A aresta `executar_cadeia → gerar_resposta_final` é normal. Como todas as `Send` rodam
no mesmo superstep, o LangGraph só avança para o superstep seguinte quando **todas** as
N cadeias terminam. Assim, `gerar_resposta_final` roda **uma vez**, já com
`cadeias_exec` completo.

### `_formatar_cadeias(cadeias_exec)` ([l.685](spmc_no_placeholder.py#L685))

Monta o texto das cadeias para o prompt final:

- ordena as cadeias por `id`;
- **descarta** os passos cuja subresposta é `SEM_RESPOSTA` (comparação sem diferenciar
  maiúsculas, após `strip()`), porque só trariam ruído;
- se uma cadeia não achou nada em nenhum passo, escreve `(no step found an answer)`.

```
Chain 1:
  Step 1:
    sub-question: ...
    sub-answer: ...
  Step 3:
    ...

Chain 2:
  (no step found an answer)
```

O `description` de cada cadeia **não** entra no prompt final; fica só no log.

### `_selecionar_chunks_finais(questao, cadeias_exec, m)` ([l.701](spmc_no_placeholder.py#L701))

Escolhe os chunks que entram no prompt final. Com N = 4, L = 6 e top-5, as cadeias
recuperam 120 chunks por pergunta, muitos repetidos. Mandar todos incharia o prompt e
diluiria o que importa, então a função seleciona os `m` (= `FINAL_DOCS`, 20) mais
relevantes:

1. **Pool**: percorre os `docs` de todos os passos de todas as cadeias, **inclusive
   os passos que deram `SEM_RESPOSTA`**. É justamente onde pode haver evidência que a
   subresposta não aproveitou. O pool é deduplicado por `idx`, e a função conta
   quantas vezes cada chunk foi recuperado (`freq`).
2. **Relevância**: codifica a **pergunta original** com o mesmo encoder E5 e faz o
   produto interno com os embeddings do pool (`emb[idxs]`, já em memória; nada é
   recalculado para os chunks). É a mesma medida do retrieval, só que contra a
   pergunta inteira em vez da subpergunta.
3. **Ordenação**: score decrescente; empate desfeito por `freq`, porque um chunk
   trazido por várias cadeias tende a ser central.
4. Devolve os top-`m` como `{doc_id, score, title, contents[:CHUNK_CHARS]}`.

Se `m <= 0` ou o pool está vazio, devolve `[]`. Se o pool tiver menos de `m` chunks
únicos, entram todos. Isso é comum com planos pequenos: no teste com N = 2 e L = 3, o
pool teve 7 a 12 chunks.

### Nó `gerar_resposta_final` ([l.739](spmc_no_placeholder.py#L739))

1. Seleciona os chunks com `_selecionar_chunks_finais(questao, cadeias_exec, FINAL_DOCS)`.
2. Monta a mensagem `user`:
   ```
   Original question: ...

   Retrieved passages:
   [1] <title>: <contents[:600]>

   [2] ...

   Reasoning chains:
   <_formatar_cadeias(...)>

   Answer the original question.
   ```
   Sem chunks (`FINAL_DOCS=0`), o bloco de passages vira `(none)`.
3. Chama o LLM com `FINAL_PROMPT`; se vier vazio, usa `SEM_RESPOSTA`.
4. Devolve `resposta_final` e `docs_finais` (`doc_id`, `title`, `score` de cada chunk
   usado, sem o texto), que vão para o log.

Os passages vêm antes das cadeias, como no prompt final do CoRAG (documentos, depois
consultas e respostas intermediárias). Isso deixa as cadeias, que guiam o raciocínio,
mais perto da instrução final.

### `construir_grafo()` ([l.772](spmc_no_placeholder.py#L772))

```python
g.add_edge(START, "gerar_plano")
g.add_conditional_edges("gerar_plano", _distribuir_cadeias, ["executar_cadeia"])
g.add_edge("executar_cadeia", "gerar_resposta_final")
g.add_edge("gerar_resposta_final", END)
```

Para invocar: `construir_grafo().invoke({"questao": ..., "N": ..., "L": ...})`.

---

## 9. Fluxo de execução passo a passo

Rastreio completo de uma execução, do comando até o log.

### 9.1 Linha de comando → `rodar_mini_dataset`

```bash
python src/SPMC/spmc_no_placeholder.py --task hotpotqa --limit 2 --N 2 --L 3
```

1. **Import do módulo**: as constantes são lidas do ambiente e `_GRAFO_CADEIA` é
   compilado. O modelo E5 **ainda não** é carregado.
2. `__main__` ([l.910](spmc_no_placeholder.py#L910)) lê os argumentos e monta a lista
   de tarefas (uma só, ou as 4 se `--task all`).
3. Para cada tarefa chama `rodar_mini_dataset(task, N, L, limit)`, que:
   - carrega `data/mini/questions/<task>` e, se houver `--limit`, pega as primeiras
     perguntas;
   - cria o `run_id` `<task>_N<N>_L<L>_k<TOP_K>_d<FINAL_DOCS>_<timestamp>` e abre
     `<run_id>.jsonl`;
   - compila o grafo de topo (`construir_grafo()`);
   - para cada pergunta, chama `grafo.invoke({"questao", "N", "L"})` (seções 9.2-9.5).

### 9.2 Plano

`gerar_plano` recebe `{questao, N, L}` e devolve um `Plano` com N cadeias × L passos.
Exemplo real (hotpotqa, N = 2, L = 3), pergunta *"What nationality was Oliver Reed's
character in the film Royal Flash?"*:

```
overview: Determine the nationality of the character portrayed by Oliver Reed in Royal Flash.
entities: [Oliver Reed, Royal Flash]   answer_type: nationality
Chain 1: 1) Identify the character name portrayed by Oliver Reed in the film Royal Flash.
         2) Find the nationality of the character identified in step 1.
         3) Provide the nationality found in step 2 as the final answer.
Chain 2: 1) Retrieve the cast list of the film Royal Flash, including character names ...
         2) From the cast list obtained in step 1, determine which character is played by Oliver Reed.
         3) Find the nationality of the character identified in step 2 and present it as the final answer.
```

### 9.3 Fan-out: N cadeias em paralelo

`_distribuir_cadeias` emite N `Send`s. Cada `executar_cadeia` roda numa thread e invoca
o subgrafo com `passo_atual = 0` e `historico = []`.

### 9.4 Dentro de uma cadeia: L voltas do laço

Para o passo `k` (índice `passo_atual = k-1`):

1. **`gerar_subpergunta`**: vê pergunta + overview/entidades/relações + passos 1..k-1 +
   `guide_text` do passo k. Resolve referências do tipo "o personagem do passo 1"
   usando a subresposta do passo 1.
2. **`retrieve`**: na primeira chamada do processo inteiro, carrega o E5 e o
   mini-corpus (sob lock). Devolve os top-k chunks.
3. **`gerar_subresposta`**: vê só os chunks + a subpergunta. Anexa o registro ao
   `historico` e faz `passo_atual += 1`.
4. **`_proximo_passo`**: volta ao item 1 ou termina.

Continuando o exemplo, a cadeia 1 ficou assim:

| Passo | Subpergunta gerada | Títulos recuperados | Subresposta |
|---|---|---|---|
| 1 | What is the name of the character portrayed by Oliver Reed in the film Royal Flash? | Royal Flash (film) ×3, The Prisoner of Zenda, Oliver Reed | Otto von Bismarck |
| 2 | What was the nationality of **Otto von Bismarck**, the character portrayed by Oliver Reed ...? | Royal Flash (film) ×3, Royal Flash, The Prisoner of Zenda | No relevant information found |
| 3 | What nationality was Otto von Bismarck, the character ...? | (os mesmos) | No relevant information found |

O passo 2 mostra o mecanismo sem placeholder: "the character identified in step 1"
virou "Otto von Bismarck" na subpergunta. O exemplo também mostra dois problemas: o
passo 3 do plano ("Provide ... as the final answer") só repete o passo 2 e gasta um
retrieve à toa; e o mesmo documento aparece várias vezes no top-k (chunks diferentes
de "Royal Flash (film)").

### 9.5 Fan-in e resposta final

Quando as N cadeias terminam, `cadeias_exec` tem N itens. `gerar_resposta_final`
junta os chunks recuperados pelas cadeias, seleciona os até 20 mais similares à
pergunta original, formata as cadeias (sem os passos `SEM_RESPOSTA`) e pede a resposta
ao LLM com passages + cadeias.

No exemplo, o pool teve só 7 chunks únicos (N = 2 e L = 3 recuperam 30 chunks, quase
todos repetidos), então todos entraram: `Royal Flash (film)` ×3, `Oliver Reed`, `The
Prisoner of Zenda`, `Royal Flash` e `List of The Flash characters`. A predição continuou `No relevant information found` (gabarito:
`Prussian`). Nenhum passo recuperou um passage que diga a nacionalidade do personagem, e
os chunks do prompt final só podem vir do que as cadeias recuperaram.

O `invoke` devolve o `PlanoState` final: `plano`, `plano_bruto`, `cadeias_exec`,
`docs_finais` e `resposta_final`.

### 9.6 Pós-processamento por pergunta

De volta em `rodar_mini_dataset`:

1. monta o registro com `prediction`, `plano.model_dump()`, `cadeias_exec` ordenado
   por id, `docs_finais` e `erro=None`;
2. calcula EM/F1 **dessa pergunta** com `_metricas`;
3. escreve a linha no `.jsonl` e dá `flush()`;
4. imprime `[i/total] pergunta -> predição (gold, em, f1, tempo)`.

Se qualquer exceção escapar do `invoke` (plano inválido, retries esgotados, erro de
retrieval...), o `except` registra `prediction = SEM_RESPOSTA`, `plano = None`,
`cadeias_exec = []`, `docs_finais = []` e a mensagem em `erro`, soma 1 em `n_falhas` e **segue para a
próxima pergunta**.

### 9.7 Fim da tarefa

Calcula EM/F1 agregados, grava `<run_id>_metrics.json`, anexa uma linha em
`index.jsonl` e imprime o resumo. Com `--task all`, no fim imprime uma tabela com as 4
tarefas.

### Custo de chamadas ao LLM por pergunta

```
1 (plano) + N · L · 2 (subpergunta + subresposta) + 1 (final)
```

Com N = 4 e L = 6: **50 chamadas**, mais eventuais regenerações do plano e retries
transitórios. Os 20 chunks não acrescentam chamadas: a seleção é um encode E5 local da
pergunta, e o prompt final cresce cerca de 20 × 600 caracteres (~3 mil tokens).

Como referência, **antes** dos chunks no prompt final, o run
`hotpotqa_N4_L6_k5_20260928-171836` (30 perguntas) levou 1673 s, cerca de 56 s por
pergunta, com EM = 53,3 e F1 = 63,5. Para comparar, rode com `SPMC_FINAL_DOCS=20`
(default) e `SPMC_FINAL_DOCS=0`.

---

## 10. Avaliação no mini dataset

### `_metricas(answers, preds)` ([l.794](spmc_no_placeholder.py#L794))

Reusa as métricas oficiais do repo, `compute_metrics_dict(..., eval_metrics="em_and_f1")`
de [`src/inference/metrics.py`](../inference/metrics.py), com a normalização SQuAD
(minúsculas, sem pontuação e sem artigos). `answers` é uma lista de listas, porque uma
pergunta pode ter mais de um gabarito aceito. Os valores saem em **porcentagem** (0-100).

Detalhes:

- adiciona `src/` ao `sys.path` na hora (import tardio), para o arquivo rodar sem
  `PYTHONPATH=src`;
- importar `inference.metrics` puxa o `logger_config` do repo, que liga o logging raiz
  em INFO. Isso faria o `httpx` logar cada request ao LLM, então o nível do `httpx` é
  baixado para WARNING.

É chamada duas vezes: por pergunta (dentro do loop) e no agregado (no fim).

### `rodar_mini_dataset(task, N, L, limit=None)` ([l.806](spmc_no_placeholder.py#L806))

Arquivos gerados em `data/spmc_runs/no_placeholder/`:

| Arquivo | Conteúdo | Escrita |
|---|---|---|
| `<run_id>.jsonl` | 1 linha por pergunta | `flush()` a cada pergunta; se o processo cair, as perguntas já feitas ficam salvas |
| `<run_id>_metrics.json` | EM/F1 agregados + config do run | no fim da tarefa |
| `index.jsonl` | 1 linha por run, append-only | no fim da tarefa; serve para comparar runs |

Formato de uma linha do `.jsonl`:

```jsonc
{
  "query_id": "hotpotqa_dev_4264",
  "query": "What nationality was Oliver Reed's character in the film Royal Flash?",
  "answers": ["Prussian"],
  "prediction": "No relevant information found",
  "plano": { "overview": ..., "entities": [...], "relations": [...],
             "answer_type": ..., "chains": [{ "id", "description", "steps": [{ "id", "guide_text" }] }] },
  "cadeias_exec": [
    { "id": 1, "description": ...,
      "historico": [{ "id", "guide_text", "subpergunta", "chunks": [títulos],
                      "docs": [idx locais], "subresposta" }] }
  ],
  "docs_finais": [{ "doc_id": 32961278, "title": "Royal Flash (film)", "score": 0.8439 }, ...],
  "erro": null,          // ou "TipoDaExcecao: mensagem"
  "em": 0.0, "f1": 0.0,  // desta pergunta
  "tempo_s": 25.7
}
```

Formato do `_metrics.json` (e de cada linha de `index.jsonl`):

```json
{
  "run_id": "hotpotqa_N4_L6_k5_d20_20260929-170000", "timestamp": "20260929-170000",
  "task": "hotpotqa", "N": 4, "L": 6, "top_k": 5, "final_docs": 20,
  "model": "gpt-oss:120b-cloud",
  "n_perguntas": 30, "n_falhas": 0, "em": 53.333, "f1": 63.541,
  "tempo_total_s": 1673.0,
  "arquivo": "data/spmc_runs/no_placeholder/hotpotqa_N4_L6_k5_d20_20260929-170000.jsonl"
}
```

(Valores ilustrativos. Runs anteriores a esta mudança não têm `_d<FINAL_DOCS>` no
`run_id` nem `final_docs` nas métricas: foram feitos sem chunks no prompt final.)

As perguntas que falham entram no agregado com predição `SEM_RESPOSTA`, ou seja,
**contam como erro** (EM = 0). O `n_falhas` mostra quantas foram.

### Uso

```bash
# teste rápido: 2 perguntas, plano pequeno
python src/SPMC/spmc_no_placeholder.py --task hotpotqa --limit 2 --N 2 --L 3

# configuração padrão (N=4, L=6) numa tarefa
python src/SPMC/spmc_no_placeholder.py --task musique

# mesma configuração sem chunks no prompt final (para comparar)
SPMC_FINAL_DOCS=0 python src/SPMC/spmc_no_placeholder.py --task musique

# as 4 tarefas, com top-10 e outro modelo do Ollama
SPMC_TOP_K=10 SPMC_MODEL=gpt-oss:20b python src/SPMC/spmc_no_placeholder.py --task all
```

Pré-requisitos: daemon Ollama rodando (com login, para os modelos `-cloud`) e o
mini-corpus gerado em `data/mini/`.

---

## 11. Limitações e pontos de atenção

1. **Uma cadeia com erro derruba a pergunta inteira.** Se um nó do subgrafo esgota os
   retries ou levanta um erro não transitório, a exceção sobe por `executar_cadeia`
   (que não tem retry nem `try/except`) e interrompe o `invoke`. As outras N-1 cadeias,
   mesmo que já tenham terminado, são perdidas e a pergunta vira `SEM_RESPOSTA`.
2. **Plano inválido não é retentado pelo LangGraph.** O `ValueError` final de
   `gerar_plano` está fora de `retry_on`. Isso é intencional (já houve 3 tentativas),
   mas faz a pergunta falhar sem plano.
3. **O filtro de `SEM_RESPOSTA` é por igualdade exata.** `_formatar_cadeias` só
   descarta subrespostas iguais a `"no relevant information found"` (ignorando
   maiúsculas e espaços nas pontas). Uma variação como `"No relevant information
   found."` ou `"Not mentioned in the passages"` passa pelo filtro e chega ao prompt
   final como se fosse um fato.
4. **Truncamento só gera aviso.** Com `finish_reason == "length"` o código segue com o
   que veio. Se veio vazio, cada nó cai no seu fallback: subpergunta → `guide_text`,
   subresposta e resposta final → `SEM_RESPOSTA`, plano → regenera. Se veio **cortado**
   mas não vazio, o texto parcial é usado como está.
5. **Passos redundantes no plano.** Quando a pergunta tem menos hops que L, o plano às
   vezes cria passos do tipo "Provide X as the final answer", que repetem o passo
   anterior e gastam um retrieve e duas chamadas de LLM (visto no exemplo da seção 9.4).
   O prompt pede para usar os passos extras para verificação, mas o modelo nem sempre
   obedece.
6. **Top-k com documentos repetidos.** O mini-corpus é dividido em chunks, e vários
   chunks do mesmo documento podem ocupar o top-k (ex. `Royal Flash (film)` ×3),
   reduzindo a diversidade dos passages.
7. **As cadeias não compartilham descobertas.** Isso é por design (independência entre
   rotas), mas se uma cadeia acha a entidade-ponte, as outras não se beneficiam; só a
   resposta final cruza as cadeias.
8. **A resposta final só vê os chunks que as cadeias recuperaram** (mitigada). Antes ela
   via só as subrespostas; agora recebe também os 20 chunks mais relevantes do pool, o
   que recupera fatos mal extraídos em `gerar_subresposta`. Continua valendo:
   - se nenhuma cadeia recuperou o passage com a resposta, os chunks não ajudam (caso
     do Royal Flash). No run hotpotqa N4 L6 sem chunks, 9 dos 14 erros eram desse tipo;
   - a relevância é medida contra a **pergunta original**. Um chunk de um hop
     intermediário, que é pouco parecido com a pergunta inteira, pode ficar fora dos 20
     mesmo sendo necessário;
   - com mais contexto, o modelo pode responder direto de um passage que casa com a
     pergunta e ignorar o raciocínio das cadeias. O ganho real precisa ser medido
     comparando `SPMC_FINAL_DOCS=20` e `0`.
9. **Campos do plano não usados na execução.** `answer_type` e o `description` das
   cadeias não entram em nenhum prompt de execução nem no prompt final; ficam só no
   log.
10. **O log não guarda o texto dos chunks.** Cada passo registra títulos e índices
    locais (`docs`), e a resposta final registra `doc_id`/título/score dos chunks usados
    (`docs_finais`). Dá para reconstruir o texto pelo índice no mini-corpus, mas o
    `.jsonl` sozinho não mostra qual trecho sustentou cada subresposta, nem os scores do
    retrieval de cada passo.
11. **Custo e escala.** São cerca de 50 chamadas por pergunta com N = 4 e L = 6; o E5
    roda em CPU e a busca é exaustiva, o que funciona para o mini-corpus mas não para o
    corpus KILT completo (para esse caso, o repo tem o índice FAISS de
    `src/search/construir_indice_faiss.py`, usado pelo `E5Searcher`). O cliente
    `OpenAI` é recriado a cada chamada, sem reuso de conexão.
