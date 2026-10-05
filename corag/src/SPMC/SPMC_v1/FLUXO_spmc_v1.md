# Fluxo de execução do `spmc_v1.py`

Este documento segue **o caminho de uma pergunta** pelo [`spmc_v1.py`](spmc_v1.py), do
plano até a resposta final, mostrando o que cada nó lê, decide e escreve no estado.
Para a explicação por componente (constantes, prompts, schema, retrieval), veja o
[`CODIGO.md`](../CODIGO.md), que documenta o mesmo código quando ele ainda se chamava
`src/SPMC/spmc.py`.

**Resumo.** A Gemini gera um **plano** com N cadeias de L passos. Cada passo tem um
texto guia e uma **tripla-template** `<sujeito; predicado; objeto>` com
**placeholders** `[S_k: tipo: ??]` nas posições que só serão conhecidas depois do passo
k. As N cadeias rodam em **threads paralelas**. Em cada passo, a cadeia:

1. substitui os placeholders já descobertos;
2. gera uma subpergunta;
3. recupera top-k chunks;
4. extrai **triplas de evidência**;
5. responde a subpergunta **só com essas triplas**.

Se não houver resposta, tenta uma subpergunta diferente. No fim, os passos respondidos
de todas as cadeias são agregados e o CoRAG-8B gera a resposta final.

---

## Sumário

0. [Visão geral](#0-visão-geral)
1. [Fase 1 — Plano](#1-fase-1--plano)
2. [Fase 2 — Distribuição das cadeias em threads](#2-fase-2--distribuição-das-cadeias-em-threads)
3. [Fase 3 — Execução de um passo (hop)](#3-fase-3--execução-de-um-passo-hop)
4. [Propagação de placeholders ao longo de uma cadeia](#4-propagação-de-placeholders-ao-longo-de-uma-cadeia)
5. [Fase 4 — Agregação e resposta final](#5-fase-4--agregação-e-resposta-final)
6. [Custo por pergunta](#6-custo-por-pergunta)
7. [Erros ao longo do fluxo](#7-erros-ao-longo-do-fluxo)
8. [O estado (`SPMCState`) ao longo do fluxo](#8-o-estado-spmcstate-ao-longo-do-fluxo)
9. [Como executar](#9-como-executar)

---

## 0. Visão geral

O fluxo usa **dois grafos LangGraph** e um `ThreadPoolExecutor` entre eles.

**Grafo de nível superior** ([`construir_grafo`](spmc_v1.py#L1268)) — linear, sem laço:

```
START ─► gerar_plano ─► parsear_plano ─► executar_cadeias ─► montar_contexto ─► gerar_resposta_final ─► END
          (Gemini)       (Pydantic)            │                (Python)            (CoRAG)
                                                │  ThreadPoolExecutor
                             ┌──────────────────┼──────────────────┐
                             ▼                  ▼                  ▼
                        subgrafo(cadeia 1)  subgrafo(cadeia 2) ... subgrafo(cadeia N)
```

**Subgrafo de uma cadeia** ([`construir_grafo_cadeia`](spmc_v1.py#L1088)) — um por
thread, com laço:

```
START ─► resolver_placeholder ─► gerar_subpergunta ─► retrieve ─► evidence_extractor ─► gerar_subresposta
                ▲                        ▲                                                     │
                │                        └──────────── "retry_subpergunta" ───────────────────┤
                └───────────────────────────────────── "proximo_passo" ───────────────────────┤
                                                                                              │
                                                                   "agregar" ─► END ◄─────────┘
```

### Os nós

| Grafo | Nó | Modelo | Teto de tokens | Pode pular a chamada? |
|---|---|---|---|---|
| topo | [`gerar_plano`](spmc_v1.py#L724) | **Gemini** (`GEMINI_MODEL`) | `PLAN_MAX_TOKENS` 8192 | não |
| topo | [`parsear_plano`](spmc_v1.py#L752) | nenhum (Pydantic) | — | — |
| topo | [`executar_cadeias`](spmc_v1.py#L1162) | nenhum (dispara as threads) | — | — |
| cadeia | [`resolver_placeholder`](spmc_v1.py#L798) | nenhum (regex) | — | — |
| cadeia | [`gerar_subpergunta`](spmc_v1.py#L840) | **CoRAG** | `SUBQ_MAX_TOKENS` 256 | sim, se há dependência pendente |
| cadeia | [`retrieve`](spmc_v1.py#L908) | nenhum (E5) | — | sim, se há dependência pendente |
| cadeia | [`evidence_extractor`](spmc_v1.py#L927) | **CoRAG** (`json_mode`) | `EVIDENCE_MAX_TOKENS` 4000 | sim, se não vieram chunks |
| cadeia | [`gerar_subresposta`](spmc_v1.py#L976) | **CoRAG** | `SUBA_MAX_TOKENS` 128 | sim, se há pendência ou não há evidência |
| topo | [`montar_contexto`](spmc_v1.py#L1194) | nenhum (Python) | — | — |
| topo | [`gerar_resposta_final`](spmc_v1.py#L1226) | **CoRAG** | `FINAL_MAX_TOKENS` 128 | não |

A Gemini só gera o plano. Todo o resto usa o CoRAG (`corag/CoRAG-Llama3.1-8B-MultihopQA`,
servido remotamente e configurado por `BASE_URL`, `API_KEY` e `MODEL_NAME` em
`src/reproducao/.env`). Todas as chamadas passam por [`_chamar_llm`](spmc_v1.py#L452),
com `temperature=0`.

---

## 1. Fase 1 — Plano

### 1.1 `gerar_plano` ([l.724](spmc_v1.py#L724))

**Lê:** `questao`, `N`, `L`. **Escreve:** `plano` (o JSON cru, como texto).

1. Formata o [`INSTRUCOES_PLANO`](spmc_v1.py#L74) com N e L e chama a Gemini com
   `json_mode=True`. A mensagem `user` é só a pergunta.
2. Tenta validar o texto contra o schema `Plano` (`_extrair_json` + `model_validate_json`).
3. Se não validar, **regenera**, até `PLAN_JSON_RETRIES` (2) vezes a mais: 3 tentativas
   no total.
4. Esgotadas as tentativas, **devolve o último texto mesmo assim**. Quem decide se o
   plano é aceitável é o nó seguinte.

O JSON pedido tem este formato por passo:

```json
{
  "k": 2,
  "guide_text": "Com base no Passo 1, descobrir o nome do diretor do filme.",
  "triple": {"subject": "Romeo and Juliet (1968)", "predicate": "hasDirector",
             "object": "[S_2: person: ??]"},
  "define": "S_2",
  "depends_on": []
}
```

- **`define`**: o placeholder que este passo descobre.
- **`depends_on`**: os placeholders de passos anteriores dos quais este passo precisa.
- A tripla do passo **tem que conter** o próprio placeholder (`[S_2: ...]`), mesmo que o
  modelo "saiba" a resposta. É assim que a execução sabe o que perguntar e onde
  guardar o valor achado.

### 1.2 `parsear_plano` ([l.752](spmc_v1.py#L752))

**Lê:** `plano`. **Escreve:** `plano_estruturado` (objeto `Plano`).

1. Valida de novo o JSON contra `Plano`. Aqui a falha é **fatal**: imprime o JSON cru no
   stderr e relança o `ValidationError`, o que derruba a pergunta.
2. Durante a validação, os validadores Pydantic **consertam variações** comuns do LLM
   (ver [`Passo._normalizar`](spmc_v1.py#L289) e [`Tripla._apelidos`](spmc_v1.py#L260)):
   - tripla com outro nome de chave (`tripla`, `triple_template`...);
   - `subject/predicate/object` soltos no passo, sem a chave `triple`;
   - nomes em português (`sujeito`, `depende_de`) e diferenças de maiúsculas;
   - `define: ""` vira `None`, e `depends_on` vira sempre uma lista.
3. Faz **checagens suaves**, que só avisam no stderr e não interrompem nada:
   - número de cadeias diferente de N;
   - número de passos de uma cadeia diferente de L;
   - tripla com algum campo vazio.

Consequência das checagens suaves: uma cadeia com 5 passos em vez de 6 **roda com 5**.
O laço do subgrafo percorre `len(cadeia.steps)`, não L.

---

## 2. Fase 2 — Distribuição das cadeias em threads

### 2.1 `executar_cadeias` ([l.1162](spmc_v1.py#L1162))

**Lê:** `plano_estruturado`, `questao`, `L`, `verbose`. **Escreve:** `cadeias_exec`.

1. Abre um `ThreadPoolExecutor` com `min(SPMC_CHAIN_THREADS, N)` workers (padrão 4).
2. Submete um `_executar_cadeia` por cadeia.
3. Coleta os resultados com `as_completed`, mas os grava **na posição original** da
   cadeia (pelo índice do `enumerate`). Assim `cadeias_exec[i]` é sempre a cadeia i,
   seja qual for a ordem de término.

Com `SPMC_CHAIN_THREADS=1`, as cadeias rodam uma depois da outra; o resultado é o mesmo,
só o tempo muda.

### 2.2 `_executar_cadeia` ([l.1132](spmc_v1.py#L1132)) — dentro de cada thread

1. Empacota a cadeia num **mini-plano de um elemento só**:
   `Plano(overview="", entities=[], relations=[], chains=[cadeia])`. Os nós do subgrafo
   acessam `plano_estruturado.chains[cadeia_atual]`, então com um só elemento e
   `cadeia_atual = 0` eles trabalham sobre a cadeia desta thread.
2. Invoca o subgrafo (compilado uma única vez e reusado, ver
   [`_grafo_cadeia`](spmc_v1.py#L1120)) com o estado inicial:
   ```python
   {"questao", "plano_estruturado": mini_plano, "cadeia_atual": 0, "passo_atual": 0,
    "resolvidos": {}, "historico": [], "cadeias_exec": [], "verbose"}
   ```
3. Passa `recursion_limit = L · (5 + 4 · SUBQ_RETRY_MAX) + 10`. Cada passo visita 5 nós
   e, a cada retry de subpergunta, mais 4. Com L = 6 e 1 retry, o limite é 64.
4. Devolve `estado["cadeias_exec"][0]`, no formato
   `{"cadeia": id, "descricao": ..., "hops": [...]}`.

O overview, as entidades e as relações do plano **não chegam** às cadeias: ficam vazios
no mini-plano, e nenhum prompt de execução os usa.

---

## 3. Fase 3 — Execução de um passo (hop)

Cada volta do subgrafo executa um passo. Os 5 nós, em ordem:

### 3.1 `resolver_placeholder` ([l.798](spmc_v1.py#L798))

**Lê:** o passo atual, `resolvidos`. **Escreve:** `tripla_resolvida`,
`pendentes_atual`, `resolvidos`, zera `tentativas_subpergunta` e `retry_subpergunta`;
no passo 0, também zera `historico`.

1. Se é o primeiro passo da cadeia (`passo_atual == 0`), começa com `resolvidos = {}` e
   `historico = []`. Cada cadeia tem seus próprios `S_1..S_L`.
2. [`_resolver_tripla`](spmc_v1.py#L371) troca cada `[S_k: tipo: ??]` pelo valor em
   `resolvidos["S_k"]`, quando existe. Os que não têm valor ficam como estão.
3. Calcula **`pendentes_atual`** = os itens de `passo.depends_on` que ainda não estão em
   `resolvidos`. É isso que decide se o passo tem como ser executado.

Atenção à distinção: o placeholder que o **próprio** passo define (ex. `S_2` no passo 2)
sempre aparece sem valor na tripla, e isso é normal. Ele **não** conta como pendência;
só as dependências declaradas em `depends_on` contam.

Exemplo, com `resolvidos = {"S_2": "Franco Zeffirelli"}` (valor ilustrativo):

```
template:   <[S_2: person: ??]; birthPlace; [S_4: location: ??]>   depends_on = ["S_2"]
resolvida:  <Franco Zeffirelli; birthPlace; [S_4: location: ??]>   pendentes_atual = []
```

### 3.2 `gerar_subpergunta` ([l.840](spmc_v1.py#L840))

**Lê:** `pendentes_atual`, `historico`, o passo, `tripla_resolvida`,
`tentativas_subpergunta`, `subpergunta_anterior`. **Escreve:** `subpergunta_atual`.

- **Se há pendência**, devolve `subpergunta_atual = ""` **sem chamar o LLM**: sem o valor
  da dependência, não há entidade concreta para perguntar.
- **Senão**, monta a mensagem `user` para o [`INSTRUCOES_SUBPERGUNTA`](spmc_v1.py#L141):
  ```
  Original question: ...
  CONTEXT (previous steps in this chain):
  Step 1: sub-question / sub-answer / triple       ← _formatar_contexto(historico)
  Current step (k=4):
    guide text: ...
    target of THIS step (ask for exactly this, nothing more): a value of type "location" (placeholder S_4)
    triple template (partially resolved): <Franco Zeffirelli; birthPlace; [S_4: location: ??]>
  [PREVIOUS ATTEMPT found no answer: "..."  ← só em retry
   Generate a DIFFERENT sub-question ...]
  Generate the sub-question.
  ```
  - A linha **"target of THIS step"** vem de [`_tipo_placeholder`](spmc_v1.py#L402), que lê
    o tipo do placeholder do passo na tripla original (`location` em
    `[S_4: location: ??]`). Ela impede a subpergunta de "vazar" para o que um passo
    futuro vai buscar. Se o plano não pôs o próprio placeholder na tripla, a linha não
    aparece.
  - O bloco **PREVIOUS ATTEMPT** só aparece em retry (`tentativas_subpergunta > 0`) e
    pede uma formulação diferente da anterior.

### 3.3 `retrieve` ([l.908](spmc_v1.py#L908))

**Lê:** `pendentes_atual`, `subpergunta_atual`. **Escreve:** `chunks_atual`.

- **Se há pendência**, devolve `chunks_atual = []` sem buscar.
- **Senão**, busca `SPMC_TOP_K` (5) chunks com a subpergunta:
  - `SPMC_RETRIEVER=mini` (padrão): [`_retrieve_mini_corpus`](spmc_v1.py#L631), busca
    densa E5 in-process sobre `data/mini/`. Na primeira chamada do processo, carrega o
    modelo E5 e os embeddings sob lock, porque várias threads podem chegar juntas;
  - `SPMC_RETRIEVER=e5_server`: [`_retrieve_e5_server`](spmc_v1.py#L657), por HTTP.
    Está incompleto: o servidor devolve só `doc_id` e `score`, então título e texto vêm
    vazios.

### 3.4 `evidence_extractor` ([l.927](spmc_v1.py#L927))

**Lê:** `chunks_atual`, `subpergunta_atual`. **Escreve:** `evidencias_atual`.

- **Sem chunks**, devolve `[]` sem chamar o LLM.
- **Senão**, manda ao CoRAG ([`INSTRUCOES_EVIDENCE`](spmc_v1.py#L162)) a subpergunta e os
  chunks, cada um como `[i] título` + os primeiros `EVIDENCE_CHUNK_CHARS` (600)
  caracteres. O modelo devolve
  `{"evidence": [{"subject", "predicate", "object"}, ...]}`, só com as triplas que
  sustentam a resposta.
- Se o JSON não validar (tipicamente porque a resposta foi **truncada** no teto de
  tokens), registra o erro no stderr e usa **lista vazia**. Toda a evidência do passo se
  perde, mesmo que parte das triplas estivesse boa.

### 3.5 `gerar_subresposta` ([l.976](spmc_v1.py#L976)) — a decisão do passo

**Lê:** o passo, `subpergunta_atual`, `evidencias_atual`, `pendentes_atual`,
`tentativas_subpergunta`, `resolvidos`, `historico`, `chunks_atual`. **Escreve:** ver
abaixo.

**Passo A — decidir a subresposta e o motivo:**

```
há pendentes_atual?  ── sim ─► subresposta = SEM_RESPOSTA, motivo = "dependencia_ausente"   (0 chamadas)
       │ não
evidências vazias?   ── sim ─► subresposta = SEM_RESPOSTA, motivo = "sem_evidencia"         (0 chamadas)
       │ não
chama o CoRAG com INSTRUCOES_SUBRESPOSTA + subpergunta + triplas de evidência
       │
resposta == "no relevant information found"?
       ├─ sim ─► motivo = "sem_resposta_llm"
       └─ não ─► motivo = None   (sucesso)
```

A subresposta é gerada **só a partir das triplas de evidência**, nunca dos chunks
brutos.

**Passo B — decidir se tenta de novo:**

```python
pode_repetir = motivo in ("sem_evidencia", "sem_resposta_llm") and tentativas < SUBQ_RETRY_MAX
```

- `dependencia_ausente` **nunca** é retentado: outra subpergunta não criaria o valor que
  falta de um passo anterior.
- Com `SUBQ_RETRY_MAX = 1` (padrão), cada passo tem no máximo 2 tentativas.
- Se `pode_repetir`, o nó devolve **só**
  `{tentativas_subpergunta + 1, subpergunta_anterior, retry_subpergunta: True}`. Ele
  **não** grava nada no histórico nem avança o passo.

**Passo C — concluir o passo** (sucesso, ou falha sem mais tentativas):

1. Se `motivo is None` e o passo tem `define`, grava `resolvidos[define] = subresposta`.
   Uma falha **nunca** preenche um placeholder com "no relevant information found".
2. **Recalcula a tripla a partir do template original**, já com o `resolvidos`
   atualizado. Assim o registro do passo mostra também o valor que ele próprio acabou
   de descobrir.
3. Anexa o registro do passo ao `historico`:
   ```python
   {"k", "subpergunta", "subresposta", "motivo", "tentativas",
    "tripla", "chunks": [títulos], "evidencias": [triplas em texto]}
   ```
4. Avança o ponteiro:
   - se não é o último passo, `passo_atual += 1`;
   - se é o último, arquiva `{cadeia, descricao, hops: historico}` em `cadeias_exec` e
     faz `cadeia_atual += 1` (vira 1), o que sinaliza ao roteador que a cadeia acabou.

### 3.6 Roteador `_proximo_apos_subresposta` ([l.1245](spmc_v1.py#L1245))

| Condição | Rota | Para onde vai |
|---|---|---|
| `retry_subpergunta == True` | `"retry_subpergunta"` | `gerar_subpergunta`, mesmo passo (sem passar por `resolver_placeholder`, então o contador de tentativas não é zerado) |
| `cadeia_atual < len(chains)` (ainda 0) | `"proximo_passo"` | `resolver_placeholder` do próximo passo |
| senão (`cadeia_atual` virou 1) | `"agregar"` | `END` do subgrafo |

Um passo sem resposta **não interrompe** a cadeia: ela sempre percorre todos os passos.
Cada passo decide sozinho, por `pendentes_atual`, se tem como ser executado.

---

## 4. Propagação de placeholders ao longo de uma cadeia

O que liga os passos é o dicionário `resolvidos`. Usando a cadeia 1 do
[`plano_exemplo.md`](plano_exemplo.md) (pergunta sobre a cidade natal do diretor de
*Romeo and Juliet* (1968)), com valores e resultados **ilustrativos**. O
`plano_exemplo.md` mostra só as triplas, então os `depends_on` abaixo foram inferidos
delas, e `[S_2]` abrevia `[S_2: person: ??]`:

| Passo | Tripla-template | `depends_on` | Resultado | `resolvidos` depois |
|---|---|---|---|---|
| 1 | `<Romeo and Juliet (1968); releaseYear; 1968>` | — | sucesso | `{}` (passo sem `define`) |
| 2 | `<Romeo and Juliet (1968); hasDirector; [S_2: person: ??]>` | — | sucesso: "Franco Zeffirelli" | `{S_2}` |
| 3 | `<[S_2]; birthDate; [S_3: date: ??]>` | `S_2` | sucesso | `{S_2, S_3}` |
| 4 | `<[S_2]; birthPlace; [S_4: location: ??]>` | `S_2` | **falha** (`sem_evidencia`, após retry) | `{S_2, S_3}`: S_4 fica sem valor |
| 5 | `<[S_4: location: ??]; isA; city>` | `S_4` | **`dependencia_ausente`**: 0 chamadas | inalterado |
| 6 | `<[S_2]; bornInCity; [S_6: city: ??]>` | `S_2` | executa normalmente | ... |

Duas consequências do mecanismo:

- **Falha em cascata.** Quando um passo falha, todos os que dependem do placeholder dele
  viram `dependencia_ausente` sem nem tentar, como o passo 5. Nesse caso é economia de
  chamadas; em outros, a cadeia perde a chance de achar o valor por outro caminho.
- **Dependência que não passa pelo placeholder.** O passo 6 depende só de `S_2` e roda
  mesmo com o passo 5 falho. O quanto uma cadeia resiste a falhas depende de como o
  plano distribuiu os `depends_on`.

---

## 5. Fase 4 — Agregação e resposta final

### 5.1 `montar_contexto` ([l.1194](spmc_v1.py#L1194))

**Lê:** `cadeias_exec`. **Escreve:** `contexto`.

Para cada cadeia, mantém **só os passos com `motivo is None`**. Passos sem resposta
seriam só ruído. Cadeias sem nenhum passo respondido são omitidas. O formato é:

```
Chain 1 (<descrição da cadeia>):
  Step 2: <subpergunta> -> <subresposta>
    triple: <Romeo and Juliet (1968); hasDirector; Franco Zeffirelli>
  Step 3: ...

Chain 3 (...):
  ...
```

Se nenhuma cadeia respondeu nada, o contexto é
`(no sub-questions were successfully answered)`. Diferente do `spmc_no_placeholder`, aqui
a **descrição** da cadeia e a **tripla resolvida** entram no contexto.

### 5.2 `gerar_resposta_final` ([l.1226](spmc_v1.py#L1226))

**Lê:** `questao`, `contexto`. **Escreve:** `resposta_final`.

Chama o CoRAG com [`INSTRUCOES_RESPOSTA_FINAL`](spmc_v1.py#L207) e a mensagem:

```
Original question: ...
Reasoning chains (answered steps only):
<contexto>
Answer the original question.
```

O prompt pede resposta curta, sem conhecimento externo, cruzando as cadeias e confiando
nas que chegaram a uma conclusão, sem exigir que todas concordem; ou
`no relevant information found`. A resposta final **não recebe nenhum chunk nem
evidência**, só o que passou pelo `montar_contexto`.

---

## 6. Custo por pergunta

Chamadas ao CoRAG em **um passo**, conforme o caminho:

| Caminho do passo | Chamadas |
|---|---|
| `dependencia_ausente` | 0 |
| `sem_evidencia` (extrator não achou nada) | 2 (subpergunta + evidência) |
| sucesso ou `sem_resposta_llm` | 3 (subpergunta + evidência + subresposta) |
| com 1 retry | soma as duas tentativas: até 6 |

Por pergunta:

```
Gemini:  1 plano (até 3 se o JSON vier inválido)
CoRAG:   N · L · (0 a 6) nos passos  +  1 resposta final
```

Com N = 4 e L = 6, sem retries e com todos os passos executando, são 72 + 1 chamadas ao
CoRAG; no pior caso, 144 + 1.

Como as cadeias rodam em paralelo, o **tempo** é dado pela cadeia mais lenta: até
`L × 6` chamadas em sequência dentro dela, mais o plano e a resposta final. As
execuções de 30 perguntas registradas no `CODIGO.md` (N = 4, L = 6, 2 threads) levaram
de 54 min a 1 h. Pesam a chamada extra do extrator de evidência por passo e os retries,
sobretudo quando o extrator é truncado no teto de tokens e gera um `sem_evidencia`.

---

## 7. Erros ao longo do fluxo

**Erros transitórios** — tratados dentro de [`_chamar_llm`](spmc_v1.py#L452):

- falha de conexão, 429 e os status 408, 409, 500, 502, 503, 504 e 529 são retentados
  até `PLAN_MAX_RETRIES` (5) vezes, com esperas de 2, 4, 8 e 16 s;
- outros status (400, 401, 403...) sobem na hora.

O retry fica dentro da função, não no LangGraph; por isso nenhum nó tem `retry_policy`.

**O que derruba a pergunta inteira:**

- plano que não valida em `parsear_plano`;
- `BASE_URL`/`API_KEY` ou `GEMINI_API_KEY` ausentes (`RuntimeError`);
- qualquer exceção dentro de uma thread. O `futuro.result()` relança o erro em
  `executar_cadeias`, mas só depois que o `with ThreadPoolExecutor` **espera as outras
  cadeias terminarem**. A pergunta falha, e o tempo delas é gasto à toa;
- resposta do LLM com `content = None`. O código faz `.strip()` e `.find()` direto no
  retorno, sem proteção.

**O que é absorvido em silêncio:**

- evidência com JSON inválido ou truncado: vira lista vazia, depois `sem_evidencia`, e
  dispara um retry;
- contagem N/L errada ou tripla incompleta no plano: só um aviso no stderr.

Os scripts de execução (seção 9) capturam a exceção por pergunta, gravam
`SEM_RESPOSTA` como predição e seguem para a próxima.

---

## 8. O estado (`SPMCState`) ao longo do fluxo

[l.691](spmc_v1.py#L691). O mesmo `TypedDict` serve aos dois grafos; cada um usa uma
parte dos campos.

| Campo | Grafo | Quem escreve | Observação |
|---|---|---|---|
| `questao`, `N`, `L`, `verbose` | ambos | entrada do `invoke` | `verbose` liga os prints de diagnóstico no stderr |
| `plano` | topo | `gerar_plano` | JSON cru |
| `plano_estruturado` | ambos | `parsear_plano` / `_executar_cadeia` | no subgrafo, é o mini-plano de uma cadeia |
| `cadeia_atual` | cadeia | `gerar_subresposta` | 0 durante a cadeia; vira 1 no fim |
| `passo_atual` | cadeia | `gerar_subresposta` | 0..L-1 |
| `resolvidos` | cadeia | `resolver_placeholder`, `gerar_subresposta` | `{"S_k": valor}`, zerado no passo 0 |
| `tripla_resolvida`, `pendentes_atual` | cadeia | `resolver_placeholder` | por passo |
| `subpergunta_atual` | cadeia | `gerar_subpergunta` | `""` quando há pendência |
| `subpergunta_anterior`, `tentativas_subpergunta`, `retry_subpergunta` | cadeia | `gerar_subresposta` (zerados em `resolver_placeholder`) | controle do retry |
| `chunks_atual` | cadeia | `retrieve` | top-k do passo |
| `evidencias_atual` | cadeia | `evidence_extractor` | triplas de evidência |
| `subresposta_atual` | cadeia | `gerar_subresposta` | última subresposta |
| `historico` | cadeia | `gerar_subresposta` | registros dos passos concluídos da cadeia |
| `cadeias_exec` | ambos | `gerar_subresposta` (1 item) / `executar_cadeias` (N itens) | `{cadeia, descricao, hops}` |
| `contexto` | topo | `montar_contexto` | texto das cadeias para o prompt final |
| `resposta_final` | topo | `gerar_resposta_final` | |
| `log` | ambos | todos os nós | `Annotated[list, operator.add]`: trilha dos nós visitados |

Só o `log` tem reducer; os outros campos são sobrescritos pelo que cada nó devolve. Como
cada thread invoca o subgrafo com um estado próprio, não há campo compartilhado entre
as cadeias.

---

## 9. Como executar

> **Atenção:** desde a reorganização em subpastas, o `REPO_ROOT = parents[2]` desses
> scripts aponta para `src/` em vez da raiz do repositório. O `spmc_v1.py` passa a
> procurar `data/mini` e o `.env` no lugar errado. Além disso, os imports
> `from SPMC import spmc` e `from SPMC.gerar_planos import ...` dos scripts de execução
> apontam para caminhos que não existem mais (o módulo agora é
> `SPMC.SPMC_v1.spmc_v1`). Os comandos abaixo só funcionam depois dessas correções.

**Uma pergunta, com diagnóstico completo** (`__main__`, [l.1303](spmc_v1.py#L1303)):

```bash
GEMINI_MODEL=gemini-flash-lite-latest SPMC_CHAIN_THREADS=2 \
  .venv/bin/python src/SPMC/SPMC_v1/spmc_v1.py
```

Roda a pergunta fixa *"What nationality was Oliver Reed's character in the film Royal
Flash?"* com `verbose=True` e imprime:

- os diagramas mermaid dos dois grafos;
- o JSON do plano;
- o fluxo passo a passo de cada cadeia: subpergunta, chunks, evidências, subresposta,
  motivo, tentativas e tripla;
- o contexto agregado e a resposta final.

**As 30 perguntas do hotpotqa** ([`rodar_hotpotqa30.py`](rodar_hotpotqa30.py)): invoca o
grafo completo pergunta a pergunta. Grava em `data/spmc_runs/` um `.jsonl` por execução,
escrito a cada pergunta, e uma linha de métricas em `index.jsonl`.

```bash
PYTHONPATH=src SPMC_N=4 SPMC_L=6 SPMC_CHAIN_THREADS=2 \
  .venv/bin/python src/SPMC/SPMC_v1/rodar_hotpotqa30.py
```

**Sobre planos já gerados** ([`rodar_spmc.py`](rodar_spmc.py)): pula a Fase 1. Lê os
planos de um arquivo de `../Gerar_plano/gerar_planos.py` e chama em sequência
`no_executar_cadeias`, `no_montar_contexto` e `no_gerar_resposta_final`, sem montar o
grafo de topo. Com `--provider ollama`, troca `_chamar_corag` por uma chamada ao Ollama
(monkeypatch), para rodar os passos com outro modelo sem alterar o `spmc_v1.py`.

**Variáveis de ambiente mais usadas:**

| Variável | Controla |
|---|---|
| `SPMC_N`, `SPMC_L` | tamanho do plano (padrão 4 e 6) |
| `SPMC_CHAIN_THREADS` | cadeias em paralelo |
| `SUBQ_RETRY_MAX` | retries de subpergunta por passo |
| `SPMC_TOP_K`, `SPMC_RETRIEVER`, `MINI_BASE_DIR` | retrieval |
| `GEMINI_MODEL`, `MODEL_NAME` | modelos do plano e dos passos |
| `*_MAX_TOKENS`, `EVIDENCE_CHUNK_CHARS` | tetos de tokens e tamanho dos chunks |
