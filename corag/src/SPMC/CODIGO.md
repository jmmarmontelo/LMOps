# `spmc.py` explicado

Este documento descreve, parte por parte, como `src/SPMC/spmc.py` implementa o fluxo
SPMC-RAG (ver `SPMCv1.md` para o algoritmo em si). O objetivo aqui é explicar **o
código**: por que cada peça existe, como elas se encaixam, e as decisões de design por
trás de cada uma.

## Visão geral do pipeline

```
gerar_plano → parsear_plano → executar_cadeias → montar_contexto → gerar_resposta_final
                                     │
                                     │  (N threads, uma por cadeia)
                                     ▼
                     resolver_placeholder → gerar_subpergunta → retrieve
                            ▲                                      │
                            │                                      ▼
                     (próximo passo)                    evidence_extractor
                            │                                      │
                            └──────────── gerar_subresposta ◄──────┘
                                       │        ▲
                                       │        │ (retry de subpergunta)
                                       └────────┘
```

O grafo de nível superior (`construir_grafo`) é **linear**: gera o plano global,
valida, despacha as N cadeias em paralelo (cada uma correndo seu próprio subgrafo,
`construir_grafo_cadeia`), agrega os resultados e produz a resposta final. O loop de
passos (e o retry de subpergunta) vive **dentro** do subgrafo por cadeia, não no grafo
de topo.

### Qual modelo cada nó usa

Dois modelos diferentes, ambos via endpoint OpenAI-compatível (mesma lib `openai`):

| Nó | Modelo | Função de chamada |
|---|---|---|
| `gerar_plano` | **Gemini** (`GEMINI_MODEL`) | `_chamar_gemini` |
| `gerar_subpergunta` | **CoRAG** (`MODEL_NAME`) | `_chamar_corag` |
| `evidence_extractor` | **CoRAG** | `_chamar_corag` (`json_mode=True`) |
| `gerar_subresposta` | **CoRAG** | `_chamar_corag` |
| `gerar_resposta_final` | **CoRAG** | `_chamar_corag` |
| `retrieve` | nenhum (E5 denso in-process) | — |
| `parsear_plano`, `resolver_placeholder`, `montar_contexto`, `executar_cadeias` | nenhum (lógica pura) | — |

A Gemini só gera o plano global; todo o trabalho por hop e a resposta final são feitos
pelo CoRAG (`corag/CoRAG-Llama3.1-8B-MultihopQA`, servido remotamente).

---

## 1. Configuração e constantes

Tudo controlável por variável de ambiente, sem precisar editar código (o `.env` lido é
`src/reproducao/.env`):

- **Cliente Gemini** (só `gerar_plano`): `GEMINI_BASE_URL`/`GEMINI_MODEL`/
  `GEMINI_API_KEY` — a Gemini é chamada através do endpoint OpenAI-compatível dela,
  reusando a lib `openai` já presente no repo (sem dependência nova). `GEMINI_MODEL`
  default é `gemini-flash-latest` (o alias `gemini-2.5-flash` já não funciona para
  chaves novas).
- **Cliente CoRAG** (todos os outros nós): `CORAG_BASE_URL`, `CORAG_API_KEY`,
  `CORAG_MODEL`, lidos das variáveis `BASE_URL`, `API_KEY`, `MODEL_NAME` do mesmo
  `.env` — as mesmas que `src/reproducao/dynamic_chain.py` e `best_of_n.py` já usam.
  `MODEL_NAME` tem default `corag-8b` (o *alias* registrado no servidor vLLM remoto, não
  o repo-id completo do HuggingFace). Sem `BASE_URL`/`API_KEY`, `_chamar_corag` levanta
  um `RuntimeError` claro na primeira chamada.
- **Tamanho do problema**: `L_DEFAULT=6` (passos por cadeia, mesmo valor de
  `max_path_length` do CoRAG original) e `N_DEFAULT=4` (número de cadeias).
- **Tetos de tokens por tipo de chamada** (`PLAN_MAX_TOKENS`, `SUBQ_MAX_TOKENS`,
  `EVIDENCE_MAX_TOKENS`, `SUBA_MAX_TOKENS`, `FINAL_MAX_TOKENS`): cada etapa gera uma
  saída de tamanho bem diferente (o plano é grande, uma subresposta é uma frase curta),
  então cada uma tem seu próprio teto — importante porque a API assume um valor baixo
  (16) se você omitir, o que truncaria respostas maiores. `EVIDENCE_MAX_TOKENS` é o
  mais sensível: o default hoje é **4000** (passou por 1024 → 2048 → 4000) porque o
  CoRAG gera o JSON de evidência bem mais verboso que a Gemini gerava — ver
  "Limitações conhecidas" no fim do documento.
- **`SUBQ_RETRY_MAX`**: quantas vezes tentar uma subpergunta *diferente* no mesmo passo
  antes de desistir, quando a primeira tentativa não encontrou resposta.
- **`PLAN_MAX_RETRIES`**: retries para erros transitórios da API (503/429/5xx/conexão).
- **`PLAN_JSON_RETRIES`**: se o JSON do plano vier malformado ou não bater com o schema
  Pydantic, gera de novo (até N vezes) em vez de falhar direto.
- **`SPMC_CHAIN_THREADS`**: quantas cadeias rodam ao mesmo tempo (ver seção 8). As N
  cadeias são independentes até a agregação final, então isso é pura paralelização de
  chamadas de rede — não muda o resultado, só o tempo total.
- **`SEM_RESPOSTA`**: a string exata (`"no relevant information found"`) que sinaliza
  "não consegui responder este hop" — usada tanto nos prompts (pedindo pro LLM
  respondê-la literalmente) quanto no código (comparando a saída do LLM com ela).
- **Bloco de retrieval** (`SPMC_RETRIEVER`, `SPMC_TOP_K`, `MINI_BASE_DIR`,
  `E5_MODEL_NAME_OR_PATH`, `E5_SERVER_HOST/PORT`): escolhe entre buscar no mini-corpus
  local (denso, in-process) ou num servidor E5 remoto — ver seção 6.

## 2. Os prompts (`INSTRUCOES_*`)

Todos em inglês, pois tudo que é enviado ao LLM (prompts, nomes de campo do JSON, blocos
montados por `_formatar_*`) fica em inglês. As mensagens de `print`/log também são em
inglês; comentários, docstrings e identificadores Python (nomes de função, chaves do
`SPMCState`) continuam em português.

- **`INSTRUCOES_PLANO`**: gera o plano global — N cadeias alternativas, cada uma com L
  passos, cada passo com um "triple template" (`<subject; predicate; object>`) contendo
  placeholders `[S_k: type: ??]` nas posições cujo valor só será conhecido depois de
  executar o passo k. As regras mais importantes:
  - o plano é só o **esqueleto** do raciocínio — não contém sub-perguntas nem respostas,
    essas são geradas depois, durante a execução;
  - um passo que `define` um placeholder (ex. `"S_2"`) **precisa** ter esse token
    literalmente na própria tripla, mesmo que o LLM já "saiba" a resposta de verdade —
    regra adicionada depois que percebemos o modelo às vezes "trapaceando" (escrevendo
    o valor real em vez do placeholder, o que quebra a cadeia porque um passo posterior
    com `depends_on: ["S_2"]` nunca recebe o valor).
- **`INSTRUCOES_SUBPERGUNTA`**: gera UMA sub-pergunta em linguagem natural para o passo
  atual, com base no texto-guia e na tripla (parcialmente resolvida). Regra chave: se um
  "target of THIS step" for informado no prompt (ver `_tipo_placeholder`), a pergunta
  tem que pedir EXATAMENTE aquele valor — nada de misturar com informação de um passo
  futuro (bug que já apareceu e foi corrigido travando o alvo explicitamente).
- **`INSTRUCOES_EVIDENCE`**: o Evidence Extractor — recebe os chunks recuperados e a
  sub-pergunta, devolve só as triplas `<s; p; o>` que sustentam a resposta (CoT em 5
  passos: identificar entidades/relações, localizar sentenças relevantes, extrair
  triplas, refinar, e produzir a saída final). Existe pra trocar K chunks ruidosos por
  poucas triplas de alta precisão antes de gerar a subresposta.
- **`INSTRUCOES_SUBRESPOSTA`**: responde a sub-pergunta usando **só** as triplas de
  evidência (nunca os chunks brutos) — se não der pra responder, devolve
  `SEM_RESPOSTA` literalmente.
- **`INSTRUCOES_RESPOSTA_FINAL`**: o último passo — recebe as N cadeias já agregadas
  (só os hops que foram respondidos de verdade) e produz a resposta final à pergunta
  original, sem precisar que todas as cadeias concordem entre si.

## 3. Schema Pydantic do plano

```python
Tripla(subject, predicate, object)          # <s; p; o>, pode conter [S_k: tipo: ??]
Passo(k, guide_text, triple, define, depends_on)   # um hop de uma cadeia
Cadeia(id, description, steps: list[Passo])        # uma rota de raciocínio
Plano(overview, entities, relations, chains: list[Cadeia])  # o plano global
Evidencias(evidence: list[Tripla])          # saida do evidence extractor
```

Todos os campos de `Tripla` e `Passo` têm `model_validator(mode="before")` que
normalizam variações que o LLM comete no JSON de saída, mesmo com
`response_format={"type": "json_object"}` (que garante JSON válido, mas não garante o
shape certo):

- `Tripla._apelidos`: aceita `sujeito/predicado/objeto` (nomes antigos em português) ou
  `s/p/o` como apelidos de `subject/predicate/object`.
- `Passo._normalizar`: conserta a tripla vindo sob outro nome de chave (`tripla`,
  `triple_template`, etc.), ou solta como campos soltos no passo; normaliza
  `define: ""` para `None`; garante que `depends_on` seja sempre uma lista.

Essa tolerância existe porque, na prática, o LLM não segue o schema 100% das vezes —
em vez de falhar toda a geração do plano por um detalhe de nomenclatura, o código
absorve as variações conhecidas.

## 4. Utilitários de placeholder e formatação

- **`_extrair_json`**: corta o objeto JSON da resposta bruta do LLM (do primeiro `{` ao
  último `}`, tolerando texto/cercas ```` ```json ```` em volta) e remove vírgulas
  sobrando antes de `}`/`]` — o LLM às vezes deixa uma vírgula a mais, o que quebra o
  parser padrão de JSON.
- **`_RE_PLACEHOLDER`** / **`_resolver_tripla`**: o regex casa `[S_k: tipo: ??]` (com ou
  sem underscore, com qualquer rótulo de tipo). `_resolver_tripla` substitui cada
  placeholder pelo valor já conhecido em `resolvidos` (dict `{"S_k": valor}`); os que
  ainda não têm valor ficam intactos, e a função devolve também a lista ordenada dos
  que continuam pendentes.
- **`_tripla_str`**: formata uma `Tripla` como `<subject; predicate; object>` — usado em
  todo log e prompt.
- **`_RE_PLACEHOLDER_TIPO`** / **`_tipo_placeholder`**: variante do regex que também
  captura o rótulo de tipo (`"character"` em `[S_2: character: ??]`). Usado para travar
  a sub-pergunta gerada no alvo certo do passo atual — sem isso, o LLM às vezes gera uma
  pergunta que "vaza" para o assunto de um passo futuro.
- **`_formatar_contexto`/`_formatar_chunks`/`_formatar_evidencias`**: montam blocos de
  texto (em inglês, pois vão para dentro dos prompts) a partir do histórico da cadeia,
  dos chunks recuperados, e das triplas de evidência, respectivamente.

## 5. `_chamar_llm`, `_chamar_gemini` e `_chamar_corag` — as portas de saída para o LLM

Todo nó que precisa de um LLM passa por uma destas três funções, todas com a mesma
assinatura `(system, user, *, max_tokens, verbose=False, json_mode=False) -> str`:

- **`_chamar_llm(base_url, api_key, model, system, user, ...)`** — o helper genérico
  onde mora toda a lógica; recebe o destino (`base_url`/`api_key`/`model`) como
  parâmetro. Faz uma chamada `chat.completions.create` via `openai.OpenAI`, com:
  - `temperature=0` (determinístico) e `max_tokens` por chamada (recebido como
    parâmetro, já que cada tipo de chamada tem seu próprio teto — seção 1);
  - `response_format={"type": "json_object"}` opcional (`json_mode=True`), usado no plano
    e no evidence extractor. Ajuda a obter JSON parseável, mas não garante o shape
    certo nem a sintaxe 100% (na Gemini já apareceu vírgula sobrando, tratada em
    `_extrair_json`); com o CoRAG o servidor aceita o parâmetro, mas o modelo pode gerar
    um JSON longo demais e ser cortado (ver "Limitações conhecidas");
  - **retry com backoff exponencial** (`2, 4, 8, 16, 30...` segundos, capado em 30) para
    `APIConnectionError` (inclui `APITimeoutError`), `RateLimitError`, e `APIStatusError`
    cujo `status_code` esteja em `_STATUS_RETENTAVEIS`
    (408/409/429/500/502/503/504/529) — outros status (ex. 400/401/403) sobem direto:
    não é erro transitório, não adianta tentar de novo. O orçamento é `PLAN_MAX_RETRIES`
    (apesar do nome, vale para as duas backends);
  - diagnóstico opcional (`verbose=True`, ou sempre que a resposta for truncada por
    `finish_reason == "length"`): imprime `[modelo] finish_reason=... completion_tokens=
    ... prompt_tokens=...` e o aviso `TRUNCATED`. O prefixo `[modelo]` mostra quem
    respondeu (ex. `[gemini-flash-lite-latest]` no plano, `[corag-8b]` nos hops).
- **`_chamar_gemini(...)`** — wrapper fino: valida `GEMINI_API_KEY` e chama
  `_chamar_llm(GEMINI_BASE_URL, ..., GEMINI_MODEL, ...)`. **Só `no_gerar_plano` usa.**
- **`_chamar_corag(...)`** — wrapper fino: valida `CORAG_BASE_URL`/`CORAG_API_KEY` e
  chama `_chamar_llm(CORAG_BASE_URL, ..., CORAG_MODEL, ...)`. **Usado por
  `no_gerar_subpergunta`, `no_evidence_extractor`, `no_gerar_subresposta` e
  `no_gerar_resposta_final`.**

Essa divisão (Gemini só no plano, CoRAG no resto) é uma escolha de configuração do
projeto — trocar o modelo de um nó é só trocar qual das duas funções ele chama.

Um cliente `OpenAI` novo é criado a cada chamada (não há estado compartilhado), o que
também é o que torna essas funções seguras de chamar concorrentemente de várias threads
(ver seção 10).

## 6. Retrieval: mini-corpus local + reserva HTTP

Duas implementações por trás da mesma interface (`(subpergunta, k) -> list[dict]` com
`doc_id/score/title/contents`), escolhidas por `SPMC_RETRIEVER`:

- **`_retrieve_mini_corpus`** (default, `"mini"`): busca densa E5 **in-process**, sem
  precisar de servidor. `_carregar_mini` carrega (uma vez, lazy) o dataset
  (`data/mini/corpus`) e os embeddings pré-computados
  (`data/mini/e5-large-index/e5-large-shard-0.pt`); `_carregar_encoder_e5` carrega o
  modelo `intfloat/e5-large-v2` e devolve uma função `encode` que reproduz a MESMA
  convenção usada para gerar aqueles embeddings (prefixo `"query: "`, average-pooling
  mascarado pela attention mask, normalização L2) — se a convenção não bater, a busca
  por produto interno não faz sentido. A busca em si é só `query @ embeddings.T` seguido
  de `topk`.
  - **`_MINI_LOCK`**: como as N cadeias rodam em threads paralelas (seção 10) e todas
    chamam retrieve, o carregamento lazy (~7s: dataset + modelo) precisa de um lock —
    sem ele, duas threads poderiam disparar o load ao mesmo tempo e uma ver `_MINI`
    parcialmente preenchido pela outra. `_carregar_mini` faz double-checked locking:
    checa sem lock, e se ainda não carregado, adquire o lock, checa de novo (outra
    thread pode ter carregado enquanto esperava) e só então chama
    `_carregar_mini_sem_lock`, que faz o carregamento de fato.
  - **Ordem de atribuição (bug já corrigido)**: o campo que o guard de
    `_carregar_mini` checa é `_MINI["corpus"]`, então ele é atribuído **por último**,
    depois de `emb` e `encode`. Na versão original era o primeiro: uma thread
    concorrente que passasse pelo check rápido (fora do lock) logo após `corpus` ser
    atribuído via `corpus != None`, concluía que o load tinha terminado e usava
    `encode`, que ainda era `None` → `TypeError: 'NoneType' object is not callable`
    dentro de `_retrieve_mini_corpus`. O erro só aparecia com mais de uma cadeia em
    paralelo, e de forma intermitente (dependia do timing das threads).
- **`_retrieve_e5_server`** (reserva, `"e5_server"`): consulta por HTTP um servidor E5
  já existente no repo (`src/search/start_e5_server_main.py`). Ainda incompleto: o
  servidor hoje só devolve `{doc_id, score}`, então `title`/`contents` ficam vazios até
  alguém ligar o lookup do corpus nessa função — deixado como está porque não há
  servidor rodando neste ambiente de desenvolvimento.

## 7. O estado do grafo: `SPMCState`

Um `TypedDict` — o "contrato" de dados que cada nó lê e escreve. Alguns campos
merecem destaque:

- `cadeia_atual`/`passo_atual`: ponteiros de posição. Como cada thread roda seu próprio
  subgrafo (uma cadeia por vez), dentro de uma chamada `cadeia_atual` é sempre `0` (só
  tem uma cadeia no `Plano` empacotado — ver `_executar_cadeia`); é `passo_atual` que
  realmente avança de `0` a `L-1`.
- `resolvidos`: dict `{"S_k": valor}` com os placeholders já descobertos NA CADEIA
  ATUAL — zerado no início de cada cadeia (não é compartilhado entre cadeias, cada uma
  tem seu próprio `S_1..S_L`).
- `pendentes_atual`: os placeholders que `passo.depends_on` precisa mas que ainda não
  estão em `resolvidos` — se não vazio, os nós seguintes pulam a chamada de LLM (não há
  entidade concreta para perguntar).
- `tentativas_subpergunta`/`retry_subpergunta`/`subpergunta_anterior`: controlam o
  mecanismo de retry de subpergunta (seção 9).
- `historico`: acumulador da cadeia ATUAL (lista de hops já concluídos, cada um com
  `k, subpergunta, subresposta, motivo, tripla, chunks, evidencias`).
- `cadeias_exec`: lista de cadeias já totalmente arquivadas (`{cadeia, descricao,
  hops}`) — é isso que `montar_contexto` consome.
- `log`: `Annotated[list[str], operator.add]` — o único campo com reducer customizado;
  LangGraph concatena (em vez de sobrescrever) o que cada nó retorna aqui, formando uma
  trilha de quais nós foram visitados.

## 8. Os nós que fazem trabalho real

Cada `no_*` recebe o `state` inteiro e devolve um `dict` com só as chaves que mudou —
LangGraph faz o merge no estado global (sobrescrevendo, exceto em `log`, que soma).

- **`no_gerar_plano`**: chama `_chamar_gemini` (o **único** nó que usa a Gemini) com
  `INSTRUCOES_PLANO` formatado com `N` e `L`. Se o JSON não validar contra `Plano` (`ValidationError`/`ValueError`), tenta de
  novo até `PLAN_JSON_RETRIES` vezes antes de desistir (devolvendo o último texto mesmo
  assim — `no_parsear_plano` relança o erro com o JSON completo no stderr, pra
  diagnóstico).
- **`no_parsear_plano`**: valida o JSON cru contra `Plano` de verdade. Faz checagens
  "suaves" (avisa no stderr se o número de cadeias/passos fugir do esperado, ou se
  alguma tripla vier incompleta) mas não aborta por causa delas — só a falha de
  validação do Pydantic é fatal. Não inicializa mais `cadeia_atual`/`resolvidos`/etc.
  (isso hoje é feito por cadeia, dentro de `_executar_cadeia`).
- **`no_resolver_placeholder`**: pega a tripla do passo atual e substitui os
  placeholders já resolvidos (`_resolver_tripla`). No primeiro passo de uma cadeia
  (`passo_atual == 0`), zera `resolvidos` e `historico` (cada cadeia começa do zero).
  Calcula `pendentes_atual` como as dependências de `passo.depends_on` que ainda faltam
  — **não** confundir com o `pendentes` que `_resolver_tripla` devolve, que também
  inclui o placeholder que o PRÓPRIO passo define (isso é normal, não é uma dependência
  faltando).
- **`no_gerar_subpergunta`**: se `pendentes_atual` não estiver vazio, pula direto (sem
  gastar tokens — o passo vai virar `SEM_RESPOSTA` de qualquer jeito). Senão, monta o
  prompt com o contexto da cadeia, o texto-guia, o "target of THIS step" (via
  `_tipo_placeholder`) e, se for um retry, uma instrução extra pedindo uma formulação
  diferente da anterior. Chama `_chamar_corag` com `INSTRUCOES_SUBPERGUNTA`.
- **`no_retrieve`**: também pula se `pendentes_atual` não estiver vazio. Escolhe
  `_retrieve_mini_corpus` ou `_retrieve_e5_server` conforme `SPMC_RETRIEVER`.
- **`no_evidence_extractor`**: pula a chamada ao LLM se não vieram chunks. Chama
  `_chamar_corag` (com `json_mode=True` e `max_tokens=EVIDENCE_MAX_TOKENS`) com
  `INSTRUCOES_EVIDENCE`, valida a saída contra `Evidencias` — se falhar o parse (por
  exemplo, JSON cortado por truncamento), usa lista vazia em vez de quebrar o fluxo.
  Esse fallback é seguro, mas descarta a evidência do hop inteiro — ver "Limitações
  conhecidas".
- **`no_gerar_subresposta`**: o nó mais denso — decide se/como responder o hop, e
  também é quem avança os ponteiros. Ver seção 9.

## 9. `no_gerar_subresposta` em detalhe

Três casos, na ordem:

1. **`pendentes`** não vazio → `SEM_RESPOSTA`, `motivo="dependencia_ausente"` (nem
   tenta chamar o LLM — já se sabia de antemão que não dava, por `no_resolver_
   placeholder`/`no_gerar_subpergunta`/`no_retrieve` terem pulado).
2. **sem evidências** (o evidence extractor não achou nada) → `SEM_RESPOSTA`,
   `motivo="sem_evidencia"` (também sem chamar o LLM — não haveria nada pra ele
   responder).
3. **tem evidências** → chama `_chamar_corag` com `INSTRUCOES_SUBRESPOSTA`; se a saída
   for literalmente `SEM_RESPOSTA`, `motivo="sem_resposta_llm"` (buscou, extraiu, mas o
   LLM mesmo assim não conseguiu responder); senão `motivo=None` (sucesso de verdade).

**Retry**: se `motivo` for um dos dois casos "genuinamente tentados" (`sem_evidencia`
ou `sem_resposta_llm` — não `dependencia_ausente`, que uma subpergunta diferente nunca
resolveria) e ainda houver orçamento (`tentativas_subpergunta < SUBQ_RETRY_MAX`), a
função devolve `retry_subpergunta=True` **sem avançar o ponteiro nem gravar no
histórico** — o roteador (`_proximo_apos_subresposta`) manda de volta pro
`gerar_subpergunta`, que vai gerar uma formulação diferente para o MESMO passo.

Quando o passo realmente conclui (com sucesso ou tendo esgotado os retries):

- só grava `resolvidos[passo.define]` se `motivo is None` (nunca substitui um
  placeholder por `"no relevant information found"`);
- **recalcula a tripla do zero** a partir do template original (`passo.triple`), usando
  o `resolvidos` já atualizado — importante porque a `tripla_resolvida` que veio de
  `no_resolver_placeholder` foi montada ANTES de sabermos a subresposta deste próprio
  passo, então nunca continha o valor que acabamos de descobrir. Recalculando aqui, o
  placeholder que este passo define também sai resolvido no registro do hop (bug já
  corrigido: antes disso, o histórico e o contexto final mostravam o placeholder
  `[S_2: ...]` intacto mesmo depois de já saber o valor);
- monta o `hop` (dict com `k, subpergunta, subresposta, motivo, tentativas, tripla,
  chunks, evidencias`) e acrescenta a `historico`;
- avança `passo_atual` (mesma cadeia) ou, se era o último passo, arquiva
  `{cadeia, descricao, hops}` em `cadeias_exec` e passa para `cadeia_atual + 1` — dentro
  do subgrafo por cadeia, isso só serve pra sinalizar "acabou" ao roteador (ver seção
  10), já que só existe uma cadeia por invocação.

## 10. Execução paralela das N cadeias

As N cadeias são independentes até a agregação final — não há razão para rodá-las em
sequência. A solução: um **subgrafo dedicado a UMA cadeia**
(`construir_grafo_cadeia`), reusando os mesmos nós de `resolver_placeholder` a
`gerar_subresposta` e o mesmo roteador (`_proximo_apos_subresposta`), mas com a rota
`"agregar"` indo direto para `END` do subgrafo (em vez de seguir para
`montar_contexto`, que agora só roda uma vez, no nível superior, depois que TODAS as
cadeias tiverem voltado).

- **`_grafo_cadeia()`**: compila `construir_grafo_cadeia()` uma única vez e cacheia
  (double-checked locking com `_GRAFO_CADEIA_LOCK`, pois várias threads podem chegar
  aqui simultaneamente na primeira chamada) — todas as cadeias/threads reusam a mesma
  instância compilada.
- **`_executar_cadeia(questao, cadeia, L, verbose)`**: roda dentro de uma thread.
  Empacota a `cadeia` sozinha dentro de um `Plano` de um elemento só (os nós existentes
  esperam `state["plano_estruturado"].chains[cadeia_atual]`; `overview/entities/
  relations` não são lidos por eles, ficam vazios) e invoca o subgrafo. Devolve
  `estado["cadeias_exec"][0]` — o hop arquivado, no mesmo formato que ia direto em
  `cadeias_exec` na versão sequencial antiga.
- **`no_executar_cadeias`**: o nó de nível superior. Abre um `ThreadPoolExecutor` com
  `min(SPMC_CHAIN_THREADS, N)` workers, submete uma `_executar_cadeia` por cadeia, e
  coleta os resultados na ordem original das cadeias (usando o índice do `enumerate`,
  não a ordem de conclusão de `as_completed`).

Cada thread faz suas próprias chamadas de API concorrentemente, então mais threads
significam mais requisições simultâneas ao endpoint remoto do CoRAG (que serve os hops)
— e, marginalmente, à Gemini apenas no plano, que roda *antes* do fan-out, uma vez por
pergunta. Um valor alto de `SPMC_CHAIN_THREADS` pode aumentar timeouts/erros
transitórios no endpoint remoto (tratados pelo retry com backoff de `_chamar_llm`) —
por isso é configurável por env em vez de fixo no código.

## 11. Agregação e resposta final

- **`no_montar_contexto`**: percorre `cadeias_exec` e, POR CADEIA, filtra só os hops
  com `motivo is None` (respondidos de verdade) — hops sem resposta não trazem
  informação nova, só ruído (`"no relevant information found"`) para o LLM final. Se
  uma cadeia inteira não teve nenhum hop respondido, ela simplesmente não aparece no
  contexto agregado. Formata cada hop como `Step k: pergunta -> resposta` + a tripla.
- **`no_gerar_resposta_final`**: chama `_chamar_corag` com `INSTRUCOES_RESPOSTA_FINAL`
  + a pergunta original + o contexto agregado, produzindo `state["resposta_final"]`.
  Note que ele recebe **só** o contexto das cadeias (sub-perguntas, sub-respostas e
  triplas resolvidas) — nenhum documento bruto.

## 12. `_proximo_apos_subresposta` — o roteador

Função usada em `add_conditional_edges` tanto no subgrafo por cadeia quanto (hoje)
implicitamente pensada para uma versão sequencial — mas como
`state["plano_estruturado"].chains` sempre tem exatamente 1 elemento dentro de uma
`_executar_cadeia` (a cadeia que aquela thread está rodando), o `"agregar"` sempre
fecha essa cadeia e cai no `END` do subgrafo, nunca num "próxima cadeia" global.

Três saídas possíveis:
- `"retry_subpergunta"`: `state["retry_subpergunta"]` é `True` — volta pro
  `gerar_subpergunta` sem avançar ponteiro.
- `"proximo_passo"`: ainda há mais passos nesta cadeia (`cadeia_atual < len(chains)`,
  que dentro do subgrafo equivale a "ainda não passamos do único elemento") — volta pro
  `resolver_placeholder` do próximo passo.
- `"agregar"`: a cadeia terminou.

## 13. `construir_grafo` — o grafo de nível superior

```python
START → gerar_plano → parsear_plano → executar_cadeias → montar_contexto
      → gerar_resposta_final → END
```

Puramente linear, sem loop — diferente do subgrafo por cadeia. O limite de recursão do
LangGraph (`recursion_limit`) pode ficar baixo (20) porque não há ciclo aqui; quem
precisa de um limite calculado em função de `L`/`SUBQ_RETRY_MAX` é o subgrafo por
cadeia, calculado em `_executar_cadeia`.

## 14. `__main__` — script de teste manual (uma pergunta)

Roda o pipeline inteiro fim-a-fim numa pergunta de exemplo fixa (escolhida porque o
mini-corpus local tem os documentos que sustentam os dois hops necessários: o filme
Royal Flash e o ator Oliver Reed). Comando:

```bash
GEMINI_MODEL=gemini-flash-lite-latest SPMC_CHAIN_THREADS=2 .venv/bin/python src/SPMC/spmc.py
```

Imprime, em ordem (tudo em inglês):

1. os diagramas mermaid dos dois grafos (nível superior e subgrafo por cadeia);
2. o cabeçalho `Running (plan: <modelo Gemini>, hops: <modelo CoRAG>, N=.. chains x
   L=.. steps, .. thread(s) in parallel)`;
3. o JSON cru do plano (primeiros 800 caracteres);
4. quantos nós foram visitados no grafo de nível superior e a validação de contagem
   N/L;
5. o fluxo de execução detalhado por cadeia (sub-question, chunks, evidence,
   sub-answer + reason/attempts, triple — por step);
6. o contexto agregado (saída de `montar_contexto`);
7. a resposta final.

Todas as variáveis de ambiente relevantes (`SPMC_N`, `SPMC_L`, `SPMC_CHAIN_THREADS`,
`SPMC_RETRIEVER`, `SPMC_TOP_K`, `MINI_BASE_DIR`, `GEMINI_MODEL`, `BASE_URL`, `API_KEY`,
`MODEL_NAME`, `EVIDENCE_MAX_TOKENS`) podem ser passadas na linha de comando sem editar o
arquivo.

## 15. `rodar_hotpotqa30.py` — driver das 30 perguntas do hotpotqa

`src/SPMC/rodar_hotpotqa30.py` roda o SPMC sobre o mesmo conjunto de 30 perguntas que
`src/reproducao/dynamic_chain.py` usa (`data/mini/questions/hotpotqa`, carregado via
`carregar_perguntas` de `reproducao/best_of_n.py`), sobre o mesmo mini-corpus — o que
torna a comparação entre as duas estratégias direta (mesmas perguntas, mesmos
documentos disponíveis).

```bash
PYTHONPATH=src GEMINI_MODEL=gemini-flash-lite-latest \
  SPMC_N=4 SPMC_L=6 SPMC_CHAIN_THREADS=2 \
  .venv/bin/python src/SPMC/rodar_hotpotqa30.py
```

(`PYTHONPATH=src` é necessário porque o script importa `from SPMC import spmc` e
`from reproducao.best_of_n import ...` — `src/` vira a raiz dos pacotes, como em
todos os scripts deste repositório.)

- **`executar_spmc_dataset(dataset, N, L)`**: percorre as perguntas **sequencialmente**
  (o paralelismo já existe *dentro* de cada pergunta, entre as N cadeias, via
  `SPMC_CHAIN_THREADS` — diferente do `dynamic_chain.py`, cujo `NUM_THREADS`
  paraleliza *entre* perguntas). Para cada uma, invoca o grafo de nível superior
  (`verbose=False`) e guarda `{query, answers, estrategia="spmc", n, max_path_length,
  prediction}` — o formato que `calcular_metricas` espera (`answers` e `prediction`).
- **Saída**: `data/rag_log_spmc_hotpotqa.jsonl` (uma linha por pergunta, escrito só ao
  final) e, no stdout, EM/F1 via `calcular_metricas` (a mesma função de
  `best_of_n.py` que o `dynamic_chain.py` usa). O stdout é *block-buffered* quando
  redirecionado para arquivo — para acompanhar o progresso de uma rodada longa, olhe o
  stderr (logs HTTP/erros), não o stdout.
- **Não há recuperação parcial**: se o processo for interrompido, nada é salvo (o log é
  escrito uma vez, no fim).

Resultados medidos em 2026-09-17 (30 perguntas, N=4, L=6, 2 threads; plano com
`gemini-flash-lite-latest`, hops com `corag-8b`):

| Estratégia | Config | EM | F1 | Tempo |
|---|---|---|---|---|
| `dynamic_chain` | n=4, max_path_length=6 | 56.667 | 70.843 | ~12 min |
| `spmc` | `EVIDENCE_MAX_TOKENS=1024` | 36.667 | 54.367 | 1h00min |
| `spmc` | `EVIDENCE_MAX_TOKENS=2048` | 43.333 | 55.319 | 54min |

## Limitações conhecidas

- **Truncamento do evidence extractor com o CoRAG.** O CoRAG às vezes lista muitas
  triplas (ou triplas longas) e bate no teto `EVIDENCE_MAX_TOKENS`
  (`finish_reason=length`, aviso `TRUNCATED` no log). O JSON sai cortado no meio de uma
  string → `Evidencias.model_validate_json` falha → `no_evidence_extractor` cai no
  fallback `evidencias = []` → `no_gerar_subresposta` marca `motivo="sem_evidencia"` →
  como esse motivo é retentável, o roteador dispara um retry de subpergunta (mais uma
  rodada de 3 chamadas ao CoRAG para o mesmo passo, com `SUBQ_RETRY_MAX=1`). Efeitos:
  (1) a evidência do hop é descartada por inteiro, mesmo que a maior parte das triplas
  antes do corte fosse válida; (2) esses hops são lentos — a chamada truncada gerou o
  teto inteiro de tokens, e o retry pode repetir o padrão. Subir o teto tem retorno
  decrescente (perguntas com muito conteúdo estouraram até 4000). Mitigações ainda **não
  implementadas**: limitar o nº de triplas no prompt (`INSTRUCOES_EVIDENCE`) e/ou
  recuperar as triplas completas de um JSON truncado em vez de descartar tudo.
- **`gerar_resposta_final` não recebe documentos brutos** — só o contexto das cadeias.
  O `dynamic_chain.py`, por padrão, passa também os `context_doc_ids` do dataset (top-100
  pré-computado) no prompt da resposta final, o que é uma diferença estrutural entre as
  duas estratégias.
- **Placeholder do próprio passo às vezes ausente/trocado no plano.** O reforço em
  `INSTRUCOES_PLANO` reduziu, mas não eliminou, casos em que um passo com `define: "S_3"`
  não contém `[S_3: ...]` na própria tripla (ex. reutiliza o token de `S_2`).
  Nesses casos `_tipo_placeholder` devolve `None` e o "target of THIS step" não é
  injetado na sub-pergunta. Não há validação de backstop em `no_parsear_plano`.
- **O formato do placeholder é rígido.** `_RE_PLACEHOLDER` e `_RE_PLACEHOLDER_TIPO` só
  reconhecem exatamente `[S_k: tipo: ??]`. Um modelo de plano que escreva `[S_1]` (sem
  tipo) ou `[S_1: type: Valor]` (com valor em vez de `??`) valida no schema Pydantic
  (os campos são strings livres), mas quebra a resolução de placeholders na execução.
  Foi o que se observou num teste de geração de plano com `gpt-oss:20b-cloud` via
  Ollama (fora do `spmc.py`, apenas experimento).
- **`_retrieve_e5_server` incompleto** — ver seção 6.
