"""SPMC-RAG sem placeholders: plano, cadeias de raciocinio e avaliacao no mini dataset.

O LLM (gpt-oss:120b via Ollama) gera um plano com N cadeias alternativas de L passos,
cada passo com um texto guia. As cadeias rodam em paralelo e de forma independente:
em cada passo, a subpergunta e gerada a partir do contexto acumulado da cadeia (plano,
entidades, relacionamentos e subperguntas/subrespostas anteriores), os top-k chunks sao
recuperados do mini-corpus por busca densa E5 e a subresposta e gerada a partir deles.
Por fim, a resposta final e produzida a partir de todas as cadeias e dos FINAL_DOCS chunks
mais relevantes para a pergunta, dentre os recuperados pelas cadeias.

`rodar_mini_dataset` executa o fluxo sobre as 30 perguntas de uma tarefa do mini
dataset e grava o log por pergunta e as metricas EM/F1 em `data/spmc_runs/no_placeholder/`.
"""

import argparse
import datetime as dt
import json
import logging
import operator
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/SPMC/SPMC_no_placeholder/ -> raiz do repo

# Todo o fluxo usa o gpt-oss:120b servido pelo daemon Ollama local, via endpoint
# OpenAI-compativel (reusa a lib openai). O Ollama local nao exige chave.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("SPMC_MODEL", "gpt-oss:120b-cloud")

N_DEFAULT = 4  # cadeias de raciocinio por plano
L_DEFAULT = 6  # passos por cadeia (mesmo L do repo: max_path_length = 6)
# Tetos de tokens -- folgados porque o gpt-oss gasta tokens de raciocinio antes da resposta.
PLAN_MAX_TOKENS = int(os.getenv("PLAN_MAX_TOKENS", "8192"))
SUBQ_MAX_TOKENS = int(os.getenv("SUBQ_MAX_TOKENS", "2048"))
SUBA_MAX_TOKENS = int(os.getenv("SUBA_MAX_TOKENS", "2048"))
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS", "2048"))
# Tentativas extras de gerar o plano quando o JSON sai invalido ou fora do formato N x L.
PLAN_JSON_RETRIES = int(os.getenv("PLAN_JSON_RETRIES", "2"))
# Quando a subresposta de um passo vem SEM_RESPOSTA, gera uma subpergunta NOVA para o
# mesmo passo (com temperatura > 0, para variar a formulacao) e repete retrieve +
# subresposta -- ate SUBQ_RETRY_MAX vezes alem da primeira tentativa.
SUBQ_RETRY_MAX = int(os.getenv("SUBQ_RETRY_MAX", "3"))
SUBQ_RETRY_TEMPERATURE = float(os.getenv("SUBQ_RETRY_TEMPERATURE", "0.7"))

# Retrieve: denso E5 in-process sobre o mini-corpus (data/mini/).
MINI_BASE_DIR = Path(os.getenv("MINI_BASE_DIR", str(REPO_ROOT / "data" / "mini")))
E5_MODEL = "intfloat/e5-large-v2"  # mesmo modelo que gerou data/mini/e5-large-index
TOP_K = int(os.getenv("SPMC_TOP_K", "5"))
CHUNK_CHARS = 600  # trunca cada chunk no prompt da subresposta e no da resposta final
# Chunks (dentre os recuperados por todas as cadeias) que entram no prompt da resposta
# final, escolhidos por similaridade E5 com a pergunta original. 0 desliga o bloco.
FINAL_DOCS = int(os.getenv("SPMC_FINAL_DOCS", "20"))
SEM_RESPOSTA = "No relevant information found"

# Execucao sobre o mini dataset (30 perguntas por tarefa em data/mini/questions/<task>).
TASKS = ("hotpotqa", "2wikimultihopqa", "musique", "bamboogle")
RUNS_DIR = REPO_ROOT / "data" / "spmc_runs" / "no_placeholder"

# Erros transitorios (rede, 429, 5xx) que o LangGraph retenta nos nos que chamam o LLM.
RETRY_LLM = RetryPolicy(
    max_attempts=5,
    initial_interval=2.0,
    retry_on=(APIConnectionError, RateLimitError, InternalServerError),
)


PLAN_PROMPT = """\
You are the Plan Generator of SPMC-RAG, a multi-hop retrieval-augmented question
answering system. Given a complex multi-hop query, you produce a reasoning PLAN made
of {N} alternative reasoning chains, each with exactly {L} steps. The plan is a
skeleton, not an answer key: sub-questions, retrieval and answers are produced later
by an executor, which runs each chain step by step, turning each step's guide text
into a sub-question and answering it with the documents it retrieves. When it
executes a step, the executor already knows the answers found by the earlier steps
of the same chain.

## Key concepts

Step: a guide text of one or two sentences stating what the step must discover.
The executor turns it into a sub-question, so it must be specific and
self-contained:
- name the query entities exactly as they appear in the query;
- state the relation being looked up (e.g. "who directed", "the city where ...
  was born", "the country where ... is located");
- make clear the kind of value to be found (a person, a film, a city, a date, a
  number, a yes/no answer, etc.);
- refer to values found by earlier steps of THIS chain by the step that found
  them and what they are (e.g. "the director identified in step 1"), never by a
  concrete value.

Chain: a sequence of steps in dependency order. A step may only use what the
query states and what earlier steps of the same chain found.

## Procedure (think it through, but output only the JSON)

1. Identify the key entities, the relations between them and the type of the
   final answer.
2. Identify the intermediate (bridge) facts that must be discovered before the
   answer can be found.
3. Build {N} chains that reach the same final answer by different routes. Chains
   may differ by:
   - using a different bridge entity or intermediate fact;
   - resolving the hops in a different order (e.g. starting from another entity
     of the query);
   - phrasing a relation differently (inverse relation, synonym, broader or more
     specific relation), which leads retrieval to different documents.
   Two chains whose steps ask for the same things in the same order are
   duplicates, not alternatives.
4. Write each chain's steps in dependency order, then check every rule below
   before answering.

## Rules

Structure
- Exactly {N} chains, with "id" from 1 to {N}; each chain has exactly {L} steps,
  with "k" from 1 to {L}.
- Every step has exactly two keys: "k" and "guide_text" (a non-empty string).
  Do not add triples, placeholders or any other field.

Values and dependencies
- NEVER write in a guide text a value that should be retrieved (a name, a date,
  a place, a number...), even if you already know it. Values are found later by
  retrieval; a guide text that already contains the value skips retrieval and
  can propagate a wrong answer to the following steps.
- Each step discovers at most one new fact.
- A step may only refer to steps with a smaller "k" in the same chain, and must
  say explicitly which of them it uses (e.g. "the city found in step 2").

Number of hops
- If the query needs fewer than {L} hops, use the extra steps BEFORE the last one
  to verify or disambiguate entities already found (e.g. confirm an entity's
  occupation, or a date that distinguishes homonyms). Never produce fewer than
  {L} steps, and never pad with repeated steps.

Final step
- Step {L} produces the final answer, of the type given in "answer_type".
- For comparison or yes/no queries ("which was released first?", "are both X?"),
  the earlier steps retrieve each attribute to be compared and the last step
  compares them, e.g. with 3 steps: "Compare the release dates found in steps 1
  and 2 and determine which of Film A and Film B was released first."

## Output

Respond ONLY with a valid JSON object: no markdown, no ``` fences, no comments,
no text before or after it. Fields:
- "overview": what the query requires discovering, in one or two sentences.
- "entities": key entities of the query.
- "relations": relations involved.
- "answer_type": type of the final answer, as a short, generic, lowercase
  semantic type: person, film, book, city, country, organization, date, year,
  number, yes/no, etc.
- "chains": the {N} chains; each has "id", "description" (how this route differs
  from the others) and "steps".

Illustrative example with 2 chains and 3 steps (your plan must have {N} chains
and {L} steps). Query: "In which country was the director of Titanic (1997) born?"
Note that the plan never writes the director's name, even though it is well known.

{{
  "overview": "Find who directed Titanic (1997), then the country where that person was born.",
  "entities": ["Titanic (1997)"],
  "relations": ["directed by", "place of birth", "located in country"],
  "answer_type": "country",
  "chains": [
    {{
      "id": 1,
      "description": "Finds the director, then the birth city, then the country containing that city.",
      "steps": [
        {{
          "k": 1,
          "guide_text": "Identify the person who directed the film Titanic (1997)."
        }},
        {{
          "k": 2,
          "guide_text": "Find the city where the director identified in step 1 was born."
        }},
        {{
          "k": 3,
          "guide_text": "Find the country in which the birth city found in step 2 is located."
        }}
      ]
    }},
    {{
      "id": 2,
      "description": "Finds the director through the inverse relation, confirms the identity, and asks directly for the country of birth.",
      "steps": [
        {{
          "k": 1,
          "guide_text": "Identify the director credited for the film Titanic (1997)."
        }},
        {{
          "k": 2,
          "guide_text": "Confirm that the person found in step 1 is a film director, to rule out homonyms."
        }},
        {{
          "k": 3,
          "guide_text": "Find the country where the director confirmed in step 2 was born."
        }}
      ]
    }}
  ]
}}
"""

SUBQ_PROMPT = """\
You are the sub-question generator of SPMC-RAG, a multi-hop retrieval-augmented
question answering system. You receive the original question, the accumulated
context of ONE reasoning chain (the plan overview, its key entities and relations,
and the sub-questions and sub-answers of the previous steps of this chain) and the
guide text of the current step.

Write ONE natural-language sub-question that carries out the current step. It will
be used to retrieve documents, so:
- ask for exactly what the guide text asks for, one fact only; do not merge in what
  later steps will look for;
- make it self-contained: when the guide text refers to a previous step (e.g. "the
  director identified in step 1"), replace the reference with that step's
  sub-answer from the context;
- if that sub-answer is "No relevant information found", phrase the sub-question
  with what the original question states instead;
- write entity names exactly as they appear in the question or in the context;
- do not answer the sub-question.

Respond with only the sub-question.
"""

SUBA_PROMPT = f"""\
You answer ONE sub-question of a multi-hop reasoning chain using ONLY the retrieved
passages given below; do not use outside knowledge and do not invent facts.

- Answer with a short phrase or entity name, not a full sentence, and no explanation.
- If the passages do not contain the answer, respond with exactly: {SEM_RESPOSTA}

Respond with only the answer.
"""

FINAL_PROMPT = f"""\
You produce the FINAL answer to the original multi-hop question of SPMC-RAG. You are
given two kinds of evidence:

1. Retrieved passages: the passages most relevant to the original question among all
   the documents retrieved while executing the reasoning chains.
2. Reasoning chains: several independent chains that each tried to reach the answer
   by a different route; for every chain, only the steps that found an answer are
   listed, with their sub-question and sub-answer.

- Use the reasoning chains to follow the multi-hop reasoning (which entity leads to
  which), and the passages to confirm, correct or complete it.
- The sub-answers were generated by an LLM and may be wrong or incomplete; when a
  sub-answer contradicts the passages, trust the passages. A passage may also
  contain a fact that no chain extracted.
- The chains may be redundant or phrase the same fact differently: cross-check them
  and rely on the chain(s) that actually reached a conclusion.
- Use only the passages and the chains; do not use outside knowledge and do not
  invent facts beyond what they establish.
- Answer the original question directly: a short phrase or entity name, not a full
  sentence, and no explanation.
- If the evidence does not provide enough information, respond with exactly:
  {SEM_RESPOSTA}

Respond with only the answer.
"""

# ---------------------------------------------------------------------------
# Estrutura do plano (Pydantic)
# ---------------------------------------------------------------------------

class Passo(BaseModel):
    """Um passo (hop) de uma cadeia: id + texto guia.

    Parameters:
        id: Posicao do passo na cadeia, de 1 a L. Aceita tambem a chave `"k"`, que e a
            emitida pelo `PLAN_PROMPT`.
        guide_text: O que este passo deve descobrir; vira a subpergunta na execucao.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(validation_alias=AliasChoices("id", "k"))  # o prompt emite "k"; 1..L
    guide_text: str = Field(min_length=1)                     # o que este passo deve descobrir


class Cadeia(BaseModel):
    """Uma rota de raciocinio: L passos em ordem de dependencia.

    Parameters:
        id: Identificador da cadeia, de 1 a N.
        description: Como esta rota difere das outras cadeias do plano.
        steps: Os L passos da cadeia.
    """

    id: int                                 # 1..N
    description: str                        # como esta rota difere das outras
    steps: list[Passo]                      # L passos


class Plano(BaseModel):
    """Plano global: descricao + entidades-chave + relacionamentos + N cadeias.

    Validado a partir do JSON devolvido pelo LLM em `no_gerar_plano`; as contagens N e L
    sao conferidas a parte, pois so sao conhecidas em tempo de execucao.

    Parameters:
        overview: Descricao do que a pergunta exige descobrir.
        entities: Entidades-chave da pergunta.
        relations: Relacionamentos envolvidos.
        answer_type: Tipo da resposta final (ex. person, country, date). Defaults to "".
        chains: As N cadeias alternativas.
    """

    overview: str                           # descricao do que a pergunta exige descobrir
    entities: list[str]                     # entidades-chave
    relations: list[str]                    # relacionamentos
    answer_type: str = ""                   # tipo da resposta final (o prompt pede; opcional)
    chains: list[Cadeia]                    # N cadeias


def _checar_formato(plano: Plano, N: int, L: int) -> str | None:
    """N e L so sao conhecidos em tempo de execucao, entao ficam fora do schema.
    Devolve a descricao do problema, ou None se o plano tem N cadeias x L passos."""
    if len(plano.chains) != N:
        return f"expected {N} chains, got {len(plano.chains)}"
    for c in plano.chains:
        ids = [p.id for p in c.steps]
        if ids != list(range(1, L + 1)):
            return f"chain {c.id}: expected step ids 1..{L}, got {ids}"
    return None



# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _extrair_json(texto: str) -> str:
    """Recorta do primeiro '{' ao ultimo '}' (tolera cercas ``` e texto solto) e remove
    virgulas sobrando antes de '}'/']'."""
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim < inicio:
        raise ValueError(f"nenhum objeto JSON encontrado na resposta: {texto[:200]!r}")
    return re.sub(r",\s*([}\]])", r"\1", texto[inicio : fim + 1])


def _chamar_llm(
    system: str, user: str, *, max_tokens: int, json_mode: bool = False, temperature: float = 0
) -> tuple[str, dict]:
    """Uma chamada ao gpt-oss via Ollama. Erros transitorios (rede, 429, 5xx) sobem e
    sao retentados pelo RetryPolicy do no (RETRY_LLM).

    Devolve (resposta, chamada): `chamada` vai pro log com tudo o que foi enviado e
    recebido -- os prompts system/user completos, a resposta, o raciocinio do modelo
    (campo `reasoning` que o Ollama devolve pro gpt-oss), finish_reason, tokens e tempo."""
    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL, max_retries=0)
    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    t0 = time.time()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    escolha = resp.choices[0]
    if escolha.finish_reason == "length":
        print(f"[{LLM_MODEL}] response TRUNCATED: increase max_tokens ({max_tokens})", file=sys.stderr)
    texto = (escolha.message.content or "").strip()
    chamada = {
        "model": LLM_MODEL,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        "temperature": temperature,
        "system": system,
        "user": user,
        "resposta": texto,
        "raciocinio": (escolha.message.model_extra or {}).get("reasoning"),
        "finish_reason": escolha.finish_reason,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "tempo_s": round(time.time() - t0, 2),
    }
    return texto, chamada


# ---------------------------------------------------------------------------
# Retrieve: denso E5 sobre o mini-corpus
# ---------------------------------------------------------------------------

# Carregados lazy na 1a chamada a _retrieve. As cadeias rodam em paralelo, entao o
# carregamento (~segundos: dataset + modelo E5) fica sob lock.
_MINI: dict = {"corpus": None, "emb": None, "encode": None}
_MINI_LOCK = threading.Lock()


def _carregar_mini() -> None:
    with _MINI_LOCK:
        if _MINI["corpus"] is not None:
            return
        import torch
        from datasets import Dataset
        from transformers import AutoModel, AutoTokenizer

        corpus_dir = MINI_BASE_DIR / "corpus"
        # Indice pode vir dividido em varios shards (data/mini_gold, pra caber no GitHub):
        # concatena na ordem numerica do shard, que e a ordem das linhas do corpus.
        shard_paths = sorted(
            (MINI_BASE_DIR / "e5-large-index").glob("*-shard-*.pt"),
            key=lambda p: int(re.search(r"-shard-(\d+)\.pt$", p.name).group(1)),
        )
        if not corpus_dir.exists() or not shard_paths:
            raise RuntimeError(
                f"mini-corpus nao encontrado em {MINI_BASE_DIR}. Gere com:\n"
                f"  PYTHONPATH=src python src/reproducao/construir_mini_corpus.py"
            )

        tok = AutoTokenizer.from_pretrained(E5_MODEL)
        mod = AutoModel.from_pretrained(E5_MODEL, torch_dtype=torch.float32).eval()

        # Mesma convencao que gerou o indice: prefixo "query: ", avg-pool mascarado, L2.
        @torch.no_grad()
        def encode(textos: list[str]):
            lote = tok(
                [f"query: {t}" for t in textos],
                max_length=512, padding=True, truncation=True, return_tensors="pt",
            )
            saida = mod(**lote).last_hidden_state
            mask = lote["attention_mask"].unsqueeze(-1).float()
            media = (saida * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return torch.nn.functional.normalize(media, p=2, dim=-1)

        _MINI["emb"] = torch.cat(
            [torch.load(str(p), weights_only=True, map_location="cpu") for p in shard_paths]
        ).float()
        _MINI["encode"] = encode
        _MINI["corpus"] = Dataset.load_from_disk(str(corpus_dir))


def _retrieve(subpergunta: str, k: int) -> list[dict]:
    """Top-k do mini-corpus por produto interno sobre os embeddings E5 pre-computados."""
    import torch

    _carregar_mini()
    corpus, emb = _MINI["corpus"], _MINI["emb"]
    scores = (_MINI["encode"]([subpergunta]) @ emb.t()).squeeze(0)
    top = torch.topk(scores, k=min(k, scores.numel()))
    resultados = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        doc = corpus[int(idx)]
        resultados.append(
            {
                "idx": int(idx),  # indice local no mini-corpus (linha de emb / corpus)
                "doc_id": int(doc.get("orig_doc_id", idx)),
                "score": float(score),
                "title": doc.get("title", ""),
                "contents": doc.get("contents", ""),
            }
        )
    return resultados


# ---------------------------------------------------------------------------
# Execucao de uma cadeia (subgrafo): subpergunta -> retrieve -> subresposta, L vezes
# ---------------------------------------------------------------------------

class CadeiaState(TypedDict):
    """Estado do subgrafo que executa UMA cadeia, passo a passo.

    Cada cadeia recebe o seu proprio estado, entao o contexto acumulado de uma nao
    interfere nas outras.

    Parameters:
        questao: Pergunta original.
        plano: Plano completo; fornece overview, entidades e relacionamentos ao contexto.
        cadeia: A cadeia executada por este subgrafo.
        passo_atual: Indice do passo em execucao, de 0 a L-1.
        subpergunta_atual: Subpergunta gerada para o passo atual.
        llm_subpergunta: Registro da chamada ao LLM que gerou a subpergunta atual (ver
            `_chamar_llm`); vai pro historico junto com o passo.
        chunks_atual: Top-k chunks recuperados para a subpergunta atual.
        tentativa_atual: Tentativa do passo atual: 0 na primeira subpergunta, 1..
            `SUBQ_RETRY_MAX` nas subperguntas novas geradas apos um SEM_RESPOSTA.
        tentativas_falhas: Tentativas do passo atual que deram SEM_RESPOSTA (mesmo
            formato do registro do passo); vao pro historico quando o passo conclui.
        historico: Um registro por passo concluido (`id`, `guide_text`, `subpergunta`,
            `chunks`, `docs`, `chunks_detalhe`, `subresposta`, `llm`, `n_tentativas`,
            `tentativas_falhas`), acumulado pelo reducer `operator.add`.
    """

    questao: str
    plano: Plano                            # overview / entities / relations do contexto
    cadeia: Cadeia                          # so a cadeia desta execucao
    passo_atual: int                        # indice do passo em execucao, 0..L-1
    subpergunta_atual: str
    llm_subpergunta: dict                   # chamada ao LLM da subpergunta atual (p/ o log)
    chunks_atual: list[dict]                # top-k chunks da subpergunta atual
    tentativa_atual: int                    # 0 = 1a subpergunta do passo; >0 = subpergunta nova
    tentativas_falhas: list[dict]           # tentativas do passo atual que deram SEM_RESPOSTA
    historico: Annotated[list[dict], operator.add]  # um registro por passo concluido


def _contexto_acumulado(plano: Plano, historico: list[dict]) -> str:
    """Contexto acumulado de UMA cadeia (em ingles, vai pro LLM): descricao do plano,
    entidades, relacionamentos e as subperguntas/subrespostas dos passos anteriores."""
    if historico:
        anteriores = "\n".join(
            f"Step {h['id']}:\n"
            f"  sub-question: {h['subpergunta']}\n"
            f"  sub-answer: {h['subresposta']}"
            for h in historico
        )
    else:
        anteriores = "(none yet)"
    return (
        f"Plan overview: {plano.overview}\n"
        f"Key entities: {'; '.join(plano.entities)}\n"
        f"Relations: {'; '.join(plano.relations)}\n"
        f"Previous steps in this chain:\n{anteriores}"
    )


# Entradas = contexto acumulado da cadeia + texto guia do passo atual
# Saida = state["subpergunta_atual"]
def no_gerar_subpergunta(state: CadeiaState) -> dict:
    """Gera a subpergunta do passo atual a partir do contexto acumulado da cadeia.

    O prompt recebe a pergunta original, o contexto acumulado (plano, entidades,
    relacionamentos e subperguntas/subrespostas anteriores) e o texto guia do passo.
    Numa nova tentativa do mesmo passo (apos um SEM_RESPOSTA), o prompt lista tambem as
    subperguntas que ja falharam e a chamada usa `SUBQ_RETRY_TEMPERATURE`.

    Args:
        state: Estado da cadeia.

    Returns:
        Atualizacao com `subpergunta_atual` (usa o proprio texto guia se o LLM devolver
        uma resposta vazia) e `llm_subpergunta` (registro da chamada, p/ o log).
    """
    passo = state["cadeia"].steps[state["passo_atual"]]
    falhas = state.get("tentativas_falhas", [])
    anteriores = ""
    if falhas:
        lista = "\n".join(f"- {t['subpergunta']}" for t in falhas)
        anteriores = (
            f"These sub-questions for this step were already tried and the retrieved "
            f"passages did not answer them:\n{lista}\n"
            f"Write a DIFFERENT sub-question for the same step: rephrase it, or approach "
            f"it from another angle.\n\n"
        )
    user = (
        f"Original question: {state['questao']}\n\n"
        f"{_contexto_acumulado(state['plano'], state['historico'])}\n\n"
        f"Current step (step {passo.id}) guide text: {passo.guide_text}\n\n"
        f"{anteriores}"
        f"Generate the sub-question."
    )
    temperatura = SUBQ_RETRY_TEMPERATURE if state.get("tentativa_atual", 0) > 0 else 0
    subpergunta, chamada = _chamar_llm(
        SUBQ_PROMPT, user, max_tokens=SUBQ_MAX_TOKENS, temperature=temperatura
    )
    # LLM devolveu vazio: busca pelo proprio texto guia em vez de uma query vazia.
    return {"subpergunta_atual": subpergunta or passo.guide_text, "llm_subpergunta": chamada}


# Entrada = state["subpergunta_atual"] ; Saida = state["chunks_atual"] (top-k do mini-corpus)
def no_retrieve(state: CadeiaState) -> dict:
    """Recupera os `TOP_K` chunks do mini-corpus mais similares a subpergunta atual.

    Args:
        state: Estado da cadeia, com `subpergunta_atual` preenchida.

    Returns:
        Atualizacao com `chunks_atual` (`doc_id`, `score`, `title`, `contents`).
    """
    return {"chunks_atual": _retrieve(state["subpergunta_atual"], TOP_K)}


# Entradas = subpergunta atual + top-k chunks
# Saida = um registro novo em state["historico"] e o avanco de state["passo_atual"]
def no_gerar_subresposta(state: CadeiaState) -> dict:
    """Responde a subpergunta atual usando apenas os chunks recuperados e avanca o passo.

    Args:
        state: Estado da cadeia, com `subpergunta_atual` e `chunks_atual` preenchidos.

    Se a subresposta vier SEM_RESPOSTA e ainda houver tentativas (`SUBQ_RETRY_MAX`),
    nao conclui o passo: guarda a tentativa em `tentativas_falhas` e o roteador volta a
    `gerar_subpergunta` para o MESMO passo, que gera uma subpergunta nova.

    Returns:
        Numa nova tentativa: `tentativa_atual` + 1 e a tentativa acrescentada a
        `tentativas_falhas`. Ao concluir o passo: um novo registro em `historico` (com
        os chunks em `chunks_detalhe`, as chamadas ao LLM em `llm` e as tentativas que
        falharam antes em `tentativas_falhas`), `passo_atual` + 1 e o contador zerado.
    """
    passo = state["cadeia"].steps[state["passo_atual"]]
    passages = "\n\n".join(
        f"[{i}] {c['title']}: {c['contents'][:CHUNK_CHARS]}"
        for i, c in enumerate(state["chunks_atual"], start=1)
    )
    user = (
        f"Passages:\n{passages}\n\n"
        f"Sub-question: {state['subpergunta_atual']}"
    )
    subresposta, chamada = _chamar_llm(SUBA_PROMPT, user, max_tokens=SUBA_MAX_TOKENS)
    subresposta = subresposta or SEM_RESPOSTA
    registro = {
        "id": passo.id,
        "guide_text": passo.guide_text,
        "subpergunta": state["subpergunta_atual"],
        "chunks": [c["title"] for c in state["chunks_atual"]],
        "docs": [c["idx"] for c in state["chunks_atual"]],  # p/ selecionar os chunks da resposta final
        "chunks_detalhe": [
            {"idx": c["idx"], "doc_id": c["doc_id"], "score": round(c["score"], 4), "title": c["title"]}
            for c in state["chunks_atual"]
        ],
        "subresposta": subresposta,
        "llm": {"subpergunta": state.get("llm_subpergunta"), "subresposta": chamada},
    }
    tentativa = state.get("tentativa_atual", 0)
    falhas = state.get("tentativas_falhas", [])
    nova_tentativa = _sem_resposta(subresposta) and tentativa < SUBQ_RETRY_MAX
    _print_passo(state["cadeia"].id, passo.id, len(state["cadeia"].steps), tentativa,
                 state["subpergunta_atual"], subresposta, nova_tentativa)

    if nova_tentativa:
        return {
            "tentativa_atual": tentativa + 1,
            "tentativas_falhas": [*falhas, {"tentativa": tentativa + 1, **registro}],
        }
    registro.update(n_tentativas=tentativa + 1, tentativas_falhas=falhas)
    return {
        "historico": [registro],
        "passo_atual": state["passo_atual"] + 1,
        "tentativa_atual": 0,
        "tentativas_falhas": [],
    }


def _sem_resposta(texto: str) -> bool:
    return texto.strip().lower() == SEM_RESPOSTA.lower()


# As cadeias rodam em threads paralelas: o lock evita que as linhas de um passo se
# misturem com as de outro.
_PRINT_LOCK = threading.Lock()


def _print_passo(
    cadeia_id: int, passo_id: int, n_passos: int, tentativa: int,
    subpergunta: str, subresposta: str, nova_tentativa: bool,
) -> None:
    """Mostra no terminal a subpergunta do passo e a resposta encontrada."""
    prefixo = f"   [chain {cadeia_id} | step {passo_id}/{n_passos}"
    if tentativa:
        prefixo += f" | retry {tentativa}/{SUBQ_RETRY_MAX}"
    prefixo += "]"
    sufixo = "  -> trying a new sub-question" if nova_tentativa else ""
    with _PRINT_LOCK:
        print(
            f"{prefixo} sub-question: {subpergunta}\n"
            f"{' ' * len(prefixo)} sub-answer:   {subresposta}{sufixo}",
            flush=True,
        )


# Uma nova tentativa nao avanca passo_atual, entao tambem volta pra gerar_subpergunta.
def _proximo_passo(state: CadeiaState) -> str:
    return "gerar_subpergunta" if state["passo_atual"] < len(state["cadeia"].steps) else END


def construir_grafo_cadeia() -> CompiledStateGraph:
    """Monta o subgrafo de uma cadeia: subpergunta -> retrieve -> subresposta, L vezes.

    Returns:
        Subgrafo compilado sobre `CadeiaState`, que repete o ciclo ate o ultimo passo.
    """
    g = StateGraph(CadeiaState)
    g.add_node("gerar_subpergunta", no_gerar_subpergunta, retry_policy=RETRY_LLM)
    g.add_node("retrieve", no_retrieve)
    g.add_node("gerar_subresposta", no_gerar_subresposta, retry_policy=RETRY_LLM)
    g.add_edge(START, "gerar_subpergunta")
    g.add_edge("gerar_subpergunta", "retrieve")
    g.add_edge("retrieve", "gerar_subresposta")
    g.add_conditional_edges("gerar_subresposta", _proximo_passo, ["gerar_subpergunta", END])
    return g.compile()


_GRAFO_CADEIA = construir_grafo_cadeia()


# ---------------------------------------------------------------------------
# Grafo de nivel superior: plano -> N cadeias em paralelo (Send)
# ---------------------------------------------------------------------------

class PlanoState(TypedDict):
    """Estado do grafo de nivel superior: plano, execucao das cadeias e resposta final.

    Parameters:
        questao: Pergunta original.
        N: Numero de cadeias pedido ao plano.
        L: Numero de passos por cadeia.
        plano_bruto: JSON cru devolvido pelo LLM na geracao do plano.
        plano: Plano validado.
        plano_tentativas: Uma chamada ao LLM por tentativa de gerar o plano (ver
            `_chamar_llm`), com `tentativa` e `erro` (None na tentativa aceita).
        cadeias_exec: Resultado de cada cadeia (`id`, `description`, `historico`),
            acumulado pelo reducer `operator.add` a medida que as cadeias terminam.
        docs_finais: Chunks que entraram no prompt da resposta final (`doc_id`, `title`,
            `score`), em ordem de relevancia.
        llm_final: Chamada ao LLM que gerou a resposta final.
        resposta_final: Resposta a pergunta original.
    """

    questao: str
    N: int
    L: int
    plano_bruto: str                        # JSON cru devolvido pelo LLM
    plano: Plano | None                     # plano validado
    plano_tentativas: list[dict]            # chamadas ao LLM de cada tentativa de plano
    cadeias_exec: Annotated[list[dict], operator.add]  # {id, description, historico} por cadeia
    docs_finais: list[dict]                 # chunks do prompt final {doc_id, title, score}
    llm_final: dict                         # chamada ao LLM da resposta final
    resposta_final: str


# Entradas = PLAN_PROMPT (com N e L) + state["questao"]
# Saida = state["plano"]: Plano validado com N cadeias x L passos (+ o JSON cru).
# Regenera (ate PLAN_JSON_RETRIES vezes) se o JSON nao validar contra Plano ou fugir de N x L.
def no_gerar_plano(state: PlanoState) -> dict:
    """Pede ao LLM o plano de N cadeias x L passos e o valida.

    Regenera o plano (ate `PLAN_JSON_RETRIES` vezes) quando o JSON nao valida contra
    `Plano` ou nao tem exatamente N cadeias de L passos.

    Args:
        state: Estado com `questao`, `N` e `L`.

    Returns:
        Atualizacao com `plano` (validado), `plano_bruto` (JSON cru) e
        `plano_tentativas` (as chamadas ao LLM de todas as tentativas).

    Raises:
        ValueError: Quando todas as tentativas produzem um plano invalido. As tentativas
            vao no atributo `plano_tentativas` da excecao, para o log.
    """
    system = PLAN_PROMPT.format(N=state["N"], L=state["L"])
    tentativas: list[dict] = []
    for tentativa in range(1, PLAN_JSON_RETRIES + 2):  # 1 tentativa normal + extras
        texto, chamada = _chamar_llm(
            system, state["questao"], max_tokens=PLAN_MAX_TOKENS, json_mode=True
        )
        try:
            plano = Plano.model_validate_json(_extrair_json(texto))
            erro = _checar_formato(plano, state["N"], state["L"])
        except (ValidationError, ValueError) as e:
            erro = f"{type(e).__name__}: {e}"
        tentativas.append({"tentativa": tentativa, "erro": erro, **chamada})
        if erro is None:
            return {"plano_bruto": texto, "plano": plano, "plano_tentativas": tentativas}
        print(f"[gerar_plano] invalid plan on attempt {tentativa}: {erro}", file=sys.stderr)

    print("[gerar_plano] raw JSON of the last attempt:\n" + texto, file=sys.stderr)
    falha = ValueError(f"plano invalido apos {PLAN_JSON_RETRIES + 1} tentativas: {erro}")
    falha.plano_tentativas = tentativas  # lido por rodar_mini_dataset ao gravar a falha
    raise falha


# Uma Send por cadeia: cada uma roda isolada, so com a pergunta, o plano e a propria cadeia.
def _distribuir_cadeias(state: PlanoState) -> list[Send]:
    return [
        Send("executar_cadeia", {"questao": state["questao"], "plano": state["plano"], "cadeia": c})
        for c in state["plano"].chains
    ]


# Entrada = payload da Send (questao, plano, cadeia)
# Saida = state["cadeias_exec"]: o historico completo desta cadeia (acumulado pelo reducer)
def no_executar_cadeia(entrada: dict) -> dict:
    """Executa uma cadeia do plano no seu proprio subgrafo, isolada das demais.

    Args:
        entrada: Payload da `Send`, com `questao`, `plano` e `cadeia`.

    Returns:
        Atualizacao com `cadeias_exec` contendo um unico item: `id`, `description` e o
        `historico` completo da cadeia.
    """
    cadeia: Cadeia = entrada["cadeia"]
    estado = _GRAFO_CADEIA.invoke(
        {**entrada, "passo_atual": 0, "historico": [], "tentativa_atual": 0, "tentativas_falhas": []},
        # 3 nos por tentativa, ate (1 + SUBQ_RETRY_MAX) tentativas por passo, + folga
        config={"recursion_limit": 3 * len(cadeia.steps) * (1 + SUBQ_RETRY_MAX) + 5},
    )
    return {
        "cadeias_exec": [
            {"id": cadeia.id, "description": cadeia.description, "historico": estado["historico"]}
        ]
    }


def _formatar_cadeias(cadeias_exec: list[dict]) -> str:
    """Cadeias ordenadas por id, so com os passos que acharam resposta (em ingles, vai
    pro LLM). Passos com SEM_RESPOSTA sao descartados: so trazem ruido."""
    blocos = []
    for ce in sorted(cadeias_exec, key=lambda c: c["id"]):
        passos = [
            f"  Step {h['id']}:\n"
            f"    sub-question: {h['subpergunta']}\n"
            f"    sub-answer: {h['subresposta']}"
            for h in ce["historico"]
            if not _sem_resposta(h["subresposta"])
        ]
        blocos.append(f"Chain {ce['id']}:\n" + ("\n".join(passos) or "  (no step found an answer)"))
    return "\n\n".join(blocos)


def _selecionar_chunks_finais(questao: str, cadeias_exec: list[dict], m: int) -> list[dict]:
    """Os `m` chunks mais relevantes para a pergunta original, dentre os recuperados em
    todos os passos de todas as cadeias (inclusive os que deram SEM_RESPOSTA e as
    tentativas com subperguntas que falharam).

    Relevancia = produto interno E5 da pergunta original com o chunk (os embeddings ja
    estao em memoria); empate desfeito por quantas vezes o chunk foi recuperado."""
    import torch

    freq: dict[int, int] = {}
    for ce in cadeias_exec:
        for h in ce["historico"]:
            for t in [h, *h.get("tentativas_falhas", [])]:
                for idx in t.get("docs", []):
                    freq[idx] = freq.get(idx, 0) + 1
    if m <= 0 or not freq:
        return []

    _carregar_mini()
    corpus, emb = _MINI["corpus"], _MINI["emb"]
    idxs = list(freq)
    scores = (_MINI["encode"]([questao]) @ emb[torch.tensor(idxs)].t()).squeeze(0).tolist()
    ordem = sorted(zip(idxs, scores), key=lambda t: (-t[1], -freq[t[0]]))[:m]
    selecionados = []
    for idx, score in ordem:
        doc = corpus[idx]
        selecionados.append(
            {
                "doc_id": int(doc.get("orig_doc_id", idx)),
                "score": float(score),
                "title": doc.get("title", ""),
                "contents": doc.get("contents", "")[:CHUNK_CHARS],
            }
        )
    return selecionados


# Entradas = state["questao"] + historico de TODAS as cadeias (state["cadeias_exec"])
#            + os FINAL_DOCS chunks mais relevantes dentre os recuperados pelas cadeias
# Saida = state["resposta_final"] e state["docs_finais"]
def no_gerar_resposta_final(state: PlanoState) -> dict:
    """Produz a resposta final a partir das cadeias e dos chunks mais relevantes.

    O prompt recebe os `FINAL_DOCS` chunks mais similares a pergunta original (dentre os
    recuperados pelas cadeias) e as subperguntas/subrespostas de todas as cadeias.

    Args:
        state: Estado com `questao` e `cadeias_exec` de todas as cadeias.

    Returns:
        Atualizacao com `resposta_final` (`SEM_RESPOSTA` quando o LLM devolve vazio),
        `docs_finais` (`doc_id`, `title`, `score` dos chunks usados) e `llm_final`
        (registro da chamada ao LLM).
    """
    chunks = _selecionar_chunks_finais(state["questao"], state["cadeias_exec"], FINAL_DOCS)
    passages = "\n\n".join(
        f"[{i}] {c['title']}: {c['contents']}" for i, c in enumerate(chunks, start=1)
    ) or "(none)"
    user = (
        f"Original question: {state['questao']}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        f"Reasoning chains:\n{_formatar_cadeias(state['cadeias_exec'])}\n\n"
        f"Answer the original question."
    )
    resposta, chamada = _chamar_llm(FINAL_PROMPT, user, max_tokens=FINAL_MAX_TOKENS)
    return {
        "resposta_final": resposta or SEM_RESPOSTA,
        "llm_final": chamada,
        "docs_finais": [
            {"doc_id": c["doc_id"], "title": c["title"], "score": round(c["score"], 4)}
            for c in chunks
        ],
    }


def construir_grafo() -> CompiledStateGraph:
    """Monta o grafo completo: plano -> N cadeias em paralelo -> resposta final.

    Returns:
        Grafo compilado sobre `PlanoState`; invocar com `questao`, `N` e `L`.
    """
    g = StateGraph(PlanoState)
    g.add_node("gerar_plano", no_gerar_plano, retry_policy=RETRY_LLM)
    g.add_node("executar_cadeia", no_executar_cadeia)
    g.add_node("gerar_resposta_final", no_gerar_resposta_final, retry_policy=RETRY_LLM)
    g.add_edge(START, "gerar_plano")
    g.add_conditional_edges("gerar_plano", _distribuir_cadeias, ["executar_cadeia"])
    # so dispara depois que todas as Sends (cadeias) terminam
    g.add_edge("executar_cadeia", "gerar_resposta_final")
    g.add_edge("gerar_resposta_final", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Log, metricas e execucao sobre o mini dataset
# ---------------------------------------------------------------------------

def _metricas(answers: list[list[str]], preds: list[str]) -> dict:
    """EM/F1 oficiais do repo (src/inference/metrics.py, normalizacao SQuAD)."""
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from inference.metrics import compute_metrics_dict

    # o logger_config do repo liga o logging raiz em INFO -> o httpx loga cada request ao LLM
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return compute_metrics_dict(labels=answers, preds=preds, eval_metrics="em_and_f1")


def rodar_mini_dataset(task: str, N: int, L: int, limit: int | None = None) -> dict:
    """Roda o grafo sobre as perguntas de data/mini/questions/<task> (30 por tarefa).

    Grava em RUNS_DIR:
      - <run_id>.jsonl         -- log: 1 linha por pergunta (predicao, em/f1, plano,
                                  cadeias, docs_finais) com TODAS as chamadas ao LLM
                                  (prompts system/user, resposta, raciocinio, tokens):
                                  plano_tentativas, historico[*].llm e llm_final. Flush
                                  a cada pergunta -- se o processo cair, as ja feitas
                                  ficam salvas. Uma excecao numa pergunta nao derruba
                                  as outras.
      - <run_id>_metrics.json  -- EM/F1 agregados + config do run.
      - index.jsonl            -- 1 linha por run (append-only), pra comparar runs.

    Args:
        task: Tarefa do mini dataset (`hotpotqa`, `2wikimultihopqa`, `musique` ou
            `bamboogle`).
        N: Numero de cadeias por plano.
        L: Numero de passos por cadeia.
        limit: Roda so as primeiras `limit` perguntas. Defaults to None (todas).

    Returns:
        Dict de metricas do run: config (`task`, `N`, `L`, `top_k`, `final_docs`, `model`), `em`,
        `f1`, `n_perguntas`, `n_falhas`, `tempo_total_s` e o caminho do log.
    """
    from datasets import load_from_disk

    perguntas = load_from_disk(str(MINI_BASE_DIR / "questions" / task))
    if limit is not None:
        perguntas = perguntas.select(range(min(limit, len(perguntas))))
    total = len(perguntas)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{task}_N{N}_L{L}_k{TOP_K}_d{FINAL_DOCS}_r{SUBQ_RETRY_MAX}_{timestamp}"
    log_path = RUNS_DIR / f"{run_id}.jsonl"

    print(
        f"=== [{task}] {total} questions ({LLM_MODEL}, N={N} x L={L}, top-k={TOP_K}, "
        f"final docs={FINAL_DOCS}, sub-question retries={SUBQ_RETRY_MAX} "
        f"@ T={SUBQ_RETRY_TEMPERATURE}) ==="
    )
    grafo = construir_grafo()
    answers, preds, n_falhas = [], [], 0
    t0 = time.time()

    with open(log_path, "w") as f:
        for i, ex in enumerate(perguntas, start=1):
            t_q = time.time()
            print(f"\n[{i}/{total}] {ex['query']}", flush=True)
            registro = {"query_id": ex["query_id"], "query": ex["query"], "answers": ex["answers"]}
            try:
                estado = grafo.invoke({"questao": ex["query"], "N": N, "L": L})
                registro.update(
                    prediction=estado["resposta_final"],
                    plano=estado["plano"].model_dump(),
                    plano_tentativas=estado["plano_tentativas"],
                    cadeias_exec=sorted(estado["cadeias_exec"], key=lambda c: c["id"]),
                    docs_finais=estado["docs_finais"],
                    llm_final=estado["llm_final"],
                    erro=None,
                )
            except Exception as e:  # noqa: BLE001 -- uma pergunta ruim nao derruba as outras
                n_falhas += 1
                print(f"   [ERROR] {type(e).__name__}: {e}", file=sys.stderr)
                registro.update(
                    prediction=SEM_RESPOSTA, plano=None,
                    plano_tentativas=getattr(e, "plano_tentativas", []),
                    cadeias_exec=[], docs_finais=[], llm_final=None,
                    erro=f"{type(e).__name__}: {e}",
                )

            registro.update(_metricas([ex["answers"]], [registro["prediction"]]))
            registro["tempo_s"] = round(time.time() - t_q, 1)
            answers.append(ex["answers"])
            preds.append(registro["prediction"])
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"   => final answer: {registro['prediction']}  (gold: {ex['answers']}, "
                f"em={registro['em']}, f1={registro['f1']}, {registro['tempo_s']}s)",
                flush=True,
            )

    metricas = {
        "run_id": run_id,
        "timestamp": timestamp,
        "task": task,
        "N": N,
        "L": L,
        "top_k": TOP_K,
        "final_docs": FINAL_DOCS,
        "subq_retry_max": SUBQ_RETRY_MAX,
        "subq_retry_temperature": SUBQ_RETRY_TEMPERATURE,
        "model": LLM_MODEL,
        "n_perguntas": total,
        "n_falhas": n_falhas,
        **_metricas(answers, preds),
        "tempo_total_s": round(time.time() - t0, 1),
        "arquivo": str(log_path.relative_to(REPO_ROOT)),
    }
    (RUNS_DIR / f"{run_id}_metrics.json").write_text(json.dumps(metricas, indent=2))
    with open(RUNS_DIR / "index.jsonl", "a") as f:
        f.write(json.dumps(metricas, ensure_ascii=False) + "\n")

    print(
        f"--- [{task}] EM={metricas['em']}  F1={metricas['f1']}  "
        f"failures={n_falhas}/{total}  ({metricas['tempo_total_s']}s)\n"
        f"    log: {log_path}"
    )
    return metricas


# Uso: python src/SPMC/SPMC_no_placeholder/spmc_no_placeholder.py [--task hotpotqa|...|all] [--limit 2] [--N 4] [--L 6]
#   SUBQ_RETRY_MAX = subperguntas novas por passo apos SEM_RESPOSTA (0 desliga) ; SUBQ_RETRY_TEMPERATURE = temperatura delas
#   SPMC_TOP_K = chunks por passo ; SPMC_FINAL_DOCS = chunks no prompt final (0 desliga) ; SPMC_MODEL troca o modelo do Ollama (default gpt-oss:120b-cloud)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SPMC sem placeholders sobre o mini dataset")
    ap.add_argument("--task", choices=[*TASKS, "all"], default="hotpotqa")
    ap.add_argument("--limit", type=int, default=None, help="roda so as primeiras N perguntas")
    ap.add_argument("--N", type=int, default=int(os.getenv("SPMC_N", N_DEFAULT)))
    ap.add_argument("--L", type=int, default=int(os.getenv("SPMC_L", L_DEFAULT)))
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else (args.task,)
    resumo = [rodar_mini_dataset(t, args.N, args.L, args.limit) for t in tasks]

    if len(resumo) > 1:
        print("\n=== Summary ===")
        for m in resumo:
            print(f"{m['task']:<16} EM={m['em']:<6} F1={m['f1']:<6} failures={m['n_falhas']}")
