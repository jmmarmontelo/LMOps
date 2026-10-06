
import operator
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]

# Carrega o .env de src/reproducao/ (onde fica a GEMINI_API_KEY).
load_dotenv(REPO_ROOT / "src" / "reproducao" / ".env")

# Gemini via endpoint OpenAI-compativel -- reusa a lib openai, sem dependencia nova.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# gemini-2.5-flash ja nao esta disponivel pra chaves novas; gemini-flash-latest e o alias flash estavel.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# CoRAG (modelo fine-tuned corag/CoRAG-Llama3.1-8B-MultihopQA, servido via endpoint
# OpenAI-compat -- mesmo padrao de src/reproducao/dynamic_chain.py e best_of_n.py).
# So gerar_plano usa a Gemini; todos os outros nos usam este modelo (ver _chamar_corag).
# BASE_URL/API_KEY/MODEL_NAME ja vem do mesmo .env carregado acima.
CORAG_BASE_URL = os.getenv("BASE_URL")
CORAG_API_KEY = os.getenv("API_KEY")
CORAG_MODEL = os.getenv("MODEL_NAME", "corag-8b")
L_DEFAULT = 6  # passos por cadeia (fixo; toda cadeia tem exatamente L passos). Mesmo L do repo (max_path_length = 6)
N_DEFAULT = 4  # cadeias de raciocinio por plano (best-of-N; 4 para teste)
# Teto de tokens da resposta. vLLM assume 16 se omitido -> truncaria o plano.
# Ajustavel por env sem mexer no codigo (ex.: PLAN_MAX_TOKENS=16384).
PLAN_MAX_TOKENS = int(os.getenv("PLAN_MAX_TOKENS", "8192"))
SUBQ_MAX_TOKENS = int(os.getenv("SUBQ_MAX_TOKENS", "256"))  # teto p/ a geracao de sub-pergunta
EVIDENCE_MAX_TOKENS = int(os.getenv("EVIDENCE_MAX_TOKENS", "4000"))  # teto p/ o evidence extractor
EVIDENCE_CHUNK_CHARS = int(os.getenv("EVIDENCE_CHUNK_CHARS", "600"))  # trunca cada chunk no prompt
SUBA_MAX_TOKENS = int(os.getenv("SUBA_MAX_TOKENS", "128"))  # teto p/ a geracao de sub-resposta
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS", "128"))  # teto p/ a resposta final
# Tentativas extras de subpergunta por passo quando a subresposta vem SEM_RESPOSTA
# (motivo "sem_evidencia" ou "sem_resposta_llm") -- gera uma subpergunta diferente e
# repete retrieve/evidence/subresposta pro MESMO passo antes de desistir.
SUBQ_RETRY_MAX = int(os.getenv("SUBQ_RETRY_MAX", "1"))
# Retentativas para erros transitorios (503 "high demand", 429, 5xx, conexao).
PLAN_MAX_RETRIES = int(os.getenv("PLAN_MAX_RETRIES", "5"))
# Tentativas extras de gerar o plano de novo quando o JSON sai invalido/nao valida
# contra Plano (LLM as vezes erra o shape mesmo com response_format=json_object).
PLAN_JSON_RETRIES = int(os.getenv("PLAN_JSON_RETRIES", "2"))
# As N cadeias sao independentes ate a agregacao final -- rodam em paralelo, uma thread
# por cadeia (ate este teto), em vez de sequencialmente. Cada thread faz suas proprias
# chamadas Gemini/retrieve, entao um valor alto aumenta a chance de esbarrar no rate
# limit do tier gratuito (ver no_executar_cadeias) -- ajustavel por env sem mexer no codigo.
SPMC_CHAIN_THREADS = int(os.getenv("SPMC_CHAIN_THREADS", "4"))
_STATUS_RETENTAVEIS = {408, 409, 429, 500, 502, 503, 504, 529}
# Sub-resposta que sinaliza "os documentos nao respondem a subpergunta" -- mesma string
# usada nos scripts de src/reproducao/ (get_generate_intermediate_answer_prompt).
SEM_RESPOSTA = "no relevant information found"

# --- Retrieval -------------------------------------------------------------
# SPMC_RETRIEVER: "mini" (default, denso in-process sobre data/mini/) | "e5_server" (reserva).
SPMC_RETRIEVER = os.getenv("SPMC_RETRIEVER", "mini")
SPMC_TOP_K = int(os.getenv("SPMC_TOP_K", "5"))
MINI_BASE_DIR = Path(os.getenv("MINI_BASE_DIR", str(REPO_ROOT / "data" / "mini")))
E5_MODEL_NAME_OR_PATH = os.getenv("E5_MODEL_NAME_OR_PATH", "intfloat/e5-large-v2")
E5_SERVER_HOST = os.getenv("E5_SERVER_HOST", "localhost")
E5_SERVER_PORT = int(os.getenv("E5_SERVER_PORT", "8090"))

INSTRUCOES_PLANO = """\
Instruction: You are a reasoning assistant specialized in multi-step question
answering tasks (a Plan Generator for a multi-hop RAG system, SPMC-RAG). Given a
complex multi-hop query, your goal is to break it down into {N} alternative
reasoning chains and, for each chain, derive a reasoning template for every step --
producing a structured reasoning plan to answer the query. The plan does NOT contain
the sub-questions or the answers themselves -- those are generated later, during
execution; here you only produce the roadmap.

Follow these steps:
1. Identify the key entities and relationships in the query.
2. Decompose the query into {N} independent, logically ordered sequences of
   reasoning steps (chains). Each chain is a route that may differ from the others
   (a different step order, different bridge entities, a different decomposition of
   the query), but EVERY chain must end up reaching the answer to the original query.
3. For each step, define a reasoning template in the format
   <[subject]; [relationship]; [object]> -- expressed in the JSON below as a "triple"
   object with "subject"/"predicate"/"object" keys -- using placeholders in the
   format [S_k: type: ??] at the positions whose value will only be known after
   executing step k.

Respond ONLY with a valid JSON object (no markdown, no ``` fences, no comments),
in exactly this format:

{{
  "overview": "what the query requires discovering, in one or two sentences",
  "entities": ["key entity", "..."],
  "relations": ["relation involved", "..."],
  "chains": [
    {{
      "id": 1,
      "description": "how this route differs from the others",
      "steps": [
        {{
          "k": 1,
          "guide_text": "what this step must discover; refer in natural language to the previous steps of THIS SAME chain it depends on",
          "triple": {{"subject": "...", "predicate": "...", "object": "..."}},
          "define": "S_1",
          "depends_on": []
        }}
      ]
    }}
  ]
}}

Rules:
- Exactly {N} chains, with "id" from 1 to {N}.
- Each chain has EXACTLY {L} steps, with "k" from 1 to {L}.
- Every object in "steps" MUST contain the "triple" key (exactly that name -- never
  "triple_template" nor loose fields on the step), and "triple" has all three
  sub-fields filled in: "subject", "predicate" and "object" (none empty).
- "define": the placeholder this step resolves (e.g. "S_2"), or null.
- "depends_on": list of placeholders from previous steps used in this step (e.g. ["S_1"]).
- If "define" is not null (e.g. "S_2"), the step's own "triple" MUST contain the
  placeholder [S_2: type: ??] literally, in whichever field is still unknown -- even
  if you already know the real-world answer, NEVER write it there. The value is
  discovered later, during execution, by retrieval -- not by you. Pre-filling it
  breaks the chain: a later step with "depends_on": ["S_2"] will never receive the
  value, because the executor only fills placeholders it can find in the text.
- Step k = {L} (the last one) produces the final answer to the query.
- The {N} chains are independent and differ in APPROACH. If the query seems to need
  fewer hops, use the remaining steps to confirm, validate, refine, or disambiguate
  entities already found -- never fewer than {L} steps per chain.
- Do not invent answers, and do not pre-fill a step's own "define"d placeholder with
  a value you already know -- the plan is only the skeleton, not the answer key.
"""

INSTRUCOES_SUBPERGUNTA = """\
You generate ONE natural-language sub-question for a step of a multi-hop reasoning
chain. It will be used to retrieve documents and answer that hop.

Rules:
- Return a single, direct, self-contained question, in the same language as the
  original question.
- Base it on the step's guide text and the triple template (already partially
  resolved).
- Use the facts from the CONTEXT (previous steps in this chain) to make the question
  specific: replace vague references with the values already discovered.
- Whatever is left as [S_k: type: ??] in the triple is exactly what the question
  should ask for.
- If a "target of THIS step" is given, the question MUST ask for EXACTLY that value
  and nothing else -- do not combine it with information that belongs to a LATER
  step, even if the chain description or later steps hint at it. One hop, one fact.
- Do not answer the question, do not explain -- respond with only the question.
"""

# SPMCv1.md Secao 3.4 -- Evidence Extractor: CoT que troca K chunks ruidosos por poucas
# triplas de alta precisao.
INSTRUCOES_EVIDENCE = """\
Instruction: You are the Evidence Extractor of a multi-hop RAG system. Extract
triples in the form <subject; predicate; object> from the provided passages that
directly support answering the sub-question. Follow these steps:

1. Identify Key Entities and Relationships
   Determine the primary entities and relationships in the sub-question.
2. Locate Relevant Sentences
   Find sentences in the passages that mention these entities or related terms.
3. Extract Triples
   Extract <subject; predicate; object> triples from these sentences, focusing on
   those that reflect relationships relevant to the sub-question -- including
   implicit relations that only emerge by combining information across different
   passages.
4. Clarify and Refine
   Ensure each triple has clear entities and resolve any pronouns or synonyms for
   accuracy. Discard unrelated triples.
5. Final Output
   Present only the refined triples that are directly relevant to the sub-question.
   If no passage contains relevant information, return an empty list -- do not
   invent facts that are not in the passages. Respond only with the final result,
   do not show your reasoning.

Respond ONLY with a valid JSON object (no markdown, no comments), in exactly this
format: {"evidence": [{"subject": "...", "predicate": "...", "object": "..."}]}
"""

# SPMCv1.md Secao 2, passo 5 -- gera a sub-resposta so a partir das triplas de
# evidencia (nao dos chunks brutos): instrucoes + sub-pergunta + triplas de evidencia.
INSTRUCOES_SUBRESPOSTA = """\
Instruction: You are answering ONE sub-question of a multi-hop reasoning chain,
using ONLY the evidence triples provided below (already extracted from retrieved
passages) -- do not use outside knowledge or invent facts.

Rules:
- Answer the sub-question directly and concisely: a short phrase or entity name,
  not a full sentence, no explanation.
- Base the answer strictly on the evidence triples given.
- If the evidence triples do not contain enough information to answer the
  sub-question, respond with exactly: {sem_resposta}
- Respond with only the answer text, nothing else.
"""

# Ultimo no do grafo -- agrega os hops respondidos de TODAS as cadeias (state["contexto"],
# ja montado por no_montar_contexto) e produz a resposta final a pergunta original.
INSTRUCOES_RESPOSTA_FINAL = """\
Instruction: You are producing the FINAL answer to the original multi-hop question of
a Chain-of-Retrieval RAG system. Below you are given several independent REASONING
CHAINS that each tried to reach the answer via a different route -- for every chain,
only the sub-questions that were actually answered, their sub-answers, and the
resolved triple for each step (unanswered steps were already filtered out).

Rules:
- Use the reasoning chains as your only source of evidence -- do not use outside
  knowledge, and do not invent facts beyond what they establish.
- The chains may be incomplete, redundant, or phrase the same fact differently --
  cross-check them and rely on whichever chain(s) actually reached a conclusive
  answer; you do not need every chain to agree.
- Answer the ORIGINAL question directly and concisely: a short phrase or entity name,
  not a full sentence, no explanation, no restating the question.
- If none of the chains provide enough information to answer the original question,
  respond with exactly: {sem_resposta}
- Respond with only the answer text, nothing else.
"""


# ---------------------------------------------------------------------------
# Schema do plano estruturado (Pydantic) -- ver src/SPMC/SPMCv1.md
# ---------------------------------------------------------------------------


def _achar_por_apelido(d: dict, *nomes: str) -> str | None:
    """Acha em `d` a chave que corresponde a algum de `nomes`, comparando sem
    diferenciar maiuscula/minuscula (o LLM as vezes manda "Subject" em vez de
    "subject", por exemplo -- sem isso, o campo cai no default vazio em silencio, sem
    nenhum erro). Devolve a chave ORIGINAL tal como esta em `d` (pra usar em
    d.pop(chave)), ou None se nenhum dos `nomes` aparecer."""
    mapa = {k.lower(): k for k in d}
    for nome in nomes:
        chave = mapa.get(nome.lower())
        if chave is not None:
            return chave
    return None


class Tripla(BaseModel):
    """Triple template <subject; predicate; object>; fields may contain [S_k: type: ??].

    Fields default to "" to tolerate a step where the LLM omitted one of them --
    no_parsear_plano warns on stderr when that happens.
    """

    subject: str = ""
    predicate: str = ""
    object: str = ""

    @model_validator(mode="before")
    @classmethod
    def _apelidos(cls, data):
        """Aceita sujeito/predicado/objeto (legado PT) e s/p/o como apelidos dos
        campos, sem diferenciar maiuscula/minuscula (ver _achar_por_apelido)."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        for canonico, apelidos in (
            ("subject", ("sujeito", "s")),
            ("predicate", ("predicado", "p")),
            ("object", ("objeto", "o")),
        ):
            if canonico not in d:
                chave = _achar_por_apelido(d, canonico, *apelidos)
                if chave is not None:
                    d[canonico] = d.pop(chave)
        return d


class Passo(BaseModel):
    """A hop of a chain."""

    k: int                                  # posicao do passo, 1..L
    guide_text: str                         # o que este hop deve descobrir
    triple: Tripla = Field(default_factory=Tripla)  # opcional: LLM as vezes omite
    define: str | None = None              # placeholder que este passo resolve (ex. "S_2")
    depends_on: list[str] = Field(default_factory=list)  # placeholders anteriores usados aqui

    @model_validator(mode="before")
    @classmethod
    def _normalizar(cls, data):
        """Conserta as variacoes de shape que o LLM costuma emitir para um passo."""
        if not isinstance(data, dict):
            return data
        d = dict(data)

        # 1) triple sob outro nome/capitalizacao (inclui "tripla", legado PT)
        if not isinstance(d.get("triple"), dict) or not d.get("triple"):
            chave = _achar_por_apelido(
                d, "triple", "tripla", "tripla_template", "triple_template",
                "tripla-template", "triplet", "tpl",
            )
            if chave is not None and isinstance(d[chave], dict):
                d["triple"] = d.pop(chave)

        # 2) subject/predicate/object (ou sujeito/predicado/objeto) soltos no passo,
        # sem diferenciar maiuscula/minuscula
        if not isinstance(d.get("triple"), dict) or not d.get("triple"):
            spo = {}
            for campo, apelidos in (
                ("subject", ("subject", "sujeito")),
                ("predicate", ("predicate", "predicado")),
                ("object", ("object", "objeto")),
            ):
                chave = _achar_por_apelido(d, *apelidos)
                if chave is not None:
                    spo[campo] = d.pop(chave)
            if spo:
                d["triple"] = spo

        # 3) define "" -> None ; depends_on sempre lista (aceita "depende_de" legado PT)
        if d.get("define") in ("", "null", "None"):
            d["define"] = None
        if "depends_on" not in d and "depende_de" in d:
            d["depends_on"] = d.pop("depende_de")
        dd = d.get("depends_on")
        if not isinstance(dd, list):
            d["depends_on"] = [] if dd in (None, "") else [dd]

        return d


class Cadeia(BaseModel):
    """A reasoning route: L steps linked by placeholders."""

    id: int                                 # 1..N
    description: str                        # como esta rota difere das outras
    steps: list[Passo]


class Plano(BaseModel):
    """Global plan: description + entities/relations + N chains."""

    overview: str
    entities: list[str]
    relations: list[str]
    chains: list[Cadeia]


class Evidencias(BaseModel):
    """Evidence Extractor output: evidence triples (concrete facts, no placeholders)."""

    evidence: list[Tripla] = Field(default_factory=list)


def _extrair_json(texto: str) -> str:
    """Recorta o objeto JSON de `texto`, tolerando cercas ```json ... ``` ou texto solto
    em volta -- pega do primeiro '{' ao ultimo '}' -- e remove virgulas sobrando antes de
    '}'/']' (o LLM as vezes deixa, ex. '"a": 1,\\n}', que JSON padrao nao aceita e derruba
    o parser com "key must be a string")."""
    inicio = texto.find("{")
    fim = texto.rfind("}")
    if inicio == -1 or fim == -1 or fim < inicio:
        raise ValueError(f"nenhum objeto JSON encontrado na resposta do plano: {texto[:200]!r}")
    bruto = texto[inicio : fim + 1]
    return re.sub(r",\s*([}\]])", r"\1", bruto)


# Casa [S_1: person: ??], [S_2: cidade: ...], [S3: ...] etc. Grupo 1 = o numero k.
_RE_PLACEHOLDER = re.compile(r"\[\s*S_?(\d+)\s*:[^\]]*\]")


def _resolver_tripla(
    tripla: "Tripla", resolvidos: dict[str, str]
) -> tuple["Tripla", list[str]]:
    """Troca os placeholders [S_k: tipo: ??] pelos valores ja conhecidos em `resolvidos`
    (chaves no formato "S_k"). Placeholders sem valor -- o do proprio passo ou uma
    dependencia ainda nao resolvida -- ficam intactos. Devolve (tripla_resolvida,
    lista ordenada das chaves S_k que continuam pendentes)."""
    pendentes: set[str] = set()

    def _sub(m: "re.Match[str]") -> str:
        chave = f"S_{m.group(1)}"
        if chave in resolvidos:
            return resolvidos[chave]
        pendentes.add(chave)
        return m.group(0)

    campos = {
        campo: _RE_PLACEHOLDER.sub(_sub, getattr(tripla, campo))
        for campo in ("subject", "predicate", "object")
    }
    return Tripla(**campos), sorted(pendentes)


def _tripla_str(t: Tripla) -> str:
    return f"<{t.subject}; {t.predicate}; {t.object}>"


# Casa [S_1: person: ??] capturando o numero (grupo 1) e o rotulo de tipo (grupo 2).
_RE_PLACEHOLDER_TIPO = re.compile(r"\[\s*S_?(\d+)\s*:\s*([^:\]]*?)\s*:\s*\?\?\s*\]")


def _tipo_placeholder(tripla: Tripla, chave: str | None) -> str | None:
    """Acha o rotulo de tipo do placeholder `chave` (ex. "S_2") na tripla ORIGINAL
    (nao resolvida) do passo -- em "[S_2: character: ??]" devolve "character". Usado
    pra travar a subpergunta no alvo certo do passo atual (evita ela "vazar" pra um
    placeholder de um passo futuro). None se `chave` for None ou nao aparecer."""
    if not chave:
        return None
    numero = chave.split("_", 1)[-1]
    for campo in (tripla.subject, tripla.predicate, tripla.object):
        for m in _RE_PLACEHOLDER_TIPO.finditer(campo):
            if m.group(1) == numero:
                return m.group(2).strip() or None
    return None


def _formatar_contexto(historico: list[dict]) -> str:
    """Renderiza o acumulador da cadeia (subpergunta + subresposta + tripla de cada
    passo ja concluido) como texto para o prompt -- em ingles, pois vai pro LLM."""
    if not historico:
        return "(no previous steps in this chain)"
    return "\n".join(
        f"Step {h['k']}:\n"
        f"  sub-question: {h['subpergunta']}\n"
        f"  sub-answer: {h['subresposta']}\n"
        f"  triple: {h['tripla']}"
        for h in historico
    )


def _formatar_chunks(chunks: list[dict]) -> str:
    """Renderiza os Top-K chunks recuperados (titulo + inicio do conteudo, truncado em
    EVIDENCE_CHUNK_CHARS) como texto para o prompt do evidence extractor -- em ingles,
    pois vai pro LLM."""
    if not chunks:
        return "(no chunks retrieved)"
    return "\n\n".join(
        f"[{i}] {c.get('title', '')}\n{(c.get('contents') or '')[:EVIDENCE_CHUNK_CHARS]}"
        for i, c in enumerate(chunks, start=1)
    )


def _formatar_evidencias(evidencias: list[Tripla]) -> str:
    """Renderiza as triplas de evidencia extraidas como texto para o prompt da
    sub-resposta -- em ingles, pois vai pro LLM. Deliberadamente NAO usa os chunks
    brutos (so as triplas ja refinadas pelo evidence extractor)."""
    if not evidencias:
        return "(no evidence extracted)"
    return "\n".join(f"- {_tripla_str(t)}" for t in evidencias)


def _chamar_llm(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int,
    verbose: bool = False,
    json_mode: bool = False,
) -> str:
    """Chamada unica a um LLM servido via endpoint OpenAI-compat (Gemini ou CoRAG --
    ver _chamar_gemini/_chamar_corag) com retry em erros transitorios (503/429/5xx/
    conexao) e backoff exponencial. Devolve o texto da resposta."""
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = None
    ultimo_erro: Exception | None = None
    for tentativa in range(1, PLAN_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except (APIConnectionError, RateLimitError) as e:
            ultimo_erro = e
        except APIStatusError as e:
            if e.status_code not in _STATUS_RETENTAVEIS:
                raise
            ultimo_erro = e

        if tentativa < PLAN_MAX_RETRIES:
            espera = min(2 ** tentativa, 30)
            print(
                f"[{model}] attempt {tentativa}/{PLAN_MAX_RETRIES} failed "
                f"({type(ultimo_erro).__name__}); retrying in {espera}s",
                file=sys.stderr,
            )
            time.sleep(espera)

    if resp is None:
        raise ultimo_erro

    texto = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason
    if verbose or finish_reason == "length":
        u = resp.usage
        aviso = "  TRUNCATED: increase max_tokens" if finish_reason == "length" else ""
        print(
            f"[{resp.model}] finish_reason={finish_reason}  "
            f"completion_tokens={u.completion_tokens}  prompt_tokens={u.prompt_tokens}  "
            f"chars={len(texto or '')}{aviso}",
            file=sys.stderr,
        )
    return texto


def _chamar_gemini(
    system: str,
    user: str,
    *,
    max_tokens: int,
    verbose: bool = False,
    json_mode: bool = False,
) -> str:
    """Gemini -- usado so por gerar_plano."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY nao encontrada. Defina-a em src/reproducao/.env ou no ambiente."
        )
    return _chamar_llm(
        GEMINI_BASE_URL, api_key, GEMINI_MODEL, system, user,
        max_tokens=max_tokens, verbose=verbose, json_mode=json_mode,
    )


def _chamar_corag(
    system: str,
    user: str,
    *,
    max_tokens: int,
    verbose: bool = False,
    json_mode: bool = False,
) -> str:
    """CoRAG (corag/CoRAG-Llama3.1-8B-MultihopQA, servidor remoto via BASE_URL/API_KEY/
    MODEL_NAME em src/reproducao/.env) -- usado por todos os nos exceto gerar_plano."""
    if not CORAG_BASE_URL or not CORAG_API_KEY:
        raise RuntimeError(
            "BASE_URL/API_KEY nao encontrados. Defina-os em src/reproducao/.env ou no ambiente."
        )
    return _chamar_llm(
        CORAG_BASE_URL, CORAG_API_KEY, CORAG_MODEL, system, user,
        max_tokens=max_tokens, verbose=verbose, json_mode=json_mode,
    )


# ---------------------------------------------------------------------------
# Retrieval: mini-corpus denso in-process (padrao) + reserva pro servidor E5
# ---------------------------------------------------------------------------

# Carregados lazy na 1a chamada a _retrieve_mini_corpus e cacheados aqui.
_MINI: dict = {"corpus": None, "emb": None, "encode": None}
# As N cadeias rodam em threads paralelas (ver no_executar_cadeias) e todas chamam
# retrieve -- sem lock, duas threads poderiam disparar o carregamento (~7s: dataset +
# modelo E5) ao mesmo tempo e uma enxergar _MINI parcialmente preenchido pela outra.
_MINI_LOCK = threading.Lock()


def _carregar_encoder_e5():
    """Devolve encode(list[str]) -> Tensor [n, 1024] (float32, L2-normalizado), com a
    MESMA convencao que gerou data/mini/e5-large-index (e5-large-v2: prefixo 'query: ',
    avg-pool mascarado, normalize). torch/transformers importados so aqui."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E5_MODEL_NAME_OR_PATH)
    mod = AutoModel.from_pretrained(E5_MODEL_NAME_OR_PATH, torch_dtype=torch.float32)
    mod.eval()

    @torch.no_grad()
    def encode(textos: list[str]):
        lote = tok(
            [f"query: {t}" for t in textos],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        saida = mod(**lote).last_hidden_state
        mask = lote["attention_mask"].unsqueeze(-1).float()
        media = (saida * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(media, p=2, dim=-1)

    return encode


def _carregar_mini() -> None:
    if _MINI["corpus"] is not None:
        return
    with _MINI_LOCK:
        if _MINI["corpus"] is not None:  # outra thread carregou enquanto esperavamos o lock
            return
        _carregar_mini_sem_lock()


def _carregar_mini_sem_lock() -> None:
    import torch
    from datasets import Dataset

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

    # "corpus" e o campo que _carregar_mini() checa (com e sem lock) pra decidir se ja
    # carregou -- por isso e atribuido por ULTIMO aqui. Se fosse o primeiro (como era
    # antes), uma thread concorrente que passasse pelo checkout RAPIDO (fora do lock, em
    # _carregar_mini) entre esta linha e a de "encode" veria corpus != None e devolveria
    # cedo, achando que jah carregou -- e encode() ainda seria None nesse instante.
    emb = torch.cat(
        [torch.load(str(p), weights_only=True, map_location="cpu") for p in shard_paths]
    ).to(torch.float32)
    encode = _carregar_encoder_e5()
    _MINI["emb"] = emb
    _MINI["encode"] = encode
    _MINI["corpus"] = Dataset.load_from_disk(str(corpus_dir))


def _retrieve_mini_corpus(subpergunta: str, k: int) -> list[dict]:
    """Top-K do mini-corpus por similaridade densa E5 (produto interno sobre os
    embeddings pre-computados). Nao precisa de servidor."""
    import torch

    _carregar_mini()
    corpus, emb, encode = _MINI["corpus"], _MINI["emb"], _MINI["encode"]

    q = encode([subpergunta]).to(emb.dtype)          # [1, 1024]
    scores = (q @ emb.t()).squeeze(0)                # [n_docs]
    top = torch.topk(scores, k=min(k, scores.numel()))

    resultados: list[dict] = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        doc = corpus[int(idx)]
        resultados.append(
            {
                "doc_id": int(doc.get("orig_doc_id", idx)),
                "score": float(score),
                "title": doc.get("title", ""),
                "contents": doc.get("contents", ""),
            }
        )
    return resultados


def _retrieve_e5_server(subpergunta: str, k: int) -> list[dict]:
    """RESERVA -- consulta o servidor E5 (src/search/start_e5_server_main.py) por HTTP.
    Ative com SPMC_RETRIEVER=e5_server. O servidor devolve so {doc_id, score}: title/
    contents ficam "" ate voce ligar o lookup do corpus aqui."""
    import requests

    resp = requests.post(
        f"http://{E5_SERVER_HOST}:{E5_SERVER_PORT}",
        json={"query": subpergunta},
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {
            "doc_id": d.get("doc_id"),
            "score": d.get("score"),
            "title": d.get("title", ""),
            "contents": d.get("contents", d.get("text", "")),
        }
        for d in resp.json()[:k]
    ]


# ---------------------------------------------------------------------------
# Grafo LangGraph do fluxo SPMC-RAG (ver src/SPMC/SPMCv1.md)
#
# Funcionais: `gerar_plano`, `parsear_plano`, `resolver_placeholder`, `gerar_subpergunta`,
# `retrieve` (mini-corpus denso / servidor E5), `evidence_extractor` e `gerar_subresposta`
# -- juntos percorrem as N cadeias x L passos, resolvendo placeholders, recuperando docs,
# extraindo evidencia e gerando a sub-resposta de cada passo, acumulando o contexto de
# cada cadeia. `montar_contexto` e `gerar_resposta_final` continuam stubs.
# ---------------------------------------------------------------------------


class SPMCState(TypedDict):
    """Estado compartilhado entre os nos do grafo."""

    questao: str
    N: int
    L: int
    verbose: bool
    plano: str                              # JSON cru produzido por no_gerar_plano
    plano_estruturado: Plano | None         # plano validado (None ate no_parsear_plano rodar)
    cadeia_atual: int                       # indice da cadeia em execucao
    passo_atual: int                        # indice do passo (hop) dentro da cadeia
    resolvidos: dict[str, str]              # {"S_1": valor, ...} da cadeia atual (zerado a cada nova cadeia)
    tripla_resolvida: Tripla | None         # tripla do passo atual com os placeholders conhecidos ja substituidos
    pendentes_atual: list[str]              # placeholders S_k que a tripla do passo atual ainda nao consegue resolver
    subpergunta_atual: str                  # sub-pergunta gerada para o passo atual
    subpergunta_anterior: str               # subpergunta da tentativa anterior (p/ pedir uma diferente no retry)
    tentativas_subpergunta: int             # tentativas de subpergunta gastas no passo atual (reset a cada hop)
    retry_subpergunta: bool                 # True se gerar_subresposta decidiu tentar uma nova subpergunta
    chunks_atual: list[dict]                # Top-K docs recuperados p/ a subpergunta do passo atual
    evidencias_atual: list[Tripla]          # triplas de evidencia extraidas dos chunks do passo atual
    subresposta_atual: str                  # subresposta do ultimo passo (usada na decisao de parada)
    historico: list[dict]                   # acumulador da cadeia ATUAL: {k, subpergunta, subresposta, tripla} dos passos ja feitos
    cadeias_exec: list[dict]                # historico completo de cada cadeia ja percorrida: {cadeia, descricao, hops}
    contexto: str                           # prompt com todo o historico da cadeia
    resposta_final: str
    log: Annotated[list[str], operator.add]  # trilha de nos visitados


# Entradas = INSTRUCOES_PLANO + state["questao"] (+ state["N"], state["L"])
# Saida = state["plano"]: plano global (JSON cru da Gemini) com N cadeias de L passos.
# Regenera (ate PLAN_JSON_RETRIES vezes) se o JSON sair invalido ou nao validar contra
# Plano -- response_format=json_object garante JSON, mas nao garante o shape certo nem
# sintaxe 100% valida (ex. virgula sobrando).
def no_gerar_plano(state: SPMCState) -> dict:
    plano = None
    for tentativa in range(1, PLAN_JSON_RETRIES + 2):  # 1 tentativa normal + N extras
        plano = _chamar_gemini(
            INSTRUCOES_PLANO.format(N=state["N"], L=state["L"]),
            state["questao"],
            max_tokens=PLAN_MAX_TOKENS,
            verbose=state.get("verbose", False),
            json_mode=True,
        )
        try:
            Plano.model_validate_json(_extrair_json(plano))
            break
        except (ValidationError, ValueError) as e:
            if tentativa <= PLAN_JSON_RETRIES:
                print(
                    f"[gerar_plano] invalid JSON on attempt {tentativa} "
                    f"({type(e).__name__}); regenerating...",
                    file=sys.stderr,
                )
            # esgotadas as tentativas: devolve o ultimo texto mesmo assim -- parsear_plano
            # relanca o erro com o JSON cru completo no stderr pra diagnostico.

    return {"plano": plano, "log": ["gerar_plano"]}


# Entrada = state["plano"] (JSON cru da Gemini)
# Saida = state["plano_estruturado"]: objeto Plano validado (N cadeias x L passos)
def no_parsear_plano(state: SPMCState) -> dict:
    try:
        plano = Plano.model_validate_json(_extrair_json(state["plano"]))
    except ValidationError:
        print(
            "[parsear_plano] raw JSON that failed validation:\n" + state["plano"],
            file=sys.stderr,
        )
        raise

    # checagem suave: avisa se as contagens fugirem de N / L, mas nao aborta
    if len(plano.chains) != state["N"]:
        print(
            f"[parsear_plano] expected {state['N']} chains, got {len(plano.chains)}",
            file=sys.stderr,
        )
    for c in plano.chains:
        if len(c.steps) != state["L"]:
            print(
                f"[parsear_plano] chain {c.id}: expected {state['L']} steps, "
                f"got {len(c.steps)}",
                file=sys.stderr,
            )
        for p in c.steps:
            t = p.triple
            if not (t.subject and t.predicate and t.object):
                print(
                    f"[parsear_plano] chain {c.id} step {p.k}: incomplete triple "
                    f"<{t.subject!r}; {t.predicate!r}; {t.object!r}>",
                    file=sys.stderr,
                )

    # cadeia_atual/passo_atual/resolvidos/historico/cadeias_exec nao sao mais
    # inicializados aqui: cada cadeia agora roda seu proprio subgrafo isolado (ver
    # no_executar_cadeias), com esses campos frescos por chamada -- nao existe mais
    # um percurso sequencial unico pelas N cadeias no grafo de nivel superior.
    return {"plano_estruturado": plano, "log": ["parsear_plano"]}


# Entrada = tripla do passo atual + state["resolvidos"] (subrespostas ja obtidas na cadeia)
# Saida = state["tripla_resolvida"]: a tripla com os [S_k: ...: ??] conhecidos substituidos;
#         state["pendentes_atual"]: os placeholders de passo.depends_on que AINDA nao tem
#         valor -- se nao vazio, os nos seguintes (gerar_subpergunta/retrieve) pulam a
#         chamada de LLM, pois a tripla nao tem entidade concreta pra perguntar. NAO conta
#         o placeholder que o proprio passo define (esse sempre aparece "pendente" na
#         tripla ate o passo rodar -- e normal, nao e dependencia faltando).
def no_resolver_placeholder(state: SPMCState) -> dict:
    plano = state["plano_estruturado"]
    cadeia = plano.chains[state["cadeia_atual"]]
    passo = cadeia.steps[state["passo_atual"]]

    # Passo 1 de uma cadeia: zera placeholders e acumulador (cada cadeia tem seu S_1..S_L).
    nova_cadeia = state["passo_atual"] == 0
    resolvidos = {} if nova_cadeia else dict(state.get("resolvidos", {}))

    tripla_resolvida, pendentes_tripla = _resolver_tripla(passo.triple, resolvidos)

    # dependencias que o passo declarou (depends_on) e que ainda faltam -- diferente de
    # pendentes_tripla, que tambem inclui o placeholder que este passo vai definir.
    faltando = [d for d in passo.depends_on if d not in resolvidos]

    if state.get("verbose", False):
        extra = f"  pending={pendentes_tripla}" if pendentes_tripla else ""
        print(
            f"[resolver_placeholder] chain {cadeia.id} step {passo.k}: "
            f"{_tripla_str(tripla_resolvida)}{extra}",
            file=sys.stderr,
        )

    saida = {
        "resolvidos": resolvidos,
        "tripla_resolvida": tripla_resolvida,
        "pendentes_atual": faltando,
        # inicio de um hop novo (nao de um retry de subpergunta): zera o contador.
        "tentativas_subpergunta": 0,
        "retry_subpergunta": False,
        "log": ["resolver_placeholder"],
    }
    if nova_cadeia:
        saida["historico"] = []
    return saida


# Entradas = state["historico"] (contexto da cadeia) + texto-guia + state["tripla_resolvida"]
# Saida = state["subpergunta_atual"]: sub-pergunta em linguagem natural gerada pela Gemini.
# Pula a chamada ao LLM se state["pendentes_atual"] nao estiver vazio -- a tripla ainda
# tem placeholder [S_k: ...] sem valor, entao nao ha entidade concreta pra perguntar (o
# passo vai virar SEM_RESPOSTA em gerar_subresposta de qualquer forma, sem gastar tokens).
def no_gerar_subpergunta(state: SPMCState) -> dict:
    plano = state["plano_estruturado"]
    cadeia = plano.chains[state["cadeia_atual"]]
    passo = cadeia.steps[state["passo_atual"]]

    if state.get("pendentes_atual"):
        if state.get("verbose", False):
            print(
                f"[gerar_subpergunta] chain {cadeia.id} step {passo.k}: "
                f"pending={state['pendentes_atual']} -- skipping",
                file=sys.stderr,
            )
        return {"subpergunta_atual": "", "log": ["gerar_subpergunta"]}

    tripla = state.get("tripla_resolvida") or passo.triple
    # tipo do PROPRIO placeholder deste passo (ex. "character"), extraido da tripla
    # original -- trava a subpergunta nesse alvo, evita ela vazar pra um passo futuro.
    alvo_tipo = _tipo_placeholder(passo.triple, passo.define)
    alvo_linha = (
        f'  target of THIS step (ask for exactly this, nothing more): a value of '
        f'type "{alvo_tipo}" (placeholder {passo.define})\n'
        if alvo_tipo
        else ""
    )

    # retry (tentativas_subpergunta > 0): a tentativa anterior nao achou resposta --
    # pede uma subpergunta DIFERENTE pro mesmo alvo, em vez de repetir a mesma.
    retry_linha = (
        f'\nPREVIOUS ATTEMPT found no answer: "{state.get("subpergunta_anterior", "")}"\n'
        f"Generate a DIFFERENT sub-question for the same target above -- broader, "
        f"rephrased, or approaching from another angle. Do not repeat the previous wording.\n"
        if state.get("tentativas_subpergunta", 0) > 0
        else ""
    )

    user = (
        f"Original question: {state['questao']}\n\n"
        f"CONTEXT (previous steps in this chain):\n"
        f"{_formatar_contexto(state.get('historico', []))}\n\n"
        f"Current step (k={passo.k}):\n"
        f"  guide text: {passo.guide_text}\n"
        f"{alvo_linha}"
        f"  triple template (partially resolved): {_tripla_str(tripla)}\n"
        f"{retry_linha}\n"
        f"Generate the sub-question."
    )
    subpergunta = _chamar_corag(
        INSTRUCOES_SUBPERGUNTA,
        user,
        max_tokens=SUBQ_MAX_TOKENS,
        verbose=state.get("verbose", False),
    ).strip()

    if state.get("verbose", False):
        tentativa = state.get("tentativas_subpergunta", 0)
        sufixo = f"  (retry {tentativa}/{SUBQ_RETRY_MAX})" if tentativa else ""
        print(
            f"[gerar_subpergunta] chain {cadeia.id} step {passo.k}: {subpergunta}{sufixo}",
            file=sys.stderr,
        )
    return {"subpergunta_atual": subpergunta, "log": ["gerar_subpergunta"]}


# Entradas = state["subpergunta_atual"]
# Saida = state["chunks_atual"]: Top-K docs [{doc_id, score, title, contents}]
# Fonte: SPMC_RETRIEVER ("mini" = denso in-process sobre data/mini/ ; "e5_server" = reserva HTTP).
# Pula a busca se state["pendentes_atual"] nao estiver vazio (gerar_subpergunta ja pulou
# e devolveu subpergunta_atual vazia -- nao ha o que buscar).
def no_retrieve(state: SPMCState) -> dict:
    if state.get("pendentes_atual"):
        return {"chunks_atual": [], "log": ["retrieve"]}

    subpergunta = state["subpergunta_atual"]
    buscar = _retrieve_e5_server if SPMC_RETRIEVER == "e5_server" else _retrieve_mini_corpus
    chunks = buscar(subpergunta, k=SPMC_TOP_K)

    if state.get("verbose", False):
        print(
            f"[retrieve:{SPMC_RETRIEVER}] {len(chunks)} chunks -> "
            f"{[c['title'] for c in chunks]}",
            file=sys.stderr,
        )
    return {"chunks_atual": chunks, "log": ["retrieve"]}


# Entradas = state["chunks_atual"] (Top-K) + state["subpergunta_atual"] + INSTRUCOES_EVIDENCE
# Saida = state["evidencias_atual"]: list[Tripla] <s; p; o> ancoradas na subpergunta
def no_evidence_extractor(state: SPMCState) -> dict:
    chunks = state.get("chunks_atual", [])
    subpergunta = state.get("subpergunta_atual", "")

    if not chunks:
        # nada recuperado -- nao vale a pena chamar o LLM, devolve vazio direto
        if state.get("verbose", False):
            print("[evidence_extractor] no chunks, skipping LLM call", file=sys.stderr)
        return {"evidencias_atual": [], "log": ["evidence_extractor"]}

    user = (
        f"Sub-question: {subpergunta}\n\n"
        f"Retrieved passages:\n{_formatar_chunks(chunks)}\n\n"
        f"Extract the evidence triples."
    )
    bruto = _chamar_corag(
        INSTRUCOES_EVIDENCE,
        user,
        max_tokens=EVIDENCE_MAX_TOKENS,
        verbose=state.get("verbose", False),
        json_mode=True,
    )

    try:
        evidencias = Evidencias.model_validate_json(_extrair_json(bruto)).evidence
    except (ValidationError, ValueError) as e:
        print(
            f"[evidence_extractor] failed to parse evidence ({type(e).__name__}): "
            f"{bruto[:300]!r} -- using empty list",
            file=sys.stderr,
        )
        evidencias = []

    if state.get("verbose", False):
        print(
            f"[evidence_extractor] {len(evidencias)} triples: "
            f"{[_tripla_str(t) for t in evidencias]}",
            file=sys.stderr,
        )

    return {"evidencias_atual": evidencias, "log": ["evidence_extractor"]}


# Entradas = state["subpergunta_atual"] + state["evidencias_atual"] (triplas de evidencia
#            -- NAO os chunks brutos) + INSTRUCOES_SUBRESPOSTA
# Saida = subresposta_k (via _chamar_corag); registra
#         passo.define em state["resolvidos"], acrescenta o hop (com o MOTIVO de uma
#         eventual falha) a state["historico"] e avanca (cadeia_atual, passo_atual). Ao
#         fim de uma cadeia, arquiva o historico completo em state["cadeias_exec"].
def no_gerar_subresposta(state: SPMCState) -> dict:
    plano = state["plano_estruturado"]
    c, k = state["cadeia_atual"], state["passo_atual"]
    cadeia = plano.chains[c]
    passo = cadeia.steps[k]

    subpergunta = state.get("subpergunta_atual", "")
    evidencias = state.get("evidencias_atual", [])
    pendentes = state.get("pendentes_atual", [])

    # motivo == None -> respondido de verdade. Senao, por que a cadeia nao travou aqui:
    #   "dependencia_ausente" -- a tripla deste passo dependia de placeholder(s) nunca
    #     resolvido(s); gerar_subpergunta/retrieve ja pularam a chamada de LLM.
    #   "sem_evidencia"       -- buscou e extraiu de verdade, mas nao achou nada relevante.
    if pendentes:
        subresposta = SEM_RESPOSTA
        motivo = "dependencia_ausente"
    elif not evidencias:
        # sem evidencia -- nao vale a pena chamar o LLM, a resposta so pode ser SEM_RESPOSTA
        subresposta = SEM_RESPOSTA
        motivo = "sem_evidencia"
    else:
        user = (
            f"Sub-question: {subpergunta}\n\n"
            f"Evidence triples:\n{_formatar_evidencias(evidencias)}\n\n"
            f"Answer the sub-question."
        )
        subresposta = _chamar_corag(
            INSTRUCOES_SUBRESPOSTA.format(sem_resposta=SEM_RESPOSTA),
            user,
            max_tokens=SUBA_MAX_TOKENS,
            verbose=state.get("verbose", False),
        ).strip()
        motivo = None if subresposta.strip().lower() != SEM_RESPOSTA else "sem_resposta_llm"

    # retry: motivo retentavel (nao "dependencia_ausente", que so uma subpergunta
    # diferente nunca resolve) + ainda ha orcamento -- gera OUTRA subpergunta pro
    # mesmo passo em vez de desistir. Nao avanca o ponteiro nem grava no historico
    # ainda; isso so acontece na tentativa que efetivamente conclui o passo.
    tentativa = state.get("tentativas_subpergunta", 0)
    pode_repetir = motivo in ("sem_evidencia", "sem_resposta_llm") and tentativa < SUBQ_RETRY_MAX

    if state.get("verbose", False):
        sufixo = f"  (reason: {motivo})" if motivo else ""
        sufixo += "  -- trying a new sub-question" if pode_repetir else ""
        print(
            f"[gerar_subresposta] chain {cadeia.id} step {passo.k}: {subresposta}{sufixo}",
            file=sys.stderr,
        )

    if pode_repetir:
        return {
            "tentativas_subpergunta": tentativa + 1,
            "subpergunta_anterior": subpergunta,
            "retry_subpergunta": True,
            "log": ["gerar_subresposta"],
        }

    resolvidos = dict(state.get("resolvidos", {}))
    # so registra o valor se realmente respondeu -- SEM_RESPOSTA fica pendente pros
    # passos seguintes (nao queremos substituir um placeholder por "no relevant
    # information found").
    if passo.define and motivo is None:
        resolvidos[passo.define] = subresposta

    # Recalcula a tripla a partir do template ORIGINAL do passo (nao de
    # tripla_resolvida, que foi montada em resolver_placeholder ANTES de sabermos a
    # subresposta deste proprio passo -- por isso nunca continha o valor que acabamos
    # de descobrir). Com `resolvidos` ja atualizado acima, o placeholder que este
    # passo define (ex. [S_2: ...]) tambem entra resolvido no registro do hop.
    tripla, _pendentes = _resolver_tripla(passo.triple, resolvidos)
    hop = {
        "k": passo.k,
        "subpergunta": subpergunta,
        "subresposta": subresposta,
        "motivo": motivo,
        "tentativas": tentativa + 1,  # quantas subperguntas foram tentadas neste passo
        "tripla": _tripla_str(tripla),
        "chunks": [ch["title"] for ch in state.get("chunks_atual", [])],
        "evidencias": [_tripla_str(t) for t in evidencias],
    }
    historico = [*state.get("historico", []), hop]

    saida = {
        "subresposta_atual": subresposta,
        "resolvidos": resolvidos,
        "historico": historico,
        "retry_subpergunta": False,
        "log": ["gerar_subresposta"],
    }

    if k + 1 < len(cadeia.steps):
        saida["cadeia_atual"], saida["passo_atual"] = c, k + 1
    else:
        # cadeia terminou: arquiva o acumulador completo e vai pra proxima cadeia
        saida["cadeias_exec"] = [
            *state.get("cadeias_exec", []),
            {"cadeia": cadeia.id, "descricao": cadeia.description, "hops": historico},
        ]
        saida["cadeia_atual"], saida["passo_atual"] = c + 1, 0

    return saida


# ---------------------------------------------------------------------------
# Execucao das N cadeias EM PARALELO (uma thread por cadeia, ver SPMC_CHAIN_THREADS)
# ---------------------------------------------------------------------------

_GRAFO_CADEIA: CompiledStateGraph | None = None
_GRAFO_CADEIA_LOCK = threading.Lock()


def construir_grafo_cadeia() -> CompiledStateGraph:
    """Subgrafo que executa UMA cadeia sozinha, do passo 1 ao L (com o loop de retry de
    subpergunta): resolver_placeholder -> gerar_subpergunta -> retrieve ->
    evidence_extractor -> gerar_subresposta -> (roteador) -- mesmos nos e mesmo roteador
    (_proximo_apos_subresposta) do grafo antigo, so que "agregar" cai direto no END em
    vez de seguir pra montar_contexto (isso agora acontece uma vez so, no nivel
    superior, depois que TODAS as cadeias tiverem voltado -- ver no_executar_cadeias)."""
    g = StateGraph(SPMCState)

    g.add_node("resolver_placeholder", no_resolver_placeholder)
    g.add_node("gerar_subpergunta", no_gerar_subpergunta)
    g.add_node("retrieve", no_retrieve)
    g.add_node("evidence_extractor", no_evidence_extractor)
    g.add_node("gerar_subresposta", no_gerar_subresposta)

    g.add_edge(START, "resolver_placeholder")
    g.add_edge("resolver_placeholder", "gerar_subpergunta")
    g.add_edge("gerar_subpergunta", "retrieve")
    g.add_edge("retrieve", "evidence_extractor")
    g.add_edge("evidence_extractor", "gerar_subresposta")
    g.add_conditional_edges(
        "gerar_subresposta",
        _proximo_apos_subresposta,
        {
            "retry_subpergunta": "gerar_subpergunta",
            "proximo_passo": "resolver_placeholder",
            "agregar": END,
        },
    )
    return g.compile()


def _grafo_cadeia() -> CompiledStateGraph:
    """Compila construir_grafo_cadeia() uma unica vez (double-checked locking -- varias
    threads de no_executar_cadeias podem chegar aqui ao mesmo tempo na 1a chamada) e
    reusa a mesma instancia compilada para todas as cadeias/threads dali em diante."""
    global _GRAFO_CADEIA
    if _GRAFO_CADEIA is None:
        with _GRAFO_CADEIA_LOCK:
            if _GRAFO_CADEIA is None:
                _GRAFO_CADEIA = construir_grafo_cadeia()
    return _GRAFO_CADEIA


def _executar_cadeia(questao: str, cadeia: Cadeia, L: int, verbose: bool) -> dict:
    """Roda UMA cadeia ate o fim (chamado de dentro de uma thread por no_executar_
    cadeias). Empacota `cadeia` sozinha num Plano de 1 elemento (os nos existentes
    esperam state["plano_estruturado"].chains[cadeia_atual] -- overview/entities/
    relations nao sao usados por eles, ficam vazios) e invoca o subgrafo dedicado.
    Devolve o hop arquivado {cadeia, descricao, hops}, no mesmo formato que ia direto
    em state["cadeias_exec"] na versao sequencial antiga."""
    mini_plano = Plano(overview="", entities=[], relations=[], chains=[cadeia])
    estado = _grafo_cadeia().invoke(
        {
            "questao": questao,
            "plano_estruturado": mini_plano,
            "cadeia_atual": 0,
            "passo_atual": 0,
            "resolvidos": {},
            "historico": [],
            "cadeias_exec": [],
            "verbose": verbose,
        },
        # 1 cadeia x L passos: cada hop pode rodar (1 + SUBQ_RETRY_MAX) vezes o ciclo
        # subpergunta->retrieve->evidence->subresposta (4 nos) + 1 resolver_placeholder.
        config={"recursion_limit": L * (5 + 4 * SUBQ_RETRY_MAX) + 10},
    )
    return estado["cadeias_exec"][0]


# Entrada = state["plano_estruturado"].chains (N cadeias, independentes ate aqui)
# Saida = state["cadeias_exec"]: uma cadeia por thread (SPMC_CHAIN_THREADS controla
#         quantas rodam ao mesmo tempo) -- cada uma no seu proprio subgrafo isolado
#         (_grafo_cadeia), sem estado compartilhado entre elas alem do resultado final.
def no_executar_cadeias(state: SPMCState) -> dict:
    plano = state["plano_estruturado"]
    threads = max(1, min(SPMC_CHAIN_THREADS, len(plano.chains)))
    verbose = state.get("verbose", False)

    if verbose:
        print(
            f"[executar_cadeias] {len(plano.chains)} chain(s), {threads} thread(s) "
            f"in parallel",
            file=sys.stderr,
        )

    resultados: list[dict | None] = [None] * len(plano.chains)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futuros = {
            ex.submit(_executar_cadeia, state["questao"], cadeia, state["L"], verbose): i
            for i, cadeia in enumerate(plano.chains)
        }
        for futuro in as_completed(futuros):
            resultados[futuros[futuro]] = futuro.result()

    return {"cadeias_exec": resultados, "log": ["executar_cadeias"]}


# Entrada = state["cadeias_exec"] (todas as cadeias ja percorridas, cada uma com seus
#           hops completos -- inclusive os que falharam)
# Saida = state["contexto"]: texto em ingles (vai pro prompt de gerar_resposta_final)
#         juntando, POR CADEIA, so os hops respondidos de verdade (motivo is None) --
#         sub-pergunta, sub-resposta e tripla resolvida. Hops sem resposta (motivo !=
#         None, ex. dependencia_ausente/sem_evidencia/sem_resposta_llm) sao descartados:
#         nao trazem informacao nova, so ruido ("no relevant information found") pro
#         LLM que gera a resposta final.
def no_montar_contexto(state: SPMCState) -> dict:
    blocos = []
    total_respondidos = 0
    total_hops = 0
    for ce in state.get("cadeias_exec", []):
        total_hops += len(ce["hops"])
        respondidos = [h for h in ce["hops"] if h.get("motivo") is None]
        if not respondidos:
            continue
        total_respondidos += len(respondidos)
        passos = "\n".join(
            f"  Step {h['k']}: {h['subpergunta']} -> {h['subresposta']}\n"
            f"    triple: {h['tripla']}"
            for h in respondidos
        )
        blocos.append(f"Chain {ce['cadeia']} ({ce['descricao']}):\n{passos}")

    contexto = "\n\n".join(blocos) if blocos else "(no sub-questions were successfully answered)"

    if state.get("verbose", False):
        print(
            f"[montar_contexto] {total_respondidos}/{total_hops} answered hops "
            f"made it into the context",
            file=sys.stderr,
        )

    return {"contexto": contexto, "log": ["montar_contexto"]}


# Entradas = state["contexto"] (agregado de no_montar_contexto) + pergunta original +
#            INSTRUCOES_RESPOSTA_FINAL
# Saida = state["resposta_final"]: resposta a pergunta original, via CoRAG.
def no_gerar_resposta_final(state: SPMCState) -> dict:
    user = (
        f"Original question: {state['questao']}\n\n"
        f"Reasoning chains (answered steps only):\n{state.get('contexto', '')}\n\n"
        f"Answer the original question."
    )
    resposta = _chamar_corag(
        INSTRUCOES_RESPOSTA_FINAL.format(sem_resposta=SEM_RESPOSTA),
        user,
        max_tokens=FINAL_MAX_TOKENS,
        verbose=state.get("verbose", False),
    ).strip()

    if state.get("verbose", False):
        print(f"[gerar_resposta_final] {resposta}", file=sys.stderr)

    return {"resposta_final": resposta, "log": ["gerar_resposta_final"]}


def _proximo_apos_subresposta(state: SPMCState) -> str:
    """Roteia o fim do hop -- usado tanto pelo subgrafo de UMA cadeia (construir_grafo_
    cadeia) quanto, na versao antiga/sequencial, por um grafo com N cadeias; hoje
    state["plano_estruturado"].chains sempre tem 1 elemento (a cadeia que esta thread
    esta executando -- ver _executar_cadeia), entao "agregar" fecha essa cadeia e cai
    no END do subgrafo, nao num "proxima cadeia" global.

    - state["retry_subpergunta"] -- a subresposta veio SEM_RESPOSTA por um motivo
      retentavel (sem_evidencia/sem_resposta_llm) e ainda ha orcamento
      (SUBQ_RETRY_MAX): volta pro gerar_subpergunta pro MESMO passo, sem avancar o
      ponteiro (no_gerar_subresposta nao avancou nesse caso).
    - senao, no_gerar_subresposta ja avancou (cadeia_atual, passo_atual). Uma
      subresposta == SEM_RESPOSTA nao aborta mais o percurso: cada passo decide por
      conta propria, via state["pendentes_atual"], se tinha como ser respondido --
      entao a cadeia sempre continua ate o fim dos seus L passos.
    """
    if state.get("retry_subpergunta"):
        return "retry_subpergunta"
    if state["cadeia_atual"] < len(state["plano_estruturado"].chains):
        return "proximo_passo"
    return "agregar"


def construir_grafo() -> CompiledStateGraph:
    """Monta o StateGraph de nivel superior do fluxo SPMC-RAG: gera o plano, valida,
    executa as N cadeias EM PARALELO (no_executar_cadeias despacha uma thread por
    cadeia, cada uma no seu proprio subgrafo -- ver construir_grafo_cadeia) e agrega na
    resposta final. E um pipeline linear (sem loop): o loop de passos/retry de cada
    cadeia vive dentro do subgrafo, nao aqui."""
    g = StateGraph(SPMCState)

    g.add_node("gerar_plano", no_gerar_plano)
    g.add_node("parsear_plano", no_parsear_plano)
    g.add_node("executar_cadeias", no_executar_cadeias)
    g.add_node("montar_contexto", no_montar_contexto)
    g.add_node("gerar_resposta_final", no_gerar_resposta_final)

    g.add_edge(START, "gerar_plano")
    g.add_edge("gerar_plano", "parsear_plano")
    g.add_edge("parsear_plano", "executar_cadeias")
    g.add_edge("executar_cadeias", "montar_contexto")
    g.add_edge("montar_contexto", "gerar_resposta_final")
    g.add_edge("gerar_resposta_final", END)

    return g.compile()





# Teste simples: compila o grafo, imprime o diagrama e roda o fluxo ponta a ponta na
# pergunta de exemplo. Todos os nos fazem trabalho real -- nenhum stub restante. As N
# cadeias rodam em paralelo (ver SPMC_CHAIN_THREADS / no_executar_cadeias).
# Uso: python src/SPMC/spmc.py
#   SPMC_N / SPMC_L         reduzem o teste (ex. SPMC_N=1 SPMC_L=3)
#   SPMC_CHAIN_THREADS=4    quantas cadeias rodam ao mesmo tempo (1 = sequencial)
#   SPMC_RETRIEVER=mini     (default) denso in-process sobre data/mini/ ; =e5_server usa a reserva HTTP
#   SPMC_TOP_K=5            docs por passo ; MINI_BASE_DIR aponta outro mini-corpus
if __name__ == "__main__":
    # Pergunta real de data/mini/questions/hotpotqa (query_id hotpotqa_dev_4264,
    # resposta "Prussian") -- ao contrario da pergunta anterior (Romeo and Juliet
    # 1968), o mini-corpus TEM os docs que sustentam os dois hops: "Royal Flash
    # (film)" (Oliver Reed interpretou Otto von Bismarck) e "Oliver Reed".
    pergunta_exemplo = (
        "What nationality was Oliver Reed's character in the film Royal Flash?"
    )
    N = int(os.getenv("SPMC_N", N_DEFAULT))
    L = int(os.getenv("SPMC_L", L_DEFAULT))

    grafo = construir_grafo()
    print("=== Top-level SPMC graph (mermaid) ===\n")
    print(grafo.get_graph().draw_mermaid())
    print("=== Per-chain subgraph -- runs 1 instance per thread (mermaid) ===\n")
    print(construir_grafo_cadeia().get_graph().draw_mermaid())

    print(f"\nQuestion: {pergunta_exemplo}\n")
    print(
        f"=== Running (plan: {GEMINI_MODEL}, hops: {CORAG_MODEL}, N={N} chains x "
        f"L={L} steps, {min(SPMC_CHAIN_THREADS, N)} thread(s) in parallel) ===\n"
    )
    try:
        estado = grafo.invoke(
            {"questao": pergunta_exemplo, "N": N, "L": L, "verbose": True},
            # o grafo de nivel superior e linear (sem loop); o limite padrao (25)
            # sobra de folga.
            config={"recursion_limit": 20},
        )
        plano = estado["plano_estruturado"]

        print("--- Raw plan JSON (start) ---")
        print(estado["plano"][:800] + (" ..." if len(estado["plano"]) > 800 else ""))

        print(
            f"\n-- nodes visited ({len(estado['log'])}): {dict(Counter(estado['log']))}"
        )
        print(
            f"-- plan validated: {len(plano.chains)} chains (expected {N}), "
            f"steps per chain: {[len(c.steps) for c in plano.chains]} (expected {L})"
        )

        print("\n=== Execution flow per chain ===")
        for ce in estado.get("cadeias_exec", []):
            print(f"\n-- Chain {ce['cadeia']}: {ce['descricao']}")
            for h in ce["hops"]:
                motivo = h.get("motivo")
                sufixo = f"  [reason: {motivo}]" if motivo else ""
                tentativas = h.get("tentativas", 1)
                if tentativas > 1:
                    sufixo += f"  [attempts: {tentativas}]"
                print(
                    f"   Step {h['k']}\n"
                    f"     sub-question: {h['subpergunta']}\n"
                    f"     chunks:       {h.get('chunks', [])}\n"
                    f"     evidence:     {h.get('evidencias', [])}\n"
                    f"     sub-answer:   {h['subresposta']}{sufixo}\n"
                    f"     triple:       {h['tripla']}"
                )

        print("\n=== Aggregated context (montar_contexto) ===")
        print(estado.get("contexto", ""))

        print("\n=== Final answer ===")
        print(estado.get("resposta_final", ""))
    except Exception as e:  # noqa: BLE001 -- teste manual, so quer ver o erro
        print(f"[failed] {type(e).__name__}: {e}")