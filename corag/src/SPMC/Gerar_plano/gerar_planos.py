"""Gera SO o plano (prompt INSTRUCOES_PLANO abaixo + validacao contra Plano, via
Gemini) para as perguntas de um dataset (hotpotqa, 2wikimultihopqa ou musique), sem
tocar no CoRAG -- util enquanto o endpoint remoto do CoRAG estiver fora do ar (403),
pra continuar coletando dados de plano sem depender dele.

Grava data/spmc_plans/<run_id>.jsonl, uma linha por pergunta, escrita incrementalmente
(mesma resiliencia de rodar_hotpotqa30.py -- uma pergunta com erro nao derruba as outras).

Uso:
  PYTHONPATH=src python src/SPMC/gerar_planos.py --task hotpotqa
  PYTHONPATH=src python src/SPMC/gerar_planos.py --task 2wikimultihopqa --llm gemini-flash-lite-latest
  PYTHONPATH=src python src/SPMC/gerar_planos.py --task musique
  # env uteis: SPMC_N=4  SPMC_L=6  (--llm tem prioridade sobre a env GEMINI_MODEL)

  # provider "ollama" -- modelo local servido por `ollama serve` (precisa de
  # `ollama pull gpt-oss:120b` antes; ver _chamar_ollama):
  PYTHONPATH=src python src/SPMC/gerar_planos.py --task hotpotqa --provider ollama --llm gpt-oss:120b
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from pydantic import ValidationError
from reproducao.best_of_n import carregar_perguntas

from SPMC import spmc

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANS_DIR = REPO_ROOT / "data" / "spmc_plans"

# Datasets suportados por este script (nomes = pastas em data/mini/questions/, ver
# reproducao.best_of_n.carregar_perguntas).
TASKS_SUPORTADAS = ("hotpotqa", "2wikimultihopqa", "musique")

# Provedores de LLM suportados pra gerar o plano -- "gemini" (API, via spmc._chamar_llm)
# ou "ollama" (modelo local servido por `ollama serve`, ex. gpt-oss:120b -- ver
# _chamar_ollama). O nome do modelo em si vem de --llm, independente do provider.
PROVIDERS_SUPORTADOS = ("gemini", "ollama")

# Mesmo prompt de spmc.INSTRUCOES_PLANO -- copiado aqui pra este script gerar o plano
# direto via Gemini (ver _gerar_plano_gemini), sem depender do no_gerar_plano do grafo
# LangGraph inteiro em spmc.py.
INSTRUCOES_PLANO = PLAN_PROMPT = """\
You are the Plan Generator of SPMC-RAG, a multi-hop retrieval-augmented question
answering system. Given a complex multi-hop query, you produce a reasoning PLAN made
of {N} alternative reasoning chains, each with exactly {L} steps. The plan is a
skeleton, not an answer key: sub-questions, retrieval and answers are produced later
by an executor, which runs each chain step by step and fills in the placeholders
with the values it retrieves.

## Key concepts

Step: each step has a guide text (what the step must discover, used to generate the
sub-question) and a triple template.

Triple: <subject; predicate; object>, written so that it reads as a short sentence,
e.g. <[S_1: person: ??]; directed; Titanic (1997)>.
- "subject" and "object" hold entity names copied verbatim from the query,
  placeholders defined by earlier steps of the same chain, the placeholder this
  step defines, or a constant used only for verification (e.g. "film director").
  They never hold a value that should be retrieved.
- "predicate" is a short relation phrase ("directed", "born in", "spouse of") and
  never contains a placeholder.

Placeholder: [S_k: type: ??] marks a value that is only known after step k is
executed. "type" is a short, generic, lowercase semantic type: person, film, book,
city, country, organization, date, year, number, yes/no, etc.
- Step k defines at most one new placeholder, and it is always named S_k.
- The executor finds placeholders by exact string match. Whenever a later step
  reuses a placeholder, copy it exactly as it was defined: if step 1 defined
  [S_1: person: ??], write [S_1: person: ??] again, never [S_1: director: ??]
  or [S_1]. A different string is never filled in.

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
   Two chains with the same triples in the same order are duplicates, not
   alternatives.
4. Write each chain's steps in dependency order, then check every rule below
   before answering.

## Rules

Structure
- Exactly {N} chains, with "id" from 1 to {N}; each chain has exactly {L} steps,
  with "k" from 1 to {L}.
- Every step has the key "triple" (exactly this name, never "triple_template"
  nor loose fields on the step), with "subject", "predicate" and "object" all
  filled in (non-empty strings).

Placeholders and dependencies
- "define" is "S_k" (the step's own k) or null. If it is "S_k", the placeholder
  [S_k: type: ??] MUST appear literally in this step's triple, in the position
  that is still unknown.
- NEVER write the real value in place of a placeholder, even if you already know
  it. Values are found later by retrieval; a pre-filled value breaks the chain,
  because the steps that depend on it never receive the retrieved value.
- "depends_on" lists exactly the earlier placeholders that appear in this step's
  triple (e.g. ["S_1"]), no more and no fewer, and only S_j with j < k from the
  same chain.
- "define": null is allowed only for verification steps, whose triple contains
  only query entities, verification constants and earlier placeholders.

Number of hops
- If the query needs fewer than {L} hops, use the extra steps BEFORE the last one
  to verify or disambiguate entities already found (e.g. confirm an entity's
  occupation, or a date that distinguishes homonyms). Never produce fewer than
  {L} steps, and never pad with repeated triples.

Final step
- Step {L} produces the final answer: its "define" is "S_{L}" and its
  placeholder type matches "answer_type".
- For comparison or yes/no queries ("which was released first?", "are both X?"),
  the earlier steps retrieve each attribute to be compared and the last step
  compares them, e.g. with 3 steps:
  {{"subject": "Film A ([S_1: date: ??]) vs Film B ([S_2: date: ??])",
    "predicate": "released earlier", "object": "[S_3: film: ??]"}}

guide_text
- One or two sentences stating what the step must discover. Refer to earlier
  steps of THIS chain by what they found (e.g. "the director found in step 1"),
  never by a concrete value.

## Output

Respond ONLY with a valid JSON object: no markdown, no ``` fences, no comments,
no text before or after it. Fields:
- "overview": what the query requires discovering, in one or two sentences.
- "entities": key entities of the query.
- "relations": relations involved.
- "answer_type": type of the final answer (same vocabulary as placeholder types).
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
          "guide_text": "Identify who directed the film Titanic (1997).",
          "triple": {{"subject": "[S_1: person: ??]", "predicate": "directed", "object": "Titanic (1997)"}},
          "define": "S_1",
          "depends_on": []
        }},
        {{
          "k": 2,
          "guide_text": "Find the city where the director identified in step 1 was born.",
          "triple": {{"subject": "[S_1: person: ??]", "predicate": "born in", "object": "[S_2: city: ??]"}},
          "define": "S_2",
          "depends_on": ["S_1"]
        }},
        {{
          "k": 3,
          "guide_text": "Find the country where the birth city found in step 2 is located.",
          "triple": {{"subject": "[S_2: city: ??]", "predicate": "located in country", "object": "[S_3: country: ??]"}},
          "define": "S_3",
          "depends_on": ["S_2"]
        }}
      ]
    }},
    {{
      "id": 2,
      "description": "Finds the director through the inverse relation, confirms the identity, and asks directly for the country of birth.",
      "steps": [
        {{
          "k": 1,
          "guide_text": "Identify the director credited for the film Titanic (1997).",
          "triple": {{"subject": "Titanic (1997)", "predicate": "directed by", "object": "[S_1: person: ??]"}},
          "define": "S_1",
          "depends_on": []
        }},
        {{
          "k": 2,
          "guide_text": "Confirm that the person found in step 1 is a film director, to rule out homonyms.",
          "triple": {{"subject": "[S_1: person: ??]", "predicate": "occupation", "object": "film director"}},
          "define": null,
          "depends_on": ["S_1"]
        }},
        {{
          "k": 3,
          "guide_text": "Find the country where the director confirmed in step 2 was born.",
          "triple": {{"subject": "[S_1: person: ??]", "predicate": "country of birth", "object": "[S_3: country: ??]"}},
          "define": "S_3",
          "depends_on": ["S_1"]
        }}
      ]
    }}
  ]
}}
"""

def _chamar_ollama(
    system: str, user: str, model: str, *, max_tokens: int,
    verbose: bool = False, json_mode: bool = False,
) -> str:
    """Chamada a um modelo local servido pelo Ollama (ex.: gpt-oss:120b) via
    ollama.chat -- mesma interface (system/user/max_tokens/json_mode) que
    spmc._chamar_llm, mas falando com um servidor Ollama (OLLAMA_HOST, default
    http://localhost:11434) em vez de um endpoint OpenAI-compat. Requer `ollama serve`
    rodando e o modelo ja baixado (`ollama pull <model>`). Import local (como
    torch/transformers em spmc.py): so paga o custo de importar `ollama` quem usa
    --provider ollama."""
    import httpx
    import ollama

    resp = None
    ultimo_erro: Exception | None = None
    for tentativa in range(1, spmc.PLAN_MAX_RETRIES + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format="json" if json_mode else None,
                options={"temperature": 0, "num_predict": max_tokens},
            )
            break
        except ollama.ResponseError as e:
            # erro de status HTTP do servidor Ollama -- so vale re-tentar em erro
            # transitorio (5xx/429); 4xx tipo 404 "model not found" ou 410 "retired"
            # (ver smoke test) nunca vai se resolver tentando de novo.
            if e.status_code not in spmc._STATUS_RETENTAVEIS:
                raise
            ultimo_erro = e
        except httpx.HTTPError as e:
            # erro de transporte (conexao recusada, timeout etc.) -- servidor Ollama
            # pode estar subindo ainda, vale re-tentar.
            ultimo_erro = e

        if tentativa < spmc.PLAN_MAX_RETRIES:
            espera = min(2 ** tentativa, 30)
            print(
                f"[ollama:{model}] attempt {tentativa}/{spmc.PLAN_MAX_RETRIES} failed "
                f"({type(ultimo_erro).__name__}: {ultimo_erro}); retrying in {espera}s",
                file=sys.stderr,
            )
            time.sleep(espera)

    if resp is None:
        raise ultimo_erro

    texto = resp.message.content
    if verbose:
        print(
            f"[{model}] done_reason={resp.done_reason}  "
            f"completion_tokens={resp.eval_count}  prompt_tokens={resp.prompt_eval_count}  "
            f"chars={len(texto or '')}",
            file=sys.stderr,
        )
    return texto


def _gerar_plano_llm(
    questao: str, N: int, L: int, provider: str, model: str, verbose: bool = False,
) -> str:
    """Chama o LLM escolhido (provider="gemini", via spmc._chamar_llm; ou
    provider="ollama", modelo local via _chamar_ollama -- ver PROVIDERS_SUPORTADOS)
    com o INSTRUCOES_PLANO deste arquivo pra gerar o plano bruto (JSON), regenerando
    (ate spmc.PLAN_JSON_RETRIES vezes) se o JSON sair invalido ou nao validar contra
    spmc.Plano -- mesma logica de spmc.no_gerar_plano, mas usando o prompt local acima
    e o provider/modelo explicitos, em vez de rodar o no dentro do grafo LangGraph (que
    sempre usa a Gemini fixa do .env)."""
    system = INSTRUCOES_PLANO.format(N=N, L=L)

    plano = None
    for tentativa in range(1, spmc.PLAN_JSON_RETRIES + 2):  # 1 tentativa normal + N extras
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY nao encontrada. Defina-a em src/reproducao/.env ou no ambiente."
                )
            plano = spmc._chamar_llm(
                spmc.GEMINI_BASE_URL, api_key, model, system, questao,
                max_tokens=spmc.PLAN_MAX_TOKENS, verbose=verbose, json_mode=True,
            )
        elif provider == "ollama":
            plano = _chamar_ollama(
                system, questao, model,
                max_tokens=spmc.PLAN_MAX_TOKENS, verbose=verbose, json_mode=True,
            )
        else:
            raise ValueError(
                f"provider {provider!r} nao suportado -- use um de {PROVIDERS_SUPORTADOS}"
            )

        try:
            spmc.Plano.model_validate_json(spmc._extrair_json(plano))
            break
        except (ValidationError, ValueError) as e:
            if tentativa <= spmc.PLAN_JSON_RETRIES:
                print(
                    f"[gerar_plano] invalid JSON on attempt {tentativa} "
                    f"({type(e).__name__}); regenerating...",
                    file=sys.stderr,
                )
            # esgotadas as tentativas: devolve o ultimo texto mesmo assim -- a
            # validacao contra Plano em gerar_planos_dataset registra plano_valido=False.
    return plano


def gerar_planos_dataset(
    dataset, N: int, L: int, detalhe_path: Path,
    provider: str = "gemini", model: str = spmc.GEMINI_MODEL,
) -> list[dict]:
    """Chama so _gerar_plano_llm (provider/model) + validacao contra Plano pra cada
    pergunta do dataset -- nao invoca o grafo inteiro, nao toca no CoRAG. Grava cada
    resultado em `detalhe_path` assim que fica pronto; uma excecao numa pergunta nao
    derruba as outras."""
    total = len(dataset)
    resultados: list[dict] = []

    with open(detalhe_path, "w") as f:
        for idx, exemplo in enumerate(dataset):
            print(f"=== [plano] pergunta {idx + 1}/{total}: {exemplo['query']} ===")
            try:
                plano_raw = _gerar_plano_llm(
                    exemplo["query"], N=N, L=L, provider=provider, model=model, verbose=True
                )

                try:
                    plano_estruturado = spmc.Plano.model_validate_json(
                        spmc._extrair_json(plano_raw)
                    )
                    valido = True
                    n_cadeias = len(plano_estruturado.chains)
                    passos_por_cadeia = [len(c.steps) for c in plano_estruturado.chains]
                except Exception as e:  # ValidationError/ValueError -- plano nao bateu com o schema
                    valido = False
                    n_cadeias = None
                    passos_por_cadeia = None
                    print(f"    plano nao validou: {type(e).__name__}: {e}")

                if valido:
                    print(f"    plano ok: {n_cadeias} cadeias, passos={passos_por_cadeia}")

                resultado = {
                    "query": exemplo["query"],
                    "answers": exemplo["answers"],
                    "N": N,
                    "L": L,
                    "llm_provider": provider,
                    "llm_model": model,
                    "plano_raw": plano_raw,
                    "plano_valido": valido,
                    "n_cadeias": n_cadeias,
                    "passos_por_cadeia": passos_por_cadeia,
                    "erro": None,
                }
            except Exception as e:  # noqa: BLE001 -- uma pergunta ruim nao pode matar as outras
                print(f"[ERRO] pergunta {idx + 1} falhou: {type(e).__name__}: {e}")
                resultado = {
                    "query": exemplo["query"],
                    "answers": exemplo["answers"],
                    "N": N,
                    "L": L,
                    "llm_provider": provider,
                    "llm_model": model,
                    "plano_raw": None,
                    "plano_valido": False,
                    "n_cadeias": None,
                    "passos_por_cadeia": None,
                    "erro": f"{type(e).__name__}: {e}",
                }

            resultados.append(resultado)
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
            f.flush()

    return resultados


def gerar_planos_para_task(
    task: str, N: int, L: int, provider: str = "gemini", model: str = spmc.GEMINI_MODEL,
) -> Path:
    """Carrega as perguntas de `task` (uma de TASKS_SUPORTADAS: hotpotqa,
    2wikimultihopqa, musique -- mesmas pastas de reproducao.best_of_n.carregar_perguntas)
    e gera um plano (N cadeias x L passos, via `provider`/`model` -- ver --provider/--llm)
    para cada uma via gerar_planos_dataset, gravando incrementalmente em
    data/spmc_plans/<run_id>.jsonl. Devolve o Path do arquivo gravado."""
    if task not in TASKS_SUPORTADAS:
        raise ValueError(f"task {task!r} nao suportada -- use uma de {TASKS_SUPORTADAS}")
    if provider not in PROVIDERS_SUPORTADOS:
        raise ValueError(f"provider {provider!r} nao suportado -- use um de {PROVIDERS_SUPORTADOS}")

    perguntas = carregar_perguntas(task)
    print(
        f"=== Gerando planos ({provider}: {model}, N={N} cadeias x L={L} "
        f"passos) sobre {len(perguntas)} perguntas de {task} ==="
    )

    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"planos_{task}_{provider}_N{N}_L{L}_{timestamp}"
    detalhe_path = PLANS_DIR / f"{run_id}.jsonl"

    t0 = time.time()
    resultados = gerar_planos_dataset(
        perguntas, N=N, L=L, detalhe_path=detalhe_path, provider=provider, model=model
    )
    tempo_total_s = time.time() - t0

    validos = sum(1 for r in resultados if r["plano_valido"])
    falhas = sum(1 for r in resultados if r.get("erro"))
    print(f"Log salvo em {detalhe_path}")
    print(
        f"-- {validos}/{len(resultados)} planos validos, {falhas} falha(s), "
        f"{tempo_total_s:.1f}s total"
    )
    return detalhe_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="hotpotqa", choices=TASKS_SUPORTADAS,
        help="dataset cujas perguntas vao gerar planos (default: hotpotqa)",
    )
    parser.add_argument(
        "--provider", default="gemini", choices=PROVIDERS_SUPORTADOS,
        help="provedor do LLM: gemini (API, default) ou ollama (modelo local, "
             "ex. gpt-oss:120b -- precisa de `ollama serve` + `ollama pull <modelo>`)",
    )
    parser.add_argument(
        "--llm", dest="model", default=None,
        help="nome do modelo no provider escolhido (default: spmc.GEMINI_MODEL/env "
             "GEMINI_MODEL para --provider gemini; obrigatorio para --provider ollama)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    N = int(os.getenv("SPMC_N", spmc.N_DEFAULT))
    L = int(os.getenv("SPMC_L", spmc.L_DEFAULT))

    model = args.model
    if model is None:
        if args.provider == "gemini":
            model = spmc.GEMINI_MODEL
        else:
            raise SystemExit(f"--llm e obrigatorio para --provider {args.provider} (ex.: --llm gpt-oss:120b)")

    gerar_planos_para_task(args.task, N=N, L=L, provider=args.provider, model=model)
