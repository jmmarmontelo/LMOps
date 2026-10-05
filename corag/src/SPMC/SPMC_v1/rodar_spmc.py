"""Roda o SPMC-RAG sobre planos JA GERADOS por gerar_planos.py (nao gera plano de
novo) -- pula direto pra execucao das cadeias (sub-pergunta -> retrieve -> evidencia ->
sub-resposta) e resposta final, reusando spmc.no_executar_cadeias / no_montar_contexto
/ no_gerar_resposta_final sem montar um StateGraph novo (a composicao e linear).

Por padrao os hops (tudo que roda DEPOIS do plano) sao respondidos via Ollama --
--provider ollama, redirecionando spmc._chamar_corag (usado por TODOS os nos de
execucao) via monkeypatch, sem alterar spmc.py -- so pra efeito de comparacao enquanto
o endpoint remoto do CoRAG estiver fora do ar (403). Com --provider corag volta pro
comportamento original de spmc.py (precisa de BASE_URL/API_KEY/MODEL_NAME no .env).

Grava dois arquivos em data/spmc_runs/ (mesma convencao de rodar_hotpotqa30.py):
  - <run_id>.jsonl   -- 1 linha por pergunta, com resposta final, cadeias_exec (hops
                        completos) e o contexto agregado. Escrita incremental (flush a
                        cada pergunta); uma pergunta com erro nao derruba as outras.
  - index.jsonl      -- 1 linha por execucao (config + metricas + caminho do arquivo
                        acima), append-only.

Uso:
  PYTHONPATH=src python src/SPMC/rodar_spmc.py \\
      --planos data/spmc_plans/planos_hotpotqa_ollama_N4_L6_20260923-163759.jsonl \\
      --provider ollama --llm gpt-oss:120b-cloud

  # so as 3 primeiras perguntas do arquivo, pra teste rapido:
  PYTHONPATH=src python src/SPMC/rodar_spmc.py --planos <arquivo> --llm gpt-oss:120b-cloud --limit 3

  # de volta pro CoRAG (precisa do endpoint remoto no ar, ver src/reproducao/.env):
  PYTHONPATH=src python src/SPMC/rodar_spmc.py --planos <arquivo> --provider corag

ATENCAO -- modelos de raciocinio (ex. gpt-oss) gastam tokens no campo "thinking" antes
do "content" (mesmo problema ja visto na geracao do plano). Os orcamentos de hop
(SUBQ_MAX_TOKENS=256, SUBA_MAX_TOKENS=128, FINAL_MAX_TOKENS=128, EVIDENCE_MAX_TOKENS=
4000) podem ser curtos demais pro Ollama e gerar SEM_RESPOSTA por estouro de tokens,
nao por falta de informacao -- ajustavel por env sem mexer no codigo.
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from reproducao.best_of_n import calcular_metricas

from SPMC import spmc
from SPMC.gerar_planos import _chamar_ollama

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "data" / "spmc_runs"

PROVIDERS_SUPORTADOS = ("corag", "ollama")


def _instalar_provider_hops(provider: str, model: str | None) -> None:
    """Redireciona spmc._chamar_corag -- usado por TODOS os nos de execucao das
    cadeias (gerar_subpergunta, evidence_extractor, gerar_subresposta) e por
    gerar_resposta_final -- pro provider escolhido. Monkeypatch deliberado: esses nos
    chamam _chamar_corag(...) por nome de modulo (lookup em tempo de chamada, nao
    vinculado na definicao), entao trocar essa referencia antes de invocar o grafo
    muda quem responde os hops sem duplicar nenhuma logica/prompt de spmc.py."""
    if provider == "corag":
        return  # ja e o comportamento original de spmc.py, nada a fazer
    if provider != "ollama":
        raise ValueError(f"provider {provider!r} nao suportado -- use um de {PROVIDERS_SUPORTADOS}")

    def _chamar_corag_via_ollama(
        system: str, user: str, *, max_tokens: int, verbose: bool = False, json_mode: bool = False,
    ) -> str:
        return _chamar_ollama(
            system, user, model, max_tokens=max_tokens, verbose=verbose, json_mode=json_mode
        )

    spmc._chamar_corag = _chamar_corag_via_ollama


def carregar_planos(planos_path: Path, limit: int | None = None) -> tuple[list[dict], int]:
    """Le o .jsonl gerado por gerar_planos.py e devolve (linhas_validas, n_puladas) --
    linhas com plano_valido=False ou sem plano_raw sao puladas (nao da pra executar um
    plano que nunca se tornou JSON valido)."""
    validas: list[dict] = []
    puladas = 0
    with open(planos_path) as f:
        for linha_raw in f:
            linha = json.loads(linha_raw)
            if not linha.get("plano_valido") or not linha.get("plano_raw"):
                puladas += 1
                continue
            validas.append(linha)
            if limit is not None and len(validas) >= limit:
                break
    return validas, puladas


def executar_spmc_sobre_planos(linhas_planos: list[dict], detalhe_path: Path, verbose: bool = False) -> list[dict]:
    """Pra cada linha (plano ja validado), parseia o plano estruturado e roda so a
    parte do grafo que vem DEPOIS do plano -- executar_cadeias -> montar_contexto ->
    gerar_resposta_final -- em vez de spmc.construir_grafo() inteiro (que comecaria
    gerando um plano novo). Grava cada resultado em `detalhe_path` assim que fica
    pronto; uma excecao numa pergunta nao derruba as outras."""
    total = len(linhas_planos)
    resultados: list[dict] = []

    with open(detalhe_path, "w") as f:
        for idx, linha in enumerate(linhas_planos):
            print(f"=== [spmc/planos] pergunta {idx + 1}/{total}: {linha['query']} ===")
            try:
                plano = spmc.Plano.model_validate_json(spmc._extrair_json(linha["plano_raw"]))

                estado: dict = {
                    "questao": linha["query"],
                    "N": linha["N"],
                    "L": linha["L"],
                    "plano_estruturado": plano,
                    "verbose": verbose,
                }
                estado.update(spmc.no_executar_cadeias(estado))
                estado.update(spmc.no_montar_contexto(estado))
                estado.update(spmc.no_gerar_resposta_final(estado))

                resposta = estado.get("resposta_final", "")
                print(f"Resposta final: {resposta}")
                resultado = {
                    "query": linha["query"],
                    "answers": linha["answers"],
                    "estrategia": "spmc",
                    "n": linha["N"],
                    "max_path_length": linha["L"],
                    "prediction": resposta,
                    "cadeias_exec": estado.get("cadeias_exec", []),
                    "contexto": estado.get("contexto", ""),
                    "erro": None,
                }
            except Exception as e:  # noqa: BLE001 -- uma pergunta ruim nao pode matar as outras
                print(f"[ERRO] pergunta {idx + 1} falhou: {type(e).__name__}: {e}")
                resultado = {
                    "query": linha["query"],
                    "answers": linha["answers"],
                    "estrategia": "spmc",
                    "n": linha.get("N"),
                    "max_path_length": linha.get("L"),
                    # SEM_RESPOSTA (nao "") pra calcular_metricas tratar como qualquer
                    # outra sub-resposta vazia, em vez de string vazia sem semantica.
                    "prediction": spmc.SEM_RESPOSTA,
                    "cadeias_exec": [],
                    "contexto": "",
                    "erro": f"{type(e).__name__}: {e}",
                }

            resultados.append(resultado)
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
            f.flush()

    return resultados


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planos", required=True, type=Path,
        help="caminho pro .jsonl de planos ja gerado por gerar_planos.py",
    )
    parser.add_argument(
        "--provider", default="ollama", choices=PROVIDERS_SUPORTADOS,
        help="quem responde os hops (sub-pergunta/evidencia/sub-resposta/resposta "
             "final): ollama (default -- CoRAG remoto fora do ar) ou corag (original)",
    )
    parser.add_argument(
        "--llm", dest="model", default=None,
        help="modelo Ollama pros hops (obrigatorio para --provider ollama, ex. gpt-oss:120b-cloud)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="roda so as N primeiras perguntas validas do arquivo de planos (teste rapido)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.provider == "ollama" and args.model is None:
        raise SystemExit("--llm e obrigatorio para --provider ollama (ex.: --llm gpt-oss:120b-cloud)")

    _instalar_provider_hops(args.provider, args.model)

    linhas_planos, n_puladas = carregar_planos(args.planos, limit=args.limit)
    threads = min(spmc.SPMC_CHAIN_THREADS, max((l["N"] for l in linhas_planos), default=1))
    hops_desc = args.model if args.provider == "ollama" else spmc.CORAG_MODEL
    print(
        f"=== SPMC sobre planos ja gerados (hops: {args.provider}:{hops_desc}, "
        f"{threads} thread(s)/cadeia) -- {len(linhas_planos)} pergunta(s) validas "
        f"de {args.planos.name} ({n_puladas} pulada(s) por plano invalido) ==="
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"spmc_{args.planos.stem}_hops-{args.provider}_{timestamp}"
    detalhe_path = RUNS_DIR / f"{run_id}.jsonl"

    t0 = time.time()
    resultados = executar_spmc_sobre_planos(linhas_planos, detalhe_path=detalhe_path)
    tempo_total_s = time.time() - t0

    print(f"Log detalhado salvo em {detalhe_path}")

    print("--- Metricas [spmc/planos] ---")
    metricas = calcular_metricas(resultados)
    print(f"{args.planos.name}: {metricas}")

    falhas = sum(1 for r in resultados if r.get("erro"))
    if falhas:
        print(f"-- {falhas}/{len(resultados)} pergunta(s) falharam com excecao (ver campo 'erro')")

    index_path = RUNS_DIR / "index.jsonl"
    entrada_index = {
        "run_id": run_id,
        "timestamp": timestamp,
        "planos_file": str(args.planos),
        "hops_provider": args.provider,
        "hops_model": args.model if args.provider == "ollama" else spmc.CORAG_MODEL,
        "chain_threads": threads,
        "n_perguntas": len(resultados),
        "n_puladas": n_puladas,
        "n_falhas": falhas,
        "em": metricas.get("em"),
        "f1": metricas.get("f1"),
        "tempo_total_s": round(tempo_total_s, 1),
        "arquivo_detalhado": str(detalhe_path.relative_to(REPO_ROOT)),
    }
    with open(index_path, "a") as f:
        f.write(json.dumps(entrada_index, ensure_ascii=False) + "\n")
    print(f"Indice atualizado em {index_path}")
