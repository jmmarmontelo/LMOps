"""Roda o SPMC-RAG sobre as mesmas 30 perguntas de hotpotqa usadas por
src/reproducao/dynamic_chain.py (data/mini/questions/hotpotqa) -- mesmo mini-corpus,
mesmas perguntas -- pra comparar diretamente com data/rag_log_dynamic_chain_hotpotqa.jsonl.

Cada execucao grava dois arquivos em data/spmc_runs/:
  - <run_id>.jsonl   -- 1 linha por pergunta, com resposta final, cadeias_exec (hops
                        completos, com motivo/tentativas/evidencias) e o contexto
                        agregado -- da pra diagnosticar uma pergunta sem rodar de novo.
                        Escrito incrementalmente (uma pergunta por vez, com flush): se
                        o processo cair no meio, as perguntas ja feitas ficam salvas.
                        Uma excecao numa pergunta nao derruba as outras.
  - index.jsonl      -- 1 linha por execucao (config + metricas + caminho do arquivo
                        acima), append-only -- historico de todas as execucoes, pra
                        comparar N/L/EVIDENCE_MAX_TOKENS sem abrir cada arquivo grande.

Uso:
  PYTHONPATH=src python src/SPMC/rodar_hotpotqa30.py
  # env uteis (defaults ja equivalem ao dynamic_chain: N=4, L=6):
  #   SPMC_N=4  SPMC_L=6  SPMC_CHAIN_THREADS=2  GEMINI_MODEL=...  EVIDENCE_MAX_TOKENS=...
"""

import datetime as dt
import json
import os
import time
from pathlib import Path

from reproducao.best_of_n import carregar_perguntas, calcular_metricas

from SPMC import spmc

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "data" / "spmc_runs"


def executar_spmc_dataset(dataset, N: int, L: int, detalhe_path: Path) -> list[dict]:
    """Roda o grafo SPMC pergunta a pergunta (sequencial -- o paralelismo ja acontece
    DENTRO de cada pergunta, entre as N cadeias, via SPMC_CHAIN_THREADS). Grava cada
    resultado em `detalhe_path` assim que fica pronto, em vez de acumular tudo em
    memoria e escrever só no final -- ver docstring do modulo. Devolve a lista completa
    de resultados (mesmo formato salvo em disco), no shape que `calcular_metricas`
    espera (precisa de "answers" e "prediction" em cada item)."""
    grafo = spmc.construir_grafo()
    total = len(dataset)
    resultados: list[dict] = []

    with open(detalhe_path, "w") as f:
        for idx, exemplo in enumerate(dataset):
            print(f"=== [hotpotqa/spmc] pergunta {idx + 1}/{total}: {exemplo['query']} ===")
            try:
                estado = grafo.invoke(
                    {"questao": exemplo["query"], "N": N, "L": L, "verbose": False},
                    # grafo de nivel superior e linear -- ver construir_grafo() em spmc.py.
                    config={"recursion_limit": 20},
                )
                resposta = estado.get("resposta_final", "")
                print(f"Resposta final: {resposta}")
                resultado = {
                    "query": exemplo["query"],
                    "answers": exemplo["answers"],
                    "estrategia": "spmc",
                    "n": N,
                    "max_path_length": L,
                    "prediction": resposta,
                    "cadeias_exec": estado.get("cadeias_exec", []),
                    "contexto": estado.get("contexto", ""),
                    "erro": None,
                }
            except Exception as e:  # noqa: BLE001 -- uma pergunta ruim nao pode matar as outras 29
                print(f"[ERRO] pergunta {idx + 1} falhou: {type(e).__name__}: {e}")
                resultado = {
                    "query": exemplo["query"],
                    "answers": exemplo["answers"],
                    "estrategia": "spmc",
                    "n": N,
                    "max_path_length": L,
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


if __name__ == "__main__":
    N = int(os.getenv("SPMC_N", spmc.N_DEFAULT))
    L = int(os.getenv("SPMC_L", spmc.L_DEFAULT))
    threads = min(spmc.SPMC_CHAIN_THREADS, N)

    perguntas = carregar_perguntas("hotpotqa")
    print(
        f"=== SPMC (plan: {spmc.GEMINI_MODEL}, hops: {spmc.CORAG_MODEL}, "
        f"N={N} cadeias x L={L} passos, {threads} thread(s)/pergunta) sobre "
        f"{len(perguntas)} perguntas de hotpotqa ==="
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"hotpotqa_N{N}_L{L}_evid{spmc.EVIDENCE_MAX_TOKENS}_{timestamp}"
    detalhe_path = RUNS_DIR / f"{run_id}.jsonl"

    t0 = time.time()
    resultados = executar_spmc_dataset(perguntas, N=N, L=L, detalhe_path=detalhe_path)
    tempo_total_s = time.time() - t0

    print(f"Log detalhado salvo em {detalhe_path}")

    print("--- Metricas [hotpotqa/spmc] ---")
    metricas = calcular_metricas(resultados)
    print(f"hotpotqa: {metricas}")

    falhas = sum(1 for r in resultados if r.get("erro"))
    if falhas:
        print(f"-- {falhas}/{len(resultados)} pergunta(s) falharam com excecao (ver campo 'erro')")

    index_path = RUNS_DIR / "index.jsonl"
    entrada_index = {
        "run_id": run_id,
        "timestamp": timestamp,
        "N": N,
        "L": L,
        "chain_threads": threads,
        "evidence_max_tokens": spmc.EVIDENCE_MAX_TOKENS,
        "gemini_model": spmc.GEMINI_MODEL,
        "corag_model": spmc.CORAG_MODEL,
        "n_perguntas": len(perguntas),
        "n_falhas": falhas,
        "em": metricas.get("em"),
        "f1": metricas.get("f1"),
        "tempo_total_s": round(tempo_total_s, 1),
        "arquivo_detalhado": str(detalhe_path.relative_to(REPO_ROOT)),
    }
    with open(index_path, "a") as f:
        f.write(json.dumps(entrada_index, ensure_ascii=False) + "\n")
    print(f"Indice atualizado em {index_path}")
