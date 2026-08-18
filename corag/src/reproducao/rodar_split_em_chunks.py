"""Roda um split completo (ex.: 2wikimultihopqa validation) em chunks sequenciais de N
perguntas, construindo um mini-corpus por chunk (em vez de carregar o corpus KILT
inteiro/indice E5 completo de uma vez). Cada chunk grava seu proprio log em
data/chunks/{TASK}/{ESTRATEGIA}/rag_log_{start}-{end}.jsonl (log/metricas isolados por
estrategia, mini-corpus em data/mini_chunks/{TASK}/ compartilhado entre estrategias);
se esse arquivo ja existir o chunk e pulado, entao interromper a execucao (Ctrl+C,
crash) so faz perder o chunk em andamento.

Por padrao cada EXECUCAO processa so 1 chunk novo (o proximo ainda sem log) e para
sozinha -- rode o script de novo (em outro horario, outra sessao) pra continuar de
onde parou. Isso e controlado por MAX_CHUNKS (ver abaixo).

Config via env vars (mesmo estilo de best_of_n.py / dynamic_chain.py):
    TASK, CHUNK_SIZE, ESTRATEGIA (greedy / best_of_n / dynamic_chain), N_CHAINS,
    MAX_PATH_LENGTH, NUM_THREADS, N_DISTRATORES_POR_CHUNK,
    BASE_URL, API_KEY, MODEL_NAME, LIMPAR_MINI_CORPUS_APOS_CHUNK,
    MAX_CHUNKS (orcamento de chunks NOVOS a processar nesta execucao antes de parar;
    default 1 = para a cada bloco. 0 ou negativo = sem limite, roda o split inteiro
    numa chamada so, igual ao comportamento antigo)

Metricas: calculadas ao final de cada execucao, sobre todos os chunks concluidos ate
ali (novos desta rodada + os que ja existiam de rodadas anteriores). Gravadas em
metricas_completo.json quando o split inteiro ja foi coberto, ou metricas_parcial.json
(sobrescrito a cada rodada) enquanto ainda houver chunks pendentes.
"""

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from data_utils import load_corpus
from logger_config import logger
from reproducao.best_of_n import (
    TASK_SPLITS, iniciar_servidor_e5, parar_servidor_e5,
    executar_rag, calcular_metricas,
)
from reproducao.dynamic_chain import _montar_agente, executar_dynamic_chain_dataset
from reproducao.construir_mini_corpus import (
    N_DISTRATORES_POR_BENCHMARK, SEED, INDEX_DIR_ORIGINAL,
    fatiar_dataset_sequencial, coletar_ids_contexto, amostrar_distratores,
    construir_mapa_ids, construir_mini_corpus_e_indice, remapear_e_salvar_datasets,
    construir_tabela_shards,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

TASK = os.getenv("TASK", "2wikimultihopqa")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
ESTRATEGIA = os.getenv("ESTRATEGIA", "best_of_n")
N_CHAINS = int(os.getenv("N_CHAINS", "4"))
MAX_PATH_LENGTH = int(os.getenv("MAX_PATH_LENGTH", "6"))
# Poucas threads por padrao (teste local); pode subir bastante ao rodar no servidor
# (ex.: NUM_THREADS=32), mesmo padrao de NUM_THREADS em best_of_n.py.
NUM_THREADS = int(os.getenv("NUM_THREADS", "2"))
N_DISTRATORES_POR_CHUNK = int(os.getenv("N_DISTRATORES_POR_CHUNK", str(N_DISTRATORES_POR_BENCHMARK)))
LIMPAR_MINI_CORPUS_APOS_CHUNK = os.getenv("LIMPAR_MINI_CORPUS_APOS_CHUNK", "0") == "1"
# Orcamento de chunks NOVOS processados nesta execucao antes de parar sozinha (None =
# sem limite, roda o split inteiro numa chamada so). Default 1: cada execucao do
# script processa so o proximo chunk pendente e para -- rode de novo depois pra
# continuar. 0/negativo tambem vira "sem limite".
_max_chunks_env = int(os.getenv("MAX_CHUNKS", "1"))
MAX_CHUNKS: Optional[int] = None if _max_chunks_env <= 0 else _max_chunks_env

# Compartilhado entre estrategias: a construcao do mini-corpus nao depende de ESTRATEGIA.
MINI_CHUNKS_DIR = REPO_ROOT / "data" / "mini_chunks" / TASK
# Isolado por estrategia: senao duas estrategias no mesmo TASK colidiriam nos mesmos
# rag_log_*.jsonl/metricas_*.json e uma "pularia" chunks que a outra ja processou.
LOGS_DIR = REPO_ROOT / "data" / "chunks" / TASK / ESTRATEGIA


def _matar_processo_na_porta(port: int = 8090) -> None:
    """Mata qualquer processo escutando na porta, mesmo que orfao de outra execucao/script
    (teste_paralelo.py, best_of_n.py, dynamic_chain.py, ou um chunk anterior travado).
    iniciar_servidor_e5 (best_of_n.py) so verifica se a porta ja esta aberta e, se estiver,
    REAPROVEITA o que ja esta rodando sem checar se aponta pro corpus/indice certos -- pra
    quem roda sempre o mesmo mini-corpus fixo (best_of_n.py/dynamic_chain.py) isso e o
    comportamento desejado, mas aqui cada chunk precisa de um servidor apontando pro SEU
    mini-corpus, entao garantimos a porta livre antes de cada chunk."""
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            try:
                proc = psutil.Process(conn.pid)
                logger.warning(
                    f"Matando processo na porta {port} (pid={conn.pid}, cmd={' '.join(proc.cmdline())}) "
                    f"antes de iniciar o servidor E5 deste chunk."
                )
                proc.terminate()
                proc.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass


def _ler_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]


def _salvar_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def processar_chunk(task: str, start: int, end: int, corpus_len: int, split: Optional[str], log_path: Path) -> List[Dict]:
    chunk_dir = MINI_CHUNKS_DIR / f"{start:06d}-{end:06d}"
    corpus_dir = chunk_dir / "corpus"
    index_dir = chunk_dir / "e5-index"
    questions_dir = chunk_dir / "questions"

    logger.info(f"[{task}] Construindo mini-corpus do chunk {start}-{end}...")
    fatia = fatiar_dataset_sequencial(task, start, end, split=split)
    ids_contexto = coletar_ids_contexto({task: fatia})
    distratores = amostrar_distratores(
        ids_contexto, corpus_len=corpus_len, tasks=[task],
        n_por_benchmark=N_DISTRATORES_POR_CHUNK, seed=SEED,
    )
    mapa_ids = construir_mapa_ids(ids_contexto | distratores)
    logger.info(f"[{task}] Chunk {start}-{end}: mini-corpus com {len(mapa_ids)} documentos.")
    construir_mini_corpus_e_indice(mapa_ids, str(corpus_dir), str(index_dir))
    remapear_e_salvar_datasets({task: fatia}, mapa_ids, str(questions_dir))

    _matar_processo_na_porta()
    processo_e5 = iniciar_servidor_e5(index_dir, corpus_dir)
    try:
        corpus_mini = load_corpus(corpus_dir=str(corpus_dir))
        dataset_remapeado = Dataset.load_from_disk(str(questions_dir / task))

        if ESTRATEGIA == "dynamic_chain":
            # executar_dynamic_chain_dataset (dynamic_chain.py) precisa de um CoRagAgent ja
            # montado (nao constroi um internamente como executar_rag faz).
            corag_agent = _montar_agente(
                model=os.getenv("MODEL_NAME", "corag-8b"),
                api_key=os.environ["API_KEY"],
                base_url=os.environ["BASE_URL"],
                corpus=corpus_mini,
            )
            resultados = executar_dynamic_chain_dataset(
                dataset_remapeado, corag_agent,
                n=N_CHAINS, max_path_length=MAX_PATH_LENGTH,
                log_path=str(log_path), num_threads=NUM_THREADS,
                rotulo=f"[{task} {start}-{end}] ",
            )
        else:
            resultados = executar_rag(
                dataset_remapeado, corpus_mini,
                base_url=os.environ["BASE_URL"], api_key=os.environ["API_KEY"],
                model=os.getenv("MODEL_NAME", "corag-8b"),
                max_path_length=MAX_PATH_LENGTH, log_path=str(log_path),
                estrategia=ESTRATEGIA, n=N_CHAINS, num_threads=NUM_THREADS,
            )
    finally:
        parar_servidor_e5(processo_e5)

    if LIMPAR_MINI_CORPUS_APOS_CHUNK:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    return resultados


def main() -> None:
    load_dotenv()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    inicio_execucao = time.time()

    split = TASK_SPLITS[TASK]
    total = len(load_dataset("corag/multihopqa", TASK, split=split))
    n_chunks_total = math.ceil(total / CHUNK_SIZE)
    logger.info(f"[{TASK}] {total} perguntas no split; {n_chunks_total} chunk(s) de {CHUNK_SIZE} no total"
                f"{f' (ate {MAX_CHUNKS} chunk(s) novo(s) nesta execucao)' if MAX_CHUNKS is not None else ''}.")

    tabela_shards = construir_tabela_shards(INDEX_DIR_ORIGINAL)
    corpus_len = tabela_shards[-1][2]

    todos_resultados: List[Dict] = []
    chunks_novos = 0
    chunks_pulados = 0
    parou_com_pendencias = False
    for i in range(n_chunks_total):
        start, end = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, total)
        log_path = LOGS_DIR / f"rag_log_{start:06d}-{end:06d}.jsonl"

        if log_path.exists():
            logger.info(f"Chunk {start}-{end} ja processado, pulando.")
            todos_resultados.extend(_ler_jsonl(log_path))
            chunks_pulados += 1
            continue

        if MAX_CHUNKS is not None and chunks_novos >= MAX_CHUNKS:
            parou_com_pendencias = True
            logger.info(f"Limite de {MAX_CHUNKS} chunk(s) novo(s) atingido nesta execucao; "
                        f"chunk {start}-{end} (e possiveis seguintes) ficam para a proxima execucao.")
            break

        inicio_chunk = time.time()
        resultados = processar_chunk(TASK, start, end, corpus_len=corpus_len, split=split, log_path=log_path)
        duracao_chunk = time.time() - inicio_chunk
        logger.info(f"[{TASK}] Chunk {start}-{end}: {duracao_chunk:.1f}s ({duracao_chunk / 60:.1f} min) "
                    f"para {len(resultados)} pergunta(s), media {duracao_chunk / len(resultados):.2f}s/pergunta.")
        todos_resultados.extend(resultados)
        chunks_novos += 1

    duracao_execucao = time.time() - inicio_execucao
    chunks_concluidos = chunks_pulados + chunks_novos
    nome_metricas = "metricas_parcial.json" if parou_com_pendencias else "metricas_completo.json"
    logger.info(f"--- Metricas [{TASK}] ({'parcial' if parou_com_pendencias else 'split completo'}, "
                f"{len(todos_resultados)}/{total} perguntas; {chunks_concluidos}/{n_chunks_total} chunk(s) "
                f"concluidos no total, {chunks_novos} novo(s) nesta execucao em {duracao_execucao:.1f}s "
                f"[{duracao_execucao / 60:.1f} min]) ---")
    metricas = calcular_metricas(todos_resultados)
    metricas["tempo_execucao_segundos"] = round(duracao_execucao, 1)
    metricas["chunks_novos_nesta_execucao"] = chunks_novos
    metricas["chunks_concluidos"] = chunks_concluidos
    metricas["chunks_total"] = n_chunks_total
    _salvar_json(metricas, LOGS_DIR / nome_metricas)
    print(f"{TASK}: {metricas}")
    if parou_com_pendencias:
        print(f"Execucao parcial ({chunks_novos} chunk(s) novo(s) processado(s)); "
              f"rode o script de novo para continuar de onde parou.")


if __name__ == "__main__":
    main()
