"""Constroi um indice FAISS (IVF+PQ) a partir dos shards de embeddings E5 ja
baixados em INDEX_DIR (data/e5-large-index/ por padrao), sem carregar os
~70-80GB de embeddings inteiros na memoria de uma vez -- cada shard e
convertido de float16 para float32 em lotes pequenos (BATCH_SIZE), nunca o
shard inteiro de uma vez.

Uso:
    PYTHONPATH=src python src/search/construir_indice_faiss.py

Parametros (env vars). PQ_M=256 (default) foi escolhido apos medir recall@10
real (vs. busca exata) em varias opcoes: m=32 dava so 10% de recall (baixo
demais, causava muitas subperguntas "no relevant information found" no
pipeline); m=256 da ~74% de recall, indice de ~9.1GB pros ~35.7M vetores do
corpus -- precisa de mais RAM que o m=32 antigo (que usava so ~1.4GB), exige
maquina/config (ex.: .wslconfig no WSL) com uns 9-10GB de RAM disponiveis:
    NLIST (default 4096), PQ_M (default 256, bytes/vetor comprimido),
    PQ_NBITS (default 8), BATCH_SIZE (default 200000, vetores por lote na
    hora de converter+adicionar), TRAIN_PER_SHARD (default 75000, quantos
    vetores de treino tirar de cada shard amostrado), TRAIN_SHARD_STRIDE
    (default 10, pega 1 a cada N shards pra amostra de treino, em vez de so
    o primeiro -- mais representativo), INDEX_DIR, OUTPUT_DIR.
"""
import os
import time

import faiss
import numpy as np
import torch
from tqdm import tqdm

from search.e5_searcher import _get_all_shards_path
from logger_config import logger

DIM = 1024
NLIST = int(os.getenv("NLIST", "4096"))
PQ_M = int(os.getenv("PQ_M", "256"))
PQ_NBITS = int(os.getenv("PQ_NBITS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200000"))
TRAIN_PER_SHARD = int(os.getenv("TRAIN_PER_SHARD", "75000"))
TRAIN_SHARD_STRIDE = int(os.getenv("TRAIN_SHARD_STRIDE", "10"))

INDEX_DIR = os.getenv("INDEX_DIR", "data/e5-large-index")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/e5-large-index-faiss")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "index.faiss")


def _carregar_shard(path: str) -> torch.Tensor:
    return torch.load(path, mmap=True, weights_only=True, map_location="cpu")


def montar_amostra_treino(shard_paths: list) -> np.ndarray:
    shards_treino = shard_paths[::TRAIN_SHARD_STRIDE] or shard_paths[:1]
    amostras = []
    for path in tqdm(shards_treino, desc="Amostrando shards p/ treino"):
        shard = _carregar_shard(path)
        n = min(TRAIN_PER_SHARD, shard.shape[0])
        amostras.append(shard[:n].float().numpy())
        del shard
    return np.concatenate(amostras, axis=0)


def main() -> None:
    t0 = time.time()
    shard_paths = _get_all_shards_path(INDEX_DIR)
    logger.info(f"{len(shard_paths)} shards encontrados em {INDEX_DIR}")

    comprimentos = [_carregar_shard(p).shape[0] for p in tqdm(shard_paths, desc="Lendo tamanhos dos shards")]
    total_esperado = sum(comprimentos)
    logger.info(f"Total esperado de vetores: {total_esperado}")

    amostra = montar_amostra_treino(shard_paths)
    logger.info(f"[{time.time() - t0:.1f}s] Amostra de treino: {amostra.shape}, {amostra.nbytes / 1e9:.2f}GB")

    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFPQ(quantizer, DIM, NLIST, PQ_M, PQ_NBITS)
    logger.info(f"Treinando indice (nlist={NLIST}, m={PQ_M}, nbits={PQ_NBITS})...")
    index.train(amostra)
    logger.info(f"[{time.time() - t0:.1f}s] Indice treinado.")
    del amostra

    total_vetores = 0
    with tqdm(total=total_esperado, desc="Adicionando vetores ao indice", unit="vec") as pbar:
        for shard_path in shard_paths:
            shard = _carregar_shard(shard_path)
            n = shard.shape[0]
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                lote = shard[start:end].float().numpy()
                index.add(lote)
                total_vetores += len(lote)
                pbar.update(len(lote))
            del shard

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    faiss.write_index(index, OUTPUT_PATH)
    logger.info(f"[{time.time() - t0:.1f}s] Indice salvo em {OUTPUT_PATH} ({total_vetores} vetores).")


if __name__ == "__main__":
    main()
