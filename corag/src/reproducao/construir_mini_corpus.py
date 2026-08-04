import bisect
import json
import os
import random
from typing import Dict, List, Set, Tuple

import torch
from datasets import Dataset

from data_utils import load_corpus
from search.e5_searcher import _get_all_shards_path
from reproducao.best_of_n import TASK_SPLITS, criar_dataset_benchmark

TASKS = list(TASK_SPLITS)

N_PERGUNTAS = 30
N_DISTRATORES_POR_BENCHMARK = 1000
SEED = 42

INDEX_DIR_ORIGINAL = "data/e5-large-index"
CORPUS_DIR_SAIDA = "data/mini/corpus"
INDEX_DIR_SAIDA = "data/mini/e5-large-index"
QUESTIONS_DIR_SAIDA = "data/mini/questions"
ID_MAP_PATH = "data/mini/id_map.json"


def amostrar_datasets(n: int = N_PERGUNTAS, seed: int = SEED) -> Dict[str, Dataset]:
    return {
        task: criar_dataset_benchmark(task, n=n, split=TASK_SPLITS[task], aleatorio=True, seed=seed)
        for task in TASKS
    }


def coletar_ids_contexto(mini_datasets: Dict[str, Dataset]) -> Set[int]:
    ids: Set[int] = set()
    for dataset in mini_datasets.values():
        for exemplo in dataset:
            ids.update(int(doc_id) for doc_id in exemplo["context_doc_ids"])
    return ids


def amostrar_distratores(
        ids_existentes: Set[int], corpus_len: int,
        n_por_benchmark: int = N_DISTRATORES_POR_BENCHMARK, seed: int = SEED,
) -> Set[int]:
    rng = random.Random(seed)
    selecionados = set(ids_existentes)
    distratores: Set[int] = set()
    for _ in TASKS:
        adicionados = 0
        while adicionados < n_por_benchmark:
            candidato = rng.randrange(corpus_len)
            if candidato not in selecionados:
                selecionados.add(candidato)
                distratores.add(candidato)
                adicionados += 1
    return distratores


def construir_mapa_ids(ids: Set[int]) -> Dict[int, int]:
    ordenados = sorted(ids)
    return {original: i for i, original in enumerate(ordenados)}


def construir_tabela_shards(index_dir: str) -> List[Tuple[str, int, int]]:
    tabela = []
    offset = 0
    for path in _get_all_shards_path(index_dir):
        tensor = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
        n_linhas = tensor.shape[0]
        tabela.append((path, offset, offset + n_linhas))
        offset += n_linhas
        del tensor
    return tabela


def extrair_embeddings(mapa_ids: Dict[int, int], tabela_shards: List[Tuple[str, int, int]]) -> torch.Tensor:
    ids_originais_ordenados = sorted(mapa_ids, key=mapa_ids.get)
    limites = [inicio for _, inicio, _ in tabela_shards] + [tabela_shards[-1][2]]
    saida = torch.empty((len(ids_originais_ordenados), 1024), dtype=torch.float16)

    por_shard: Dict[int, List[int]] = {}
    for id_original in ids_originais_ordenados:
        shard_idx = bisect.bisect_right(limites, id_original) - 1
        por_shard.setdefault(shard_idx, []).append(id_original)

    for shard_idx, ids_do_shard in por_shard.items():
        path, inicio, _ = tabela_shards[shard_idx]
        tensor = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
        indices_locais = torch.tensor([id_original - inicio for id_original in ids_do_shard])
        linhas = tensor[indices_locais]
        for linha, id_original in zip(linhas, ids_do_shard):
            saida[mapa_ids[id_original]] = linha
        del tensor

    return saida


def construir_mini_corpus_e_indice(mapa_ids: Dict[int, int]) -> None:
    corpus = load_corpus()
    ids_originais_ordenados = sorted(mapa_ids, key=mapa_ids.get)

    mini_corpus = corpus.select(ids_originais_ordenados)
    mini_corpus = mini_corpus.add_column("orig_doc_id", ids_originais_ordenados)
    mini_corpus.save_to_disk(CORPUS_DIR_SAIDA)

    tabela = construir_tabela_shards(INDEX_DIR_ORIGINAL)
    mini_embeddings = extrair_embeddings(mapa_ids, tabela)
    os.makedirs(INDEX_DIR_SAIDA, exist_ok=True)
    torch.save(mini_embeddings, os.path.join(INDEX_DIR_SAIDA, "e5-large-shard-0.pt"))


def remapear_e_salvar_datasets(mini_datasets: Dict[str, Dataset], mapa_ids: Dict[int, int]) -> None:
    for task, dataset in mini_datasets.items():
        dataset_remapeado = dataset.map(lambda exemplo: {
            "context_doc_ids": [str(mapa_ids[int(doc_id)]) for doc_id in exemplo["context_doc_ids"]]
        })
        dataset_remapeado.save_to_disk(os.path.join(QUESTIONS_DIR_SAIDA, task))


if __name__ == "__main__":
    print("Amostrando 30 perguntas por benchmark...")
    mini_datasets = amostrar_datasets()

    print("Coletando context_doc_ids...")
    ids_contexto = coletar_ids_contexto(mini_datasets)
    print(f"  {len(ids_contexto)} ids de contexto unicos")

    corpus_len = len(load_corpus())
    tabela_shards = construir_tabela_shards(INDEX_DIR_ORIGINAL)
    assert tabela_shards[-1][2] == corpus_len, (
        f"soma das linhas dos shards ({tabela_shards[-1][2]}) != linhas do corpus ({corpus_len})"
    )

    print("Sorteando distratores...")
    distratores = amostrar_distratores(ids_contexto, corpus_len=corpus_len)
    print(f"  {len(distratores)} ids distratores")

    mapa_ids = construir_mapa_ids(ids_contexto | distratores)
    print(f"Mini-corpus final: {len(mapa_ids)} documentos")

    print("Construindo mini-corpus e mini-indice...")
    construir_mini_corpus_e_indice(mapa_ids)

    print("Remapeando context_doc_ids e salvando mini-datasets...")
    remapear_e_salvar_datasets(mini_datasets, mapa_ids)

    os.makedirs(os.path.dirname(ID_MAP_PATH), exist_ok=True)
    with open(ID_MAP_PATH, "w") as f:
        json.dump({str(k): v for k, v in mapa_ids.items()}, f)

    print("Concluido. Artefatos em data/mini/")
