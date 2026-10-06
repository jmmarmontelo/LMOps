"""Constroi um mini-corpus (gold passages + top-100 E5 da pergunta original + distratores
aleatorios) e o mini-indice E5 correspondente, alinhados linha a linha, a partir do corpus
KILT completo. As gold passages vem dos dev oficiais de cada benchmark e sao mapeadas pra
passagens do KILT por titulo + sobreposicao de tokens."""

import argparse
import bisect
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download

from data_utils import load_corpus
from search.e5_searcher import _get_all_shards_path
from reproducao.best_of_n import TASK_SPLITS, criar_dataset_benchmark

N_PERGUNTAS = 50
# Usado pelo rodar_split_em_chunks.py (distratores por chunk) -- nao mexer sem olhar la.
N_DISTRATORES_POR_BENCHMARK = 1000
N_DISTRATORES_GOLD = 50000
SEED = 42

INDEX_DIR_ORIGINAL = "data/e5-large-index"
OUTPUT_DIR_PADRAO = "data/mini_gold"
# Arquivos < 50 MB pra caber no GitHub (limite duro de 100 MB, aviso acima de 50 MB):
# 24000 linhas x 1024 x fp16 ~= 49 MB por shard do indice.
LINHAS_POR_SHARD_MINI = 24000
MAX_SHARD_SIZE_CORPUS_MINI = "45MB"

# Dev oficiais de onde saem as gold passages (o corag/multihopqa nao traz essa informacao).
# Bamboogle nao tem gold, entao fica de fora.
FONTES_GOLD = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet"),
    "2wikimultihopqa": ("framolfese/2WikiMultihopQA", "data/validation-00000-of-00001.parquet"),
    "musique": ("bdsaglam/musique", "musique_ans_v1.0_dev.jsonl"),
}
TASKS_GOLD = list(FONTES_GOLD)


def amostrar_datasets(tasks: List[str], n: int = N_PERGUNTAS, seed: int = SEED) -> Dict[str, Dataset]:
    return {
        task: criar_dataset_benchmark(task, n=n, split=TASK_SPLITS[task], aleatorio=True, seed=seed)
        for task in tasks
    }


def fatiar_dataset_sequencial(task: str, start_idx: int, end_idx: int, split: str = None) -> Dataset:
    """Fatia sequencial (nao aleatoria) do split real de `task`, para processar o dataset
    inteiro em partes (ex.: chunks de 1000 perguntas) em vez de uma amostra aleatoria."""
    from datasets import load_dataset

    split = split or TASK_SPLITS[task]
    dataset = load_dataset("corag/multihopqa", task, split=split)
    end_idx = min(end_idx, len(dataset))
    dataset = dataset.select(range(start_idx, end_idx))
    dataset = dataset.add_column("task_desc", ["answer multi-hop questions"] * len(dataset))
    return dataset


def coletar_ids_contexto(mini_datasets: Dict[str, Dataset]) -> Set[int]:
    ids: Set[int] = set()
    for dataset in mini_datasets.values():
        for exemplo in dataset:
            ids.update(int(doc_id) for doc_id in exemplo["context_doc_ids"])
    return ids


def _normalizar(texto: str) -> str:
    return " ".join(re.findall(r"\w+", texto.lower()))


def _tokens(texto: str) -> Set[str]:
    return set(re.findall(r"\w+", texto.lower()))


def carregar_golds_originais(task: str) -> Dict[str, List[Tuple[str, str]]]:
    """pergunta normalizada -> [(titulo, texto gold)] a partir do dev oficial de `task`.
    HotpotQA/2Wiki: texto gold = sentencas de suporte daquele titulo. MuSiQue: paragrafo
    inteiro marcado is_supporting. Pergunta repetida no dev tem as golds unidas."""
    repo, arquivo = FONTES_GOLD[task]
    path = hf_hub_download(repo, arquivo, repo_type="dataset")
    golds: Dict[str, Dict[str, str]] = defaultdict(dict)

    if task == "musique":
        with open(path, encoding="utf-8") as f:
            for linha in f:
                exemplo = json.loads(linha)
                for paragrafo in exemplo["paragraphs"]:
                    if paragrafo["is_supporting"]:
                        golds[_normalizar(exemplo["question"])][paragrafo["title"]] = paragrafo["paragraph_text"]
    else:
        for exemplo in pd.read_parquet(path).to_dict("records"):
            sentencas_por_titulo = dict(zip(exemplo["context"]["title"], exemplo["context"]["sentences"]))
            sent_ids_por_titulo: Dict[str, List[int]] = defaultdict(list)
            for titulo, sent_id in zip(exemplo["supporting_facts"]["title"], exemplo["supporting_facts"]["sent_id"]):
                sent_ids_por_titulo[titulo].append(int(sent_id))
            for titulo, sent_ids in sent_ids_por_titulo.items():
                sentencas = [s.strip() for s in sentencas_por_titulo.get(titulo, [])]
                texto = " ".join(sentencas[i] for i in sent_ids if i < len(sentencas)) or " ".join(sentencas)
                golds[_normalizar(exemplo["question"])][titulo] = texto

    return {pergunta: list(por_titulo.items()) for pergunta, por_titulo in golds.items()}


def _indices_por_titulo(
        coluna_titulos: pa.ChunkedArray, titulos: List[str], ignorar_caixa: bool = False,
) -> Dict[str, List[int]]:
    """Indices das linhas cujo titulo esta em `titulos`, pedaco a pedaco da coluna (sem
    materializar os ~36M titulos de uma vez -- importante com pouca RAM)."""
    value_set = pa.array(titulos, type=coluna_titulos.type)
    por_titulo: Dict[str, List[int]] = defaultdict(list)
    offset = 0
    for pedaco in coluna_titulos.chunks:
        comparado = pc.utf8_lower(pedaco) if ignorar_caixa else pedaco
        locais = pc.indices_nonzero(pc.is_in(comparado, value_set=value_set))
        for idx_local, titulo in zip(locais.to_pylist(), pc.take(comparado, locais).to_pylist()):
            por_titulo[titulo].append(offset + idx_local)
        offset += len(pedaco)
    return por_titulo


def mapear_golds_para_kilt(
        corpus: Dataset, pares: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], Tuple[Optional[int], float]]:
    """(titulo, texto gold) -> (id no KILT, score). Candidatos = passagens com o mesmo titulo
    (exato; se nao achar, ignorando caixa); escolhe a de maior recall dos tokens do texto
    gold. Titulo inexistente no KILT -> (None, 0.0)."""
    coluna_titulos = corpus.data.column("title")
    titulos = sorted({titulo for titulo, _ in pares})
    candidatos = _indices_por_titulo(coluna_titulos, titulos)

    faltando = [t for t in titulos if t not in candidatos]
    if faltando:
        por_titulo_minusculo = _indices_por_titulo(coluna_titulos, [t.lower() for t in faltando], ignorar_caixa=True)
        for titulo in faltando:
            if titulo.lower() in por_titulo_minusculo:
                candidatos[titulo] = por_titulo_minusculo[titulo.lower()]

    todos_ids = sorted({idx for ids in candidatos.values() for idx in ids})
    conteudos = dict(zip(todos_ids, corpus.select(todos_ids)["contents"]))

    mapeamento: Dict[Tuple[str, str], Tuple[Optional[int], float]] = {}
    for titulo, texto in pares:
        tokens_gold = _tokens(texto)
        melhor: Tuple[Optional[int], float] = (None, 0.0)
        for idx in candidatos.get(titulo, []):
            score = len(tokens_gold & _tokens(conteudos[idx])) / max(len(tokens_gold), 1)
            if melhor[0] is None or score > melhor[1]:
                melhor = (idx, score)
        mapeamento[(titulo, texto)] = melhor
    return mapeamento


def anexar_golds(
        mini_datasets: Dict[str, Dataset], corpus: Dataset,
) -> Tuple[Dict[str, Dataset], Set[int], Dict[str, dict]]:
    """Adiciona `gold_doc_ids` (ids originais do KILT, str como context_doc_ids) e
    `gold_titles` a cada dataset. Devolve os datasets, o conjunto de ids gold e um
    relatorio de cobertura por task."""
    golds_por_task: Dict[str, List[List[Tuple[str, str]]]] = {}
    for task, dataset in mini_datasets.items():
        golds_originais = carregar_golds_originais(task)
        golds_por_task[task] = []
        for exemplo in dataset:
            chave = _normalizar(exemplo["query"])
            if chave not in golds_originais:
                raise KeyError(f"[{task}] pergunta {exemplo['query_id']} nao encontrada no dev oficial: {exemplo['query']!r}")
            golds_por_task[task].append(golds_originais[chave])

    pares = {par for golds in golds_por_task.values() for lista in golds for par in lista}
    mapeamento = mapear_golds_para_kilt(corpus, pares)

    novos_datasets: Dict[str, Dataset] = {}
    ids_gold: Set[int] = set()
    relatorio: Dict[str, dict] = {}
    for task, dataset in mini_datasets.items():
        gold_doc_ids, n_golds, n_mapeadas, n_no_top100, scores, nao_encontradas = [], 0, 0, 0, [], []
        for exemplo, golds in zip(dataset, golds_por_task[task]):
            top100 = {int(doc_id) for doc_id in exemplo["context_doc_ids"]}
            ids_exemplo = []
            for titulo, texto in golds:
                n_golds += 1
                idx, score = mapeamento[(titulo, texto)]
                if idx is None:
                    nao_encontradas.append({"query_id": exemplo["query_id"], "title": titulo})
                    continue
                n_mapeadas += 1
                n_no_top100 += idx in top100
                scores.append(score)
                ids_exemplo.append(str(idx))
            ids_gold.update(int(i) for i in ids_exemplo)
            gold_doc_ids.append(ids_exemplo)
        novos_datasets[task] = (
            dataset
            .add_column("gold_doc_ids", gold_doc_ids)
            .add_column("gold_titles", [[titulo for titulo, _ in golds] for golds in golds_por_task[task]])
        )
        relatorio[task] = {
            "n_perguntas": len(dataset),
            "n_golds": n_golds,
            "n_mapeadas_kilt": n_mapeadas,
            "n_ja_no_top100": n_no_top100,
            "score_medio_sobreposicao": round(sum(scores) / max(len(scores), 1), 4),
            "n_score_abaixo_0.5": sum(s < 0.5 for s in scores),
            "nao_encontradas": nao_encontradas,
        }
    return novos_datasets, ids_gold, relatorio


def amostrar_distratores(
        ids_existentes: Set[int], corpus_len: int, tasks: List[str],
        n_por_benchmark: int = N_DISTRATORES_POR_BENCHMARK, seed: int = SEED,
) -> Set[int]:
    rng = random.Random(seed)
    selecionados = set(ids_existentes)
    distratores: Set[int] = set()
    for _ in tasks:
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


def construir_mini_corpus_e_indice(
        mapa_ids: Dict[int, int], corpus_dir_saida: str, index_dir_saida: str,
        corpus: Optional[Dataset] = None, tabela_shards: Optional[List[Tuple[str, int, int]]] = None,
) -> None:
    corpus = corpus if corpus is not None else load_corpus()
    ids_originais_ordenados = sorted(mapa_ids, key=mapa_ids.get)

    mini_corpus = corpus.select(ids_originais_ordenados)
    mini_corpus = mini_corpus.add_column("orig_doc_id", ids_originais_ordenados)
    mini_corpus.save_to_disk(corpus_dir_saida, max_shard_size=MAX_SHARD_SIZE_CORPUS_MINI)

    tabela = tabela_shards if tabela_shards is not None else construir_tabela_shards(INDEX_DIR_ORIGINAL)
    mini_embeddings = extrair_embeddings(mapa_ids, tabela)
    salvar_indice_em_shards(mini_embeddings, index_dir_saida)


def salvar_indice_em_shards(embeddings: torch.Tensor, index_dir_saida: str) -> None:
    """Fatias contiguas de LINHAS_POR_SHARD_MINI linhas em e5-large-shard-{i}.pt (ordem
    preservada: _get_all_shards_path ordena pelo indice do shard)."""
    os.makedirs(index_dir_saida, exist_ok=True)
    for i, inicio in enumerate(range(0, embeddings.shape[0], LINHAS_POR_SHARD_MINI)):
        # clone(): torch.save de uma view gravaria o storage inteiro do tensor original.
        fatia = embeddings[inicio:inicio + LINHAS_POR_SHARD_MINI].clone()
        torch.save(fatia, os.path.join(index_dir_saida, f"e5-large-shard-{i}.pt"))


def remapear_e_salvar_datasets(
        mini_datasets: Dict[str, Dataset], mapa_ids: Dict[int, int], questions_dir_saida: str,
) -> None:
    for task, dataset in mini_datasets.items():
        # gold_doc_ids so existe nos datasets do mini-corpus com gold (nao nos chunks).
        colunas_ids = [c for c in ("context_doc_ids", "gold_doc_ids") if c in dataset.column_names]
        dataset_remapeado = dataset.map(lambda exemplo: {
            coluna: [str(mapa_ids[int(doc_id)]) for doc_id in exemplo[coluna]] for coluna in colunas_ids
        })
        dataset_remapeado.save_to_disk(os.path.join(questions_dir_saida, task))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=TASKS_GOLD, choices=TASKS_GOLD)
    parser.add_argument("--n-perguntas", type=int, default=N_PERGUNTAS)
    parser.add_argument("--n-distratores", type=int, default=N_DISTRATORES_GOLD)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_PADRAO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir_saida = os.path.join(args.output_dir, "corpus")
    index_dir_saida = os.path.join(args.output_dir, "e5-large-index")
    questions_dir_saida = os.path.join(args.output_dir, "questions")
    id_map_path = os.path.join(args.output_dir, "id_map.json")
    gold_report_path = os.path.join(args.output_dir, "gold_report.json")

    print(f"Amostrando {args.n_perguntas} perguntas por benchmark ({', '.join(args.tasks)})...")
    mini_datasets = amostrar_datasets(args.tasks, n=args.n_perguntas, seed=args.seed)

    print("Coletando context_doc_ids...")
    ids_contexto = coletar_ids_contexto(mini_datasets)
    print(f"  {len(ids_contexto)} ids de contexto unicos")

    corpus = load_corpus()
    corpus_len = len(corpus)
    tabela_shards = construir_tabela_shards(INDEX_DIR_ORIGINAL)
    assert tabela_shards[-1][2] == corpus_len, (
        f"soma das linhas dos shards ({tabela_shards[-1][2]}) != linhas do corpus ({corpus_len})"
    )

    print("Mapeando gold passages dos dev oficiais pro KILT...")
    mini_datasets, ids_gold, relatorio_gold = anexar_golds(mini_datasets, corpus)
    for task, rel in relatorio_gold.items():
        print(f"  {task}: {rel['n_mapeadas_kilt']}/{rel['n_golds']} golds mapeadas, "
              f"{rel['n_ja_no_top100']} ja no top-100, score medio {rel['score_medio_sobreposicao']}")

    print("Sorteando distratores...")
    distratores = amostrar_distratores(
        ids_contexto | ids_gold, corpus_len=corpus_len, tasks=args.tasks,
        n_por_benchmark=args.n_distratores, seed=args.seed,
    )
    print(f"  {len(distratores)} ids distratores")

    mapa_ids = construir_mapa_ids(ids_contexto | ids_gold | distratores)
    print(f"Mini-corpus final: {len(mapa_ids)} documentos")

    print("Construindo mini-corpus e mini-indice...")
    construir_mini_corpus_e_indice(mapa_ids, corpus_dir_saida, index_dir_saida, corpus=corpus, tabela_shards=tabela_shards)

    print("Remapeando context_doc_ids/gold_doc_ids e salvando mini-datasets...")
    remapear_e_salvar_datasets(mini_datasets, mapa_ids, questions_dir_saida)

    os.makedirs(os.path.dirname(id_map_path), exist_ok=True)
    with open(id_map_path, "w") as f:
        json.dump({str(k): v for k, v in mapa_ids.items()}, f)
    with open(gold_report_path, "w", encoding="utf-8") as f:
        json.dump(relatorio_gold, f, ensure_ascii=False, indent=2)

    print(f"Concluido. Artefatos em {args.output_dir}/")


if __name__ == "__main__":
    main()
