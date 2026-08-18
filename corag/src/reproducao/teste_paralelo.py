"""
Teste de execucao em paralelo (best_of_n ou dynamic_chain) sobre o mini dataset de
2wikimultihopqa (100 perguntas). Usa poucas threads por padrao (NUM_THREADS=2) para rodar
localmente sem sobrecarregar o backend remoto; ao rodar no servidor, basta exportar NUM_THREADS
com um valor maior (ex.: 16-32) antes de chamar este mesmo script - nenhum outro ajuste e
necessario.

Uso:
    PYTHONPATH=src python src/reproducao/teste_paralelo.py                              # best_of_n, 100 perguntas, 2 threads
    PYTHONPATH=src python src/reproducao/teste_paralelo.py --estrategia dynamic_chain    # dynamic_chain
    PYTHONPATH=src python src/reproducao/teste_paralelo.py --n-perguntas 10              # so as 10 primeiras (smoke test rapido)
    NUM_THREADS=16 PYTHONPATH=src python src/reproducao/teste_paralelo.py                # mais threads (servidor)
"""
import argparse
import os
import time

os.environ.setdefault("MINI_BASE_DIR", "data/mini_2wiki100")

from dotenv import load_dotenv
from data_utils import load_corpus
from reproducao.best_of_n import (
    iniciar_servidor_e5, carregar_perguntas, executar_rag, calcular_metricas,
    MINI_E5_INDEX_DIR, MINI_CORPUS_DIR, NUM_THREADS, DOC_SOURCE, NUM_CONTEXTS, RRF_K,
)
from reproducao.dynamic_chain import _montar_agente, executar_dynamic_chain_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estrategia", choices=["best_of_n", "dynamic_chain"], default="best_of_n")
    parser.add_argument("--n-perguntas", type=int, default=None, help="limita a quantidade de perguntas (default: todas as 100)")
    parser.add_argument("--n-chains", type=int, default=4, help="N (numero de cadeias candidatas)")
    parser.add_argument("--max-path-length", type=int, default=6, help="L, tamanho maximo de cada cadeia")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Teste paralelo {args.estrategia}: num_threads={NUM_THREADS} doc_source={DOC_SOURCE} "
          f"num_contexts={NUM_CONTEXTS} rrf_k={RRF_K}")

    load_dotenv()
    iniciar_servidor_e5(MINI_E5_INDEX_DIR, MINI_CORPUS_DIR)

    corpus = load_corpus(corpus_dir=str(MINI_CORPUS_DIR))
    dataset = carregar_perguntas("2wikimultihopqa")
    if args.n_perguntas:
        dataset = dataset.select(range(min(args.n_perguntas, len(dataset))))
    print(f"{len(dataset)} pergunta(s) carregada(s).")

    inicio = time.time()
    if args.estrategia == "best_of_n":
        resultados = executar_rag(
            dataset, corpus,
            base_url=os.environ["BASE_URL"],
            api_key=os.environ["API_KEY"],
            model=os.getenv("MODEL_NAME", "corag-8b"),
            estrategia="best_of_n",
            n=args.n_chains,
            max_path_length=args.max_path_length,
            log_path="data/rag_log_teste_paralelo.jsonl",
            num_threads=NUM_THREADS,
        )
    else:
        corag_agent = _montar_agente(
            model=os.getenv("MODEL_NAME", "corag-8b"),
            api_key=os.environ["API_KEY"],
            base_url=os.environ["BASE_URL"],
            corpus=corpus,
        )
        resultados = executar_dynamic_chain_dataset(
            dataset, corag_agent,
            n=args.n_chains,
            max_path_length=args.max_path_length,
            log_path="data/rag_log_teste_paralelo.jsonl",
            num_threads=NUM_THREADS,
        )
    duracao = time.time() - inicio

    print(f"\nConcluido em {duracao:.1f}s ({duracao / 60:.1f} min) para {len(resultados)} pergunta(s).")
    print(f"Media por pergunta: {duracao / len(resultados):.2f}s")
    print(f"Estimativa para 100 perguntas nesse ritmo: {duracao / len(resultados) * 100 / 60:.1f} min")
    print(calcular_metricas(resultados))
