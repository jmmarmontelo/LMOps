import json
import os
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from openai import OpenAI

from vllm_client import VllmClient
import agent.corag_agent as corag_agent_module
from agent import CoRagAgent
from agent.agent_utils import RagPath
from data_utils import load_corpus, format_documents_for_final_answer
from inference.metrics import compute_metrics_dict
from logger_config import logger


TASK_SPLITS = {
    "hotpotqa": "validation",
    "2wikimultihopqa": "validation",
    "musique": "validation",
    "bamboogle": "test",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI_CORPUS_DIR = REPO_ROOT / "data" / "mini" / "corpus"
MINI_QUESTIONS_DIR = REPO_ROOT / "data" / "mini" / "questions"
MINI_E5_INDEX_DIR = REPO_ROOT / "data" / "mini" / "e5-large-index"
E5_SERVER_LOG = REPO_ROOT / "e5_server_mini.log"

SEM_RESPOSTA = "no relevant information found"


def _porta_aberta(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0


def iniciar_servidor_e5(
        index_dir: Path, corpus_dir: Path, host: str = "localhost", port: int = 8090, timeout: int = 600,
) -> Optional[subprocess.Popen]:
    if _porta_aberta(host, port):
        print(f"Servidor E5 ja rodando em {host}:{port}.")
        return None

    print(f"Iniciando servidor E5 (mini-corpus) em {host}:{port}, log em {E5_SERVER_LOG}...")
    env = os.environ.copy()
    env["INDEX_DIR"] = str(index_dir)
    env["CORPUS_DIR"] = str(corpus_dir)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    log_file = open(E5_SERVER_LOG, "w")
    processo = subprocess.Popen(
        ["uvicorn", "src.search.start_e5_server_main:app", "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    elapsed = 0
    while not _porta_aberta(host, port):
        if processo.poll() is not None:
            raise RuntimeError(f"Servidor E5 encerrou antes de subir; veja {E5_SERVER_LOG}")
        time.sleep(2)
        elapsed += 2
        if elapsed >= timeout:
            raise TimeoutError(f"Servidor E5 nao respondeu em {timeout}s; veja {E5_SERVER_LOG}")

    print("Servidor E5 pronto (o modelo de encoding ainda pode estar carregando em segundo plano).")
    return processo


def carregar_perguntas(task: str) -> Dataset:
    return Dataset.load_from_disk(str(MINI_QUESTIONS_DIR / task))


def criar_dataset_benchmark(
        task: str, n: int = 30, split: Optional[str] = None,
        aleatorio: bool = True, seed: int = 42,
) -> Dataset:
    mini_dir = os.getenv("MINI_DATASET_DIR")
    if mini_dir:
        return Dataset.load_from_disk(os.path.join(mini_dir, task))

    split = split or TASK_SPLITS[task]
    dataset = load_dataset("corag/multihopqa", task, split=split)
    if aleatorio:
        dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(n, len(dataset))))
    dataset = dataset.add_column("task_desc", ["answer multi-hop questions"] * len(dataset))
    return dataset


def criar_dataset_hotpotqa(n: int = 5, split: str = "validation", aleatorio: bool = False, seed: int = 42) -> Dataset:
    return criar_dataset_benchmark("hotpotqa", n=n, split=split, aleatorio=aleatorio, seed=seed)


def calcular_metricas(resultados: list) -> dict:
    labels = [r["answers"] for r in resultados]
    preds = [r["prediction"] for r in resultados]
    metricas = compute_metrics_dict(labels=labels, preds=preds, eval_metrics="em_and_f1")
    logger.info(f"Métricas: {metricas}")
    return metricas


def contar_hops_sem_resposta(path: RagPath) -> int:
    return sum(1 for subanswer in path.past_subanswers if subanswer.strip().lower() == SEM_RESPOSTA)


def selecionar_por_penalizacao(caminhos: List[RagPath]) -> int:
    """Índice da cadeia com menos hops "No relevant information found" (empate resolvido pela
    1ª ocorrência, ou seja, a amostra greedy/temperature=0).

    Substitui o scoring por prompt_logprobs do best_of_n original (exclusivo do vLLM, não
    suportado pelo backend usado aqui) por contagem direta de hops sem informação relevante —
    mesma convenção de comparação de texto já usada em dynamic_chain.py (SEM_RESPOSTA).
    """
    penalidades = [contar_hops_sem_resposta(c) for c in caminhos]
    return penalidades.index(min(penalidades))


def executar_rag(
        dataset: Dataset, corpus: Dataset, base_url: str, api_key: str, model: str,
        tokenizer_name_or_path: str = "corag/CoRAG-Llama3.1-8B-MultihopQA",
        max_path_length: int = 3,
        log_path: str = "data/rag_log.jsonl",
        estrategia: str = "greedy",
        n: int = 4,
        temperature: float = 0.7,
):
    vllm_client = VllmClient(model=model, api_key=api_key)
    # VllmClient monta a base_url fixando o esquema http:// (host/port); como o endpoint
    # remoto usa https, substituímos o client OpenAI interno pela base_url completa e correta.
    vllm_client.client = OpenAI(base_url=base_url, api_key=api_key)

    # contorna o get_vllm_model_id() hardcoded em localhost:8000 dentro de CoRagAgent.__init__
    # (model é só o apelido usado nas chamadas da API; o tokenizer precisa do repo real no HF Hub)
    corag_agent_module.get_vllm_model_id = lambda *args, **kwargs: tokenizer_name_or_path

    corag_agent = CoRagAgent(vllm_client=vllm_client, corpus=corpus)

    args = SimpleNamespace(num_contexts=5, max_len=3072, context_placement="backward")
    resultados = []
    with open(log_path, "w") as log_file:
        for idx, exemplo in enumerate(dataset):
            logger.info(f"[{idx + 1}/{len(dataset)}] Pergunta: {exemplo['query']}")

            documentos = format_documents_for_final_answer(
                args=args,
                context_doc_ids=exemplo["context_doc_ids"],
                tokenizer=corag_agent.tokenizer,
                corpus=corpus,
            )

            num_amostras = n if estrategia == "best_of_n" else 1
            caminhos: List[RagPath] = []
            for amostra in range(num_amostras):
                path: RagPath = corag_agent.sample_path(
                    query=exemplo["query"],
                    task_desc=exemplo["task_desc"],
                    max_path_length=max_path_length,
                    temperature=0. if amostra == 0 else temperature,
                    max_tokens=64,
                )
                caminhos.append(path)
                if estrategia == "best_of_n":
                    penalidade = contar_hops_sem_resposta(path)
                    logger.info(f"  Candidato {amostra + 1}/{num_amostras}: {penalidade} hop(s) sem informação relevante")

            idx_escolhido = selecionar_por_penalizacao(caminhos) if estrategia == "best_of_n" else 0
            path = caminhos[idx_escolhido]

            resposta = corag_agent.generate_final_answer(
                corag_sample=path,
                task_desc=exemplo["task_desc"],
                documents=documentos,
                max_message_length=3072,
                temperature=0.,
                max_tokens=128,
            )

            for hop, (subquery, subanswer) in enumerate(zip(path.past_subqueries, path.past_subanswers)):
                logger.info(f"  Hop {hop + 1} subquery: {subquery}")
                logger.info(f"  Hop {hop + 1} subanswer: {subanswer}")
            logger.info(f"  Resposta final (escolhida): {resposta}")
            logger.info(f"  Resposta esperada: {exemplo['answers']}")

            resultado = {
                "query": exemplo["query"],
                "answers": exemplo["answers"],
                "subqueries": path.past_subqueries,
                "subanswers": path.past_subanswers,
                "doc_ids": path.past_doc_ids,
                "penalizacoes": [contar_hops_sem_resposta(c) for c in caminhos],
                "prediction": resposta,
            }
            resultados.append(resultado)
            log_file.write(json.dumps(resultado, ensure_ascii=False) + "\n")
            log_file.flush()

    return resultados


if __name__ == "__main__":
    estrategia = "best_of_n"
    n = 2
    max_path_length = 6
    log_dir = "data"

    load_dotenv()

    iniciar_servidor_e5(MINI_E5_INDEX_DIR, MINI_CORPUS_DIR)

    corpus = load_corpus(corpus_dir=str(MINI_CORPUS_DIR))

    metricas_por_task: dict = {}
    for task in TASK_SPLITS:
        dataset = carregar_perguntas(task)
        resultados = executar_rag(
            dataset,
            corpus,
            base_url=os.environ["BASE_URL"],
            api_key=os.environ["API_KEY"],
            model=os.getenv("MODEL_NAME", "corag-8b"),
            estrategia=estrategia,
            n=n,
            max_path_length=max_path_length,
            log_path=os.path.join(log_dir, f"rag_log_{task}.jsonl"),
        )
        for resultado in resultados:
            print(resultado)

        print(f"--- Métricas [{task}] ---")
        metricas_por_task[task] = calcular_metricas(resultados)

    print("=== Métricas finais (por task) ===")
    for task, metricas in metricas_por_task.items():
        print(f"{task}: {metricas}")
