

# Objetivo iniciar o CoRag na estratégia Best of N
# E antes do pipeline escolher uma chain para responder a pergunta original
# Pegar todas as chains para fazer uma chain dinamica

import os
import socket
import subprocess
import time

from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

import agent.corag_agent as corag_agent_module

from agent import CoRagAgent
from agent.agent_utils import RagPath
from prompts import get_generate_final_answer_prompt
from reproducao.best_of_n import TASK_SPLITS, criar_dataset_benchmark, calcular_metricas

from vllm_client import VllmClient

tokenizer_name_or_path = "corag/CoRAG-Llama3.1-8B-MultihopQA"

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI_CORPUS_DIR = REPO_ROOT / "data" / "mini" / "corpus"
MINI_QUESTIONS_DIR = REPO_ROOT / "data" / "mini" / "questions"
MINI_E5_INDEX_DIR = REPO_ROOT / "data" / "mini" / "e5-large-index"
E5_SERVER_LOG = REPO_ROOT / "e5_server_mini.log"

# resposta padrao instruida em get_generate_intermediate_answer_prompt (src/prompts.py)
# quando os documentos recuperados nao respondem a subpergunta.
SEM_RESPOSTA = "no relevant information found"


def carregar_mini_corpus() -> Dataset:
    return Dataset.load_from_disk(str(MINI_CORPUS_DIR))


def carregar_mini_perguntas(task: str) -> Dataset:
    # criar_dataset_benchmark (reproducao.py) carrega direto do disco quando
    # MINI_DATASET_DIR esta definido, em vez de baixar/amostrar do HF.
    os.environ["MINI_DATASET_DIR"] = str(MINI_QUESTIONS_DIR)
    return criar_dataset_benchmark(task, split=TASK_SPLITS[task])


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


def _montar_agente(model, api_key, base_url, corpus: Dataset) -> CoRagAgent:
    vllm_client = VllmClient(model=model, api_key=api_key)

    vllm_client.client = OpenAI(base_url=base_url, api_key=api_key)

    corag_agent_module.get_vllm_model_id = lambda *args, **kwargs: tokenizer_name_or_path

    return CoRagAgent(vllm_client=vllm_client, corpus=corpus)


def _imprimir_path(path: RagPath):
    print(f"Query: {path.query}")
    for hop, (subquery, subanswer, doc_ids) in enumerate(
        zip(path.past_subqueries, path.past_subanswers, path.past_doc_ids), start=1
    ):
        print(f"Hop {hop} subquery: {subquery}")
        print(f"Hop {hop} subanswer: {subanswer}")
        print(f"Hop {hop} doc_ids: {doc_ids}")


def selecionar_subperguntas_respondidas(path: RagPath) -> RagPath:
    subqueries: List[str] = []
    subanswers: List[str] = []
    doc_ids: List[List[str]] = []

    for subquery, subanswer, ids in zip(path.past_subqueries, path.past_subanswers, path.past_doc_ids):
        if subanswer.strip().lower() == SEM_RESPOSTA:
            continue

        subqueries.append(subquery)
        subanswers.append(subanswer)
        doc_ids.append(ids)

    return RagPath(
        query=path.query,
        past_subqueries=subqueries,
        past_subanswers=subanswers,
        past_doc_ids=doc_ids,
    )


def montar_chain_final(paths: List[RagPath]) -> RagPath:
    subqueries: List[str] = []
    subanswers: List[str] = []
    doc_ids: List[List[str]] = []

    for path in paths:
        subqueries.extend(path.past_subqueries)
        subanswers.extend(path.past_subanswers)
        doc_ids.extend(path.past_doc_ids)

    chain_bruta = RagPath(
        query=paths[0].query,
        past_subqueries=subqueries,
        past_subanswers=subanswers,
        past_doc_ids=doc_ids,
    )
    return selecionar_subperguntas_respondidas(chain_bruta)


def montar_prompt_final(
        path: RagPath, task_desc: str, documents: Optional[List[str]] = None
) -> List[Dict]:
    return get_generate_final_answer_prompt(
        query=path.query,
        past_subqueries=path.past_subqueries,
        past_subanswers=path.past_subanswers,
        task_desc=task_desc,
        documents=documents,
    )


def gerar_resposta_final(
        corag_agent: CoRagAgent, path: RagPath, task_desc: str, documents: Optional[List[str]] = None,
) -> str:
    messages: List[Dict] = montar_prompt_final(path, task_desc, documents=documents)
    resposta: str = corag_agent.vllm_client.call_chat(messages=messages, temperature=0., max_tokens=128)

    print(f"Resposta final: {resposta}")
    return resposta


def executar_teste(corag_agent: CoRagAgent, query: str, task_desc: str) -> RagPath:
    path: RagPath = corag_agent.sample_path(
        query=query,
        task_desc=task_desc,
        max_path_length=3,
        temperature=0.,
        max_tokens=64,
    )

    _imprimir_path(path)
    return path


def executar_dynamic_chain(corag_agent: CoRagAgent, query: str, task_desc: str, n: int = 4) -> List[RagPath]:
    # CoRagAgent.best_of_n pontua os candidatos pelo logprob de "No relevant information
    # found" (extra_body prompt_logprobs), recurso exclusivo de um servidor vLLM real;
    # o endpoint remoto usado aqui nao retorna esse campo. Por isso so amostramos as N
    # cadeias (mesmo loop de sample_path que best_of_n usa internamente) e imprimimos
    # todas, sem escolher uma — a selecao final e feita depois via
    # selecionar_subperguntas_respondidas/montar_chain_final.
    paths: List[RagPath] = []
    for idx in range(n):
        path: RagPath = corag_agent.sample_path(
            query=query,
            task_desc=task_desc,
            max_path_length=3,
            temperature=0. if idx == 0 else 0.7,
            max_tokens=64,
        )
        paths.append(path)

    for idx, path in enumerate(paths, start=1):
        print(f"--- Candidato {idx}/{n} ---")
        _imprimir_path(path)

    return paths


def executar_pipeline_pergunta(corag_agent: CoRagAgent, exemplo: Dict, n: int = 4) -> Dict:
    query = exemplo["query"]
    task_desc = exemplo["task_desc"]

    paths = executar_dynamic_chain(corag_agent, query, task_desc, n=n)
    chain_final = montar_chain_final(paths)

    print("--- Chain final (subperguntas respondidas) ---")
    _imprimir_path(chain_final)

    resposta = gerar_resposta_final(corag_agent, chain_final, task_desc)

    return {
        "query": query,
        "answers": exemplo["answers"],
        "subqueries": chain_final.past_subqueries,
        "subanswers": chain_final.past_subanswers,
        "doc_ids": chain_final.past_doc_ids,
        "prediction": resposta,
    }


if __name__ == "__main__":
    n = 8 

    load_dotenv()

    iniciar_servidor_e5(MINI_E5_INDEX_DIR, MINI_CORPUS_DIR)

    corpus = carregar_mini_corpus()
    corag_agent = _montar_agente(
        model=os.getenv("MODEL_NAME", "corag-8b"),
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
        corpus=corpus,
    )

    metricas_por_task: Dict[str, Dict] = {}
    for task in TASK_SPLITS:
        perguntas = carregar_mini_perguntas(task)

        resultados = []
        for idx, exemplo in enumerate(perguntas):
            print(f"=== [{task}] pergunta {idx + 1}/{len(perguntas)}: {exemplo['query']} ===")
            resultados.append(executar_pipeline_pergunta(corag_agent, exemplo, n=n))

        print(f"--- Metricas [{task}] ---")
        metricas_por_task[task] = calcular_metricas(resultados)

    print("=== Metricas finais (por task) ===")
    for task, metricas in metricas_por_task.items():
        print(f"{task}: {metricas}")
