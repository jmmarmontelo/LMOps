import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

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
MINI_BASE_DIR = Path(os.getenv("MINI_BASE_DIR", str(REPO_ROOT / "data" / "mini")))
MINI_CORPUS_DIR = MINI_BASE_DIR / "corpus"
MINI_QUESTIONS_DIR = MINI_BASE_DIR / "questions"
MINI_E5_INDEX_DIR = MINI_BASE_DIR / "e5-large-index"
E5_SERVER_LOG = REPO_ROOT / "e5_server_mini.log"

SEM_RESPOSTA = "no relevant information found"

# Ablação: fonte dos documentos usados no prompt de resposta final.
# "dataset" (default) = context_doc_ids do dataset (retrieval estático, top-100 E5 pra query
# original, pré-computado no corag/multihopqa). "chain" = fusão por RRF dos doc_ids que a própria
# chain recuperou em cada hop.
DOC_SOURCE = os.getenv("DOC_SOURCE", "dataset")
NUM_CONTEXTS = int(os.getenv("NUM_CONTEXTS", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Poucas threads por padrão (teste local); pode subir bastante ao rodar no servidor
# (ex.: NUM_THREADS=32, mesmo default do pipeline oficial em src/config.py).
NUM_THREADS = int(os.getenv("NUM_THREADS", "2"))


def fundir_doc_ids_por_rrf(listas_de_doc_ids: List[List[str]], k: int = RRF_K) -> List[str]:
    """Reciprocal Rank Fusion: funde múltiplas listas de doc_ids já ordenadas por score
    (uma por hop) numa única lista ordenada por soma de 1/(k + rank)."""
    scores: Dict[str, float] = {}
    for lista in listas_de_doc_ids:
        for rank, doc_id in enumerate(lista):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _porta_aberta(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0


def iniciar_servidor_e5(
        index_dir: Path, corpus_dir: Optional[Path] = None, host: str = "localhost", port: int = 8090, timeout: int = 600,
) -> Optional[subprocess.Popen]:
    if _porta_aberta(host, port):
        print(f"Servidor E5 ja rodando em {host}:{port}.")
        return None

    print(f"Iniciando servidor E5 (mini-corpus) em {host}:{port}, log em {E5_SERVER_LOG}...")
    env = os.environ.copy()
    env["INDEX_DIR"] = str(index_dir)
    # CORPUS_DIR so setado quando um corpus local e passado -- sem isso, load_corpus()
    # (data_utils.py) cai no default de baixar o corpus completo do HuggingFace.
    if corpus_dir is not None:
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


def parar_servidor_e5(processo: Optional[subprocess.Popen], timeout: int = 10) -> None:
    """Derruba um servidor E5 iniciado por `iniciar_servidor_e5` (usado para trocar de
    mini-corpus entre chunks). Nao faz nada se `processo` for None (porta ja estava
    ocupada por um servidor externo, que nao e nosso pra matar)."""
    if processo is None:
        return
    processo.terminate()
    try:
        processo.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        processo.kill()
        processo.wait()


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


def _processar_exemplo(
        idx: int, exemplo: Dict, total: int, corag_agent: CoRagAgent, corpus: Dataset,
        args: SimpleNamespace, max_path_length: int, estrategia: str, n: int, temperature: float,
) -> Dict:
    logger.info(f"[{idx + 1}/{total}] Pergunta: {exemplo['query']}")

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
            logger.info(f"  [{idx + 1}] Candidato {amostra + 1}/{num_amostras}: {penalidade} hop(s) sem informação relevante")

    idx_escolhido = selecionar_por_penalizacao(caminhos) if estrategia == "best_of_n" else 0
    path = caminhos[idx_escolhido]

    doc_ids_fonte = fundir_doc_ids_por_rrf(path.past_doc_ids) if DOC_SOURCE == "chain" \
        else exemplo["context_doc_ids"]
    documentos = format_documents_for_final_answer(
        args=args,
        context_doc_ids=doc_ids_fonte,
        tokenizer=corag_agent.tokenizer,
        corpus=corpus,
        # mesmo lock usado internamente por CoRagAgent._truncate_long_messages — garante que as
        # duas chamadas de batch_truncate (aqui e dentro do agent) nunca rodem em paralelo sobre
        # o mesmo tokenizer, mesmo com múltiplas threads processando exemplos ao mesmo tempo.
        lock=corag_agent.lock,
    )

    resposta = corag_agent.generate_final_answer(
        corag_sample=path,
        task_desc=exemplo["task_desc"],
        documents=documentos,
        max_message_length=3072,
        temperature=0.,
        max_tokens=128,
    )

    for hop, (subquery, subanswer) in enumerate(zip(path.past_subqueries, path.past_subanswers)):
        logger.info(f"  [{idx + 1}] Hop {hop + 1} subquery: {subquery}")
        logger.info(f"  [{idx + 1}] Hop {hop + 1} subanswer: {subanswer}")
    logger.info(f"  [{idx + 1}] Resposta final (escolhida): {resposta}")
    logger.info(f"  [{idx + 1}] Resposta esperada: {exemplo['answers']}")

    return {
        "query": exemplo["query"],
        "answers": exemplo["answers"],
        "estrategia": estrategia,
        "n": n,
        "max_path_length": max_path_length,
        "subqueries": path.past_subqueries,
        "subanswers": path.past_subanswers,
        "doc_ids": path.past_doc_ids,
        "penalizacoes": [contar_hops_sem_resposta(c) for c in caminhos],
        "doc_source": DOC_SOURCE,
        "prediction": resposta,
    }


def executar_rag(
        dataset: Dataset, corpus: Dataset, base_url: str, api_key: str, model: str,
        tokenizer_name_or_path: str = "corag/CoRAG-Llama3.1-8B-MultihopQA",
        max_path_length: int = 3,
        log_path: str = "data/rag_log.jsonl",
        estrategia: str = "greedy",
        n: int = 4,
        temperature: float = 0.7,
        num_threads: int = NUM_THREADS,
):
    vllm_client = VllmClient(model=model, api_key=api_key)
    # VllmClient monta a base_url fixando o esquema http:// (host/port); como o endpoint
    # remoto usa https, substituímos o client OpenAI interno pela base_url completa e correta.
    vllm_client.client = OpenAI(base_url=base_url, api_key=api_key)

    # contorna o get_vllm_model_id() hardcoded em localhost:8000 dentro de CoRagAgent.__init__
    # (model é só o apelido usado nas chamadas da API; o tokenizer precisa do repo real no HF Hub)
    corag_agent_module.get_vllm_model_id = lambda *args, **kwargs: tokenizer_name_or_path

    corag_agent = CoRagAgent(vllm_client=vllm_client, corpus=corpus)

    args = SimpleNamespace(num_contexts=NUM_CONTEXTS, max_len=3072, context_placement="backward")
    total = len(dataset)

    # Mesmo padrão do pipeline oficial (src/inference/run_inference.py): ThreadPoolExecutor.map
    # preserva a ordem de entrada na saída, então "resultados" fica alinhado com "dataset" mesmo
    # rodando em paralelo. O client OpenAI/httpx e o Dataset (Arrow) já são seguros para uso
    # concorrente; o único ponto que precisa de lock é o tokenizer (via corag_agent.lock acima).
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        resultados: List[Dict] = list(executor.map(
            lambda par: _processar_exemplo(
                par[0], par[1], total, corag_agent, corpus, args,
                max_path_length, estrategia, n, temperature,
            ),
            enumerate(dataset),
        ))

    # Escreve o log só depois de coletar tudo — evita múltiplas threads escrevendo no mesmo
    # arquivo ao mesmo tempo (o que podia embaralhar/corromper linhas do JSONL).
    with open(log_path, "w") as log_file:
        for resultado in resultados:
            log_file.write(json.dumps(resultado, ensure_ascii=False) + "\n")

    return resultados


if __name__ == "__main__":
    estrategia = "best_of_n"
    n = 4
    max_path_length = 6
    log_dir = "data"

    print(f"doc_source={DOC_SOURCE} num_contexts={NUM_CONTEXTS} rrf_k={RRF_K} num_threads={NUM_THREADS}")

    load_dotenv()

    iniciar_servidor_e5(MINI_E5_INDEX_DIR, MINI_CORPUS_DIR)

    corpus = load_corpus(corpus_dir=str(MINI_CORPUS_DIR))

    tasks = os.getenv("TASKS", ",".join(TASK_SPLITS)).split(",")

    metricas_por_task: dict = {}
    for task in tasks:
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
