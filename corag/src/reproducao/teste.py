import json
import os
from collections import Counter
from types import SimpleNamespace
from typing import List

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


def criar_dataset_hotpotqa(n: int = 5, split: str = "validation", aleatorio: bool = False, seed: int = 42) -> Dataset:
    dataset = load_dataset("corag/multihopqa", "hotpotqa", split=split)
    if aleatorio:
        dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(n, len(dataset))))
    dataset = dataset.add_column("task_desc", ["answer multi-hop questions"] * len(dataset))
    return dataset


def calcular_metricas(resultados: list) -> dict:
    labels = [r["answers"] for r in resultados]
    preds = [r["prediction"] for r in resultados]
    metricas = compute_metrics_dict(labels=labels, preds=preds, eval_metrics="em_and_f1")
    logger.info(f"Métricas: {metricas}")
    return metricas


def selecionar_por_self_consistency(respostas: list) -> int:
    """Índice da resposta mais frequente entre as amostras (empate resolvido pela 1ª ocorrência).

    Substitui o scoring por prompt_logprobs do best_of_n original (exclusivo do vLLM,
    não suportado pelo Ollama) por votação de maioria entre as respostas finais.
    """
    contagem = Counter(r.strip().lower() for r in respostas)
    mais_comum, _ = contagem.most_common(1)[0]
    return next(i for i, r in enumerate(respostas) if r.strip().lower() == mais_comum)


def executar_rag(
        dataset: Dataset, base_url: str, api_key: str, model: str,
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

    corpus = load_corpus()
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

            num_amostras = n if estrategia == "best_of_n_self_consistency" else 1
            caminhos: List[RagPath] = []
            candidatos: List[str] = []
            for amostra in range(num_amostras):
                path: RagPath = corag_agent.sample_path(
                    query=exemplo["query"],
                    task_desc=exemplo["task_desc"],
                    max_path_length=max_path_length,
                    temperature=0. if amostra == 0 else temperature,
                    max_tokens=64,
                )
                resposta_candidata = corag_agent.generate_final_answer(
                    corag_sample=path,
                    task_desc=exemplo["task_desc"],
                    documents=documentos,
                    max_message_length=3072,
                    temperature=0.,
                    max_tokens=128,
                )
                caminhos.append(path)
                candidatos.append(resposta_candidata)
                if estrategia == "best_of_n_self_consistency":
                    logger.info(f"  Candidato {amostra + 1}/{num_amostras}: {resposta_candidata}")

            if estrategia == "best_of_n_self_consistency":
                idx_escolhido = selecionar_por_self_consistency(candidatos)
            else:
                idx_escolhido = 0
            path, resposta = caminhos[idx_escolhido], candidatos[idx_escolhido]

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
                "candidatos": candidatos,
                "prediction": resposta,
            }
            resultados.append(resultado)
            log_file.write(json.dumps(resultado, ensure_ascii=False) + "\n")
            log_file.flush()

    return resultados


if __name__ == "__main__":
    load_dotenv()

    dataset = criar_dataset_hotpotqa(n=30, aleatorio=True)
    resultados = executar_rag(
        dataset,
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
        model=os.getenv("MODEL_NAME", "corag-8b"),
        estrategia="best_of_n_self_consistency",
        n=4,
        temperature=0.7,
    )
    for resultado in resultados:
        print(resultado)

    calcular_metricas(resultados)
