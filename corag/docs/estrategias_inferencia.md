# Estratégias de inferência do CoRAG

Este documento descreve os métodos de `CoRagAgent` (`src/agent/corag_agent.py`)
responsáveis por executar a cadeia de retrieval (subquery → retrieval → subanswer,
repetido por até `max_path_length` hops) e como as **3 estratégias de decodificação**
do repositório — `greedy`, `tree_search` e `best_of_n` — usam esses métodos de formas
diferentes.

## Visão geral do fluxo

Em uma avaliação completa (`src/inference/run_inference.py`), cada exemplo passa por
`_generate_single_example()`, que escolhe a estratégia com base na flag de linha de
comando `--decode_strategy` (`run_inference.py:40-61`):

```python
if args.decode_strategy == 'greedy' or args.max_path_length < 1:
    path = corag_agent.sample_path(...)
elif args.decode_strategy == 'tree_search':
    path = corag_agent.tree_search(...)
elif args.decode_strategy == 'best_of_n':
    path = corag_agent.best_of_n(...)
```

As 3 estratégias produzem o mesmo tipo de saída — um `RagPath`
(`src/agent/agent_utils.py`), com a pergunta original e as listas paralelas
`past_subqueries` / `past_subanswers` / `past_doc_ids` — e depois **compartilham o mesmo
passo final**:

```python
prediction = corag_agent.generate_final_answer(
    corag_sample=path, task_desc=..., documents=..., ...
)
```

Ou seja: as estratégias diferem só em *como* o `RagPath` é construído (guloso, busca em
árvore, ou melhor de N amostras); a geração da resposta final é idêntica nas 3.

O `teste.py` usado nesta sessão (`src/reproducao/teste.py`) usa só a estratégia
**greedy**, chamando `sample_path()` diretamente.

---

## 1. `sample_path()` — estratégia `greedy`

```python
def sample_path(self, query, task_desc, max_path_length=3, max_message_length=4096,
                 temperature=0.7, **kwargs) -> RagPath
```
(`corag_agent.py:39-85`)

Laço guloso e sequencial. A cada iteração:

1. Monta o prompt de geração de subquery (`get_generate_subquery_prompt`), incluindo as
   subqueries/subanswers já geradas até agora.
2. Chama o modelo (`vllm_client.call_chat`) para gerar **uma** subquery.
3. Se a subquery já apareceu antes (`subquery in past_subqueries`), aumenta a temperatura
   para pelo menos `0.7` e tenta de novo (evita ficar preso repetindo a mesma subquery).
4. Caso contrário, recupera documentos via `search_by_http` e gera **uma** subanswer
   (`_get_subanswer_and_doc_ids`).
5. Acumula `(subquery, subanswer, doc_ids)` no caminho e segue para o próximo hop.

O laço termina quando `len(past_subqueries) == max_path_length` **ou** quando o número de
chamadas ao modelo (`num_llm_calls`) atinge `max_num_llm_calls = 4 * (max_path_length -
hops_já_feitos)` — essa segunda condição é uma salvaguarda contra loops infinitos quando o
modelo insiste em repetir a mesma subquery.

Por ser sequencial e sem exploração de alternativas, é a estratégia **mais rápida e mais
barata** em chamadas ao modelo (exatamente `2 × max_path_length` chamadas na prática, salvo
repetições) — mas também a mais suscetível a ficar presa em um caminho de raciocínio ruim
sem chance de correção.

**Também é reaproveitado internamente** por `best_of_n()` (para gerar cada candidato) e por
`_eval_state_without_answer()` do `tree_search()` (para os "rollouts" de avaliação).

---

## 2. `tree_search()` — estratégia `tree_search`

```python
def tree_search(self, query, task_desc, max_path_length=3, max_message_length=4096,
                 temperature=0.7, expand_size=4, num_rollouts=2, beam_size=1,
                 **kwargs) -> RagPath
```
(`corag_agent.py:144-159`, delega para `_search()` em `corag_agent.py:183-228`)

Busca em feixe (beam search) sobre o espaço de subqueries. Para cada um dos
`max_path_length` passos:

1. **Expansão**: para cada candidato do feixe atual, gera `expand_size` subqueries
   diferentes com `sample_subqueries()` (usa `n` amostras do modelo com temperatura,
   deduplicadas).
2. Para cada subquery candidata, recupera documentos e gera uma subanswer
   (`_get_subanswer_and_doc_ids`), criando um novo candidato de caminho para cada uma.
3. **Poda**: se o número de candidatos resultantes passar de `beam_size`, cada um é
   pontuado por `_eval_state_without_answer()` (ver seção 4) e só os `beam_size` de menor
   score sobrevivem para o próximo passo.

No final, retorna o melhor caminho (`candidates[0]`) depois de `max_path_length` passos de
expansão/poda.

**Custo**: é a estratégia mais cara — a cada passo faz `expand_size` expansões por
candidato, e cada candidato extra ainda dispara `num_rollouts` simulações completas de
`sample_path()` dentro de `_eval_state_without_answer()` só para pontuação. Em troca,
explora múltiplos caminhos de raciocínio em paralelo antes de se comprometer com um.

> **Nota sobre configuração**: `expand_size`, `num_rollouts` e `beam_size` **não têm flag
> de CLI** em `src/config.py` — `run_inference.py` chama `tree_search()` sem passá-los,
> então sempre usa os valores default do método (`expand_size=4`, `num_rollouts=2`,
> `beam_size=1`). Para mudar esses valores é preciso chamar `corag_agent.tree_search(...)`
> diretamente em código Python, passando os argumentos explicitamente.

---

## 3. `best_of_n()` — estratégia `best_of_n`

```python
def best_of_n(self, query, task_desc, max_path_length=3, max_message_length=4096,
              temperature=0.7, n=4, **kwargs) -> RagPath
```
(`corag_agent.py:161-181`)

A mais simples das duas estratégias "com exploração": roda `sample_path()` (a versão
greedy) `n` vezes de forma independente — a primeira amostra com `temperature=0.` (caminho
determinístico/guloso puro) e as demais com a `temperature` configurada (`sample_temperature`
na CLI, default `0.7`) — e depois escolhe o caminho de menor score segundo
`_eval_single_path()` (seção 4).

Controlado pela flag `--best_n` em `config.py` (default `4`).

**Custo**: `n` execuções completas e independentes de `sample_path()` (cada uma já custa
`~2 × max_path_length` chamadas ao modelo), mais uma chamada de `_eval_single_path()` por
candidato para pontuar. Mais barato que `tree_search()` (não faz rollouts recursivos), mas
ainda `n` vezes mais caro que `greedy` puro.

---

## 4. A heurística de scoring compartilhada

`tree_search()` e `best_of_n()` dependem dos mesmos dois métodos para decidir qual caminho
é "melhor":

### `_eval_single_path()` (`corag_agent.py:230-248`)

Monta um prompt com o histórico de subqueries/subanswers do caminho e força o modelo a
"continuar" com a frase fixa `"No relevant information found"`, pedindo os
`prompt_logprobs` dessa continuação (`extra_body={"prompt_logprobs": 1}`). O score é a
média desses logprobs (`parse_answer_logprobs`).

Intuição: quanto **menor** (mais negativo) o logprob de "não encontrei informação
relevante", mais o modelo já "acredita" que tem informação suficiente para responder — por
isso as duas estratégias buscam **minimizar** esse score ao escolher o caminho final.

### `_eval_state_without_answer()` (`corag_agent.py:250-274`)

Usado só pelo `tree_search()` durante a poda do feixe. Em vez de pontuar o caminho parcial
diretamente, faz `num_rollouts` simulações — cada uma continuando o caminho parcial com
`sample_path()` por até mais 2 hops — e pontua cada rollout com `_eval_single_path()`,
retornando a média. Isso estima "se eu continuasse por este caminho, quão perto ficaria de
uma resposta?", em vez de avaliar o estado parcial isoladamente.

---

## Limitação importante: `tree_search`/`best_of_n` exigem vLLM, não funcionam com Ollama

`_eval_single_path()` depende de `extra_body={"prompt_logprobs": 1}` — uma extensão da API
de chat completions **exclusiva do vLLM**. Backends compatíveis com a API da OpenAI mas que
não são vLLM (por exemplo, o **Ollama**) simplesmente não implementam esse campo:

- Testado contra um endpoint Ollama real: a chamada retorna normalmente, mas
  `response.prompt_logprobs` nem existe no objeto de resposta →
  `AttributeError: 'ChatCompletion' object has no attribute 'prompt_logprobs'`.
- Também não há uma alternativa simples via `/v1/completions` com `echo=True` e
  `logprobs=1` (o jeito "clássico" da API da OpenAI de obter logprobs do prompt, teacher
  forcing): o Ollama aceita os parâmetros mas devolve `"logprobs": null`, sem implementá-los.

Ou seja: com um backend Ollama, **só a estratégia `greedy` (`sample_path`) funciona**. Para
`tree_search`/`best_of_n` funcionarem como desenhados no paper, o modelo precisa estar
servido por um vLLM de verdade (local ou remoto).

### Adaptação usada em `src/reproducao/teste.py`: `best_of_n` via self-consistency

Para viabilizar uma forma de "melhor de N" mesmo sem `prompt_logprobs`, `teste.py`
implementa uma variante que **não é o `best_of_n` oficial do `CoRagAgent`**: em vez de
pontuar os caminhos intermediários pelo logprob de "No relevant information found", ela
gera as `n` respostas finais normalmente (`sample_path` + `generate_final_answer`, que só
usam chat comum, compatível com qualquer backend) e escolhe a resposta que **mais se repete**
entre as `n` amostras (self-consistency / votação de maioria), via
`selecionar_por_self_consistency()`.

Diferenças em relação ao `best_of_n()` do `corag_agent.py`:
- Funciona com qualquer backend de chat compatível com OpenAI (Ollama incluso).
- Troca a função de valor do paper (confiança do modelo no caminho, medida por logprob)
  por votação entre respostas finais já geradas — uma técnica diferente, mais parecida com
  self-consistency prompting do que com o método de scoring original do CoRAG.
- Custo similar (ainda `n` execuções completas de `sample_path` + `generate_final_answer`
  cada), só troca o critério de seleção do candidato vencedor.

`tree_search()` continua sem alternativa implementada neste repositório de reprodução —
a mesma técnica de self-consistency poderia, em princípio, ser adaptada para lá, mas exigiria
reescrever `_search()` para não depender de `_eval_state_without_answer()`.

---

## Resumo dos parâmetros de CLI (`src/config.py`)

| Flag | Default | Usado por | Efeito |
|---|---|---|---|
| `--decode_strategy` | `greedy` | despacho em `run_inference.py` | escolhe `sample_path` / `tree_search` / `best_of_n` |
| `--max_path_length` | `3` | as 3 estratégias | número máximo de hops (subquery→subanswer) na cadeia |
| `--sample_temperature` | `0.7` | `tree_search`, `best_of_n` | temperatura usada nas amostras não-gulosas |
| `--best_n` | `4` | `best_of_n` | quantos caminhos independentes amostrar antes de escolher o melhor |

`expand_size`, `num_rollouts` e `beam_size` do `tree_search` não aparecem nessa tabela por
não terem flag de CLI (ver nota na seção 2).
