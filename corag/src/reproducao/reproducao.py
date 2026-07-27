import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["API_KEY"],
)

# 3. Faça a requisição de teste
print("Enviando requisição...")

try:
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "corag-8b"),
        messages=[
            {"role": "system", "content": "Você é um assistente útil e direto."},
            {"role": "user", "content": "Olá! Responda 'Estou online!' se você recebeu esta mensagem."}
        ],
        temperature=0.7,
        max_tokens=50
    )

    # 4. Extrai e imprime a resposta
    texto_resposta = response.choices[0].message.content
    print("\n✅ Resposta recebida com sucesso!")
    print(f"🤖 Modelo: {texto_resposta}")

except Exception as e:
    print("\n❌ Ocorreu um erro ao conectar ou processar:")
    print(e)