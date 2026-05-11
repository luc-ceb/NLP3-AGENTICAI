import os
from openai import OpenAI
from dotenv import load_dotenv

# Carga las variables del .env
load_dotenv()

# Accede a la API key desde la variable de entorno
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generar_rag_summary(documentos):
    """
    Genera un resumen de documentos tipo RAG.
    documentos: lista de dicts con keys 'source', 'page', 'text'
    return: resumen generado por el LLM
    """
    prompt = "Resumí los siguientes documentos de forma concisa, clara y manteniendo los conceptos clave. Usá referencias tipo [source, p. X] en el resumen.\n\n"
    for doc in documentos:
        prompt += f"[{doc['source']}, p. {doc['page']}]: {doc['text']}\n"
    prompt += "\nResumen:\n"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
