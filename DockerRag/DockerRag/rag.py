import os
import json
import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

region = os.environ.get("REGION")
knowledgeBaseId = os.environ.get("BEDROCK_KB_ID")

bedrock_client = boto3.client(service_name='bedrock-agent-runtime', region_name=region)


def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False


def retrieve_results(prompt: str, knowledgeBaseId: str):
    knowledge_base_retrieval = bedrock_client.retrieve(
        knowledgeBaseId=knowledgeBaseId,
        retrievalQuery={
            "text": prompt
        }
    )
    return knowledge_base_retrieval['retrievalResults']


def parse_results(retrieval_results):
    contexts = []
    for result in retrieval_results:
        contexts.append({
            "text": result["content"]["text"],
            "score": result["score"],
            "source": result["location"]
        })
    return contexts


def generate_with_ollama(query: str, contexts: list, model: str = "mistral:latest"):
    context_text = "\n\n".join([f"[Source {i+1}]: {c['text']}" for i, c in enumerate(contexts)])
    past_queries = [];
    prompt = f"""Use the following context to answer the question.

    Context:
    {context_text}

    Conversation till now: 
    {past_queries}
    
    Question: {query}

    Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if not data.get("done"):
                yield data.get("response", "")

    past_queries += query
    print(past_queries)


def generate_with_bedrock(query: str, contexts: list):
    context_text = "\n\n".join([f"[Source {i+1}]: {c['text']}" for i, c in enumerate(contexts)])

    prompt = f"""Use the following context to answer the question.

    Context:
    {context_text}

    Question: {query}

    Answer:"""

    response = bedrock_client.retrieve_and_generate(
        input={"text": prompt},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": knowledgeBaseId,
                "modelArn": "arn:aws:bedrock:eu-north-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
            }
        }
    )

    # Yield as single chunk to keep interface consistent with Ollama
    yield response['output']['text']


def rag_pipeline(query: str, knowledge_base_id: str = knowledgeBaseId, ollama_model: str = "mistral:latest"):
    print("🔍 Retrieving from Knowledge Base...")
    raw_results = retrieve_results(query, knowledge_base_id)

    print(f"Found {len(raw_results)} chunks")
    contexts = parse_results(raw_results)

    if is_ollama_running():
        print("✅ Ollama is running — using local generation...")
        answer_generator = generate_with_ollama(query, contexts, ollama_model)
    else:
        print("⚠️ Ollama is not running — falling back to Bedrock generation...")
        answer_generator = generate_with_bedrock(query, contexts)

    return {
        "query": query,
        "answer": answer_generator,
        "accuracy": round(contexts[0]["score"] * 100) if contexts else 0,
        "chunk_count": len(contexts),
        "sources": [c["source"] for c in contexts],
        "engine": "ollama" if is_ollama_running() else "bedrock"
    }


if __name__ == "__main__":
    query = input(">>> ")
    result = rag_pipeline(query=query)
    for token in result["answer"]:
        print(token, end="", flush=True)
    print()