import boto3
import requests

bedrock_client = boto3.client(service_name='bedrock-agent-runtime', region_name='eu-north-1')
knowledgeBaseId="D9GSQGHM9G"

def retrieve_results(prompt:str, knowledgeBaseId:str):
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
            "text": result["content"]["text"],         # The actual text chunk
            "score": result["score"],                  # Relevance score
            "source": result["location"]               # S3 URI or source info
        })
    return contexts

def generate_with_ollama(query: str, contexts: list, model: str = "mistral:latest"):
    context_text = "\n\n".join([f"[Source {i+1}]: {c['text']}" for i, c in enumerate(contexts)])

    prompt = f"""Use the following context to answer the question.

    Context:
    {context_text}

    Question: {query}

    Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True       # changed to True
        },
        stream=True              # stream the HTTP response too
    )

    for line in response.iter_lines():
        if line:
            chunk = line.decode('utf-8')
            import json
            data = json.loads(chunk)
            if not data.get("done"):
                yield data.get("response", "")
                
                

def rag_pipeline(query: str, knowledge_base_id: str = "D9GSQGHM9G", ollama_model: str = "mistral:latest"):
    print(f"🔍 Retrieving from Knowledge Base...")
    raw_results = retrieve_results(query, knowledge_base_id)
    
    print(f"Found {len(raw_results)} chunks")
    contexts = parse_results(raw_results)
    
    print(f"Generating with Ollama ({ollama_model})...")
    answer = generate_with_ollama(query, contexts, ollama_model)
    
    return {
        "query": query,
        "answer": answer,
        "accuracy": round(contexts[0]["score"] * 100) if contexts else 0,  # top chunk score as accuracy
        "chunk_count": len(contexts),
        "sources": [c["source"] for c in contexts]
    }

if __name__ == "__main__":
    # Usage
    
    query = input(">>> ")
    
    result = rag_pipeline(
        query=query,
        knowledge_base_id=knowledgeBaseId,
        ollama_model="mistral:latest"
    )

    print(result["answer"])