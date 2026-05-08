import os
import json
import boto3
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

region = os.environ.get("REGION")
knowledgeBaseId = os.environ.get("BEDROCK_KB_ID")

bedrock_client = boto3.client(service_name='bedrock-agent-runtime', region_name=region)

# ── ChromaDB setup — persists to disk across restarts ──
CHROMA_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
embeddings = OllamaEmbeddings(model="mistral:latest")
chroma_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def ingest_document(file_path: str, original_filename: str) -> int:
    """
    Loads a document, splits it into chunks,
    generates embeddings via Ollama, stores in ChromaDB.
    Returns the number of chunks created.
    """
    ext = os.path.splitext(original_filename)[1].lower()

    # Step 1 — Load the document based on file type
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.csv': CSVLoader,
        '.md': UnstructuredMarkdownLoader,
    }

    loader_class = loaders.get(ext)
    if not loader_class:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = loader_class(file_path)
    documents = loader.load()

    # Step 2 — Split into chunks
    # chunk_size: max characters per chunk
    # chunk_overlap: overlap between chunks to preserve context at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    # Step 3 — Add source filename as metadata to each chunk
    for chunk in chunks:
        chunk.metadata['source_file'] = original_filename

    # Step 4 — Generate embeddings and store in ChromaDB
    chroma_store.add_documents(chunks)

    print(f"✅ Ingested {original_filename} → {len(chunks)} chunks stored in ChromaDB")
    return len(chunks)


def retrieve_from_chroma(query: str, num_results: int = 3) -> list:
    """
    Searches ChromaDB for chunks relevant to the query.
    Returns results in the same format as parse_results()
    so they can be merged with Bedrock results seamlessly.
    """
    # Skip if ChromaDB is empty
    if chroma_store._collection.count() == 0:
        return []

    results = chroma_store.similarity_search_with_score(query, k=num_results)

    contexts = []
    for doc, score in results:
        # ChromaDB returns L2 distance — lower is better
        # Convert to a 0-1 similarity score to match Bedrock's format
        similarity = 1 / (1 + score)
        contexts.append({
            "text": doc.page_content,
            "score": similarity,
            "source": doc.metadata.get('source_file', 'local document')
        })

    return contexts


def retrieve_results(prompt: str, knowledgeBaseId: str):
    knowledge_base_retrieval = bedrock_client.retrieve(
        knowledgeBaseId=knowledgeBaseId,
        retrievalQuery={"text": prompt}
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


def generate_with_ollama(query: str, contexts: list, history: list = [], model: str = "mistral:latest"):
    context_text = "\n\n".join([f"[Source {i+1}]: {c['text']}" for i, c in enumerate(contexts)])

    history_text = ""
    if history:
        history_text = "\n\nConversation so far:\n"
        for message in history:
            role = "User" if message["role"] == "user" else "Assistant"
            history_text += f"{role}: {message['content']}\n"

    prompt = f"""Use the following context to answer the question.
If the question refers to something mentioned earlier in the conversation, use that context too.

Context:
{context_text}
{history_text}
Current Question: {query}

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


def generate_with_bedrock(query: str, contexts: list, history: list = []):
    context_text = "\n\n".join([f"[Source {i+1}]: {c['text']}" for i, c in enumerate(contexts)])

    history_text = ""
    if history:
        history_text = "\n\nConversation so far:\n"
        for message in history:
            role = "User" if message["role"] == "user" else "Assistant"
            history_text += f"{role}: {message['content']}\n"

    prompt = f"""Use the following context to answer the question.

Context:
{context_text}
{history_text}
Current Question: {query}

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

    yield response['output']['text']


def rag_pipeline(query: str, history: list = [], knowledge_base_id: str = None, ollama_model: str = "mistral:latest"):
    knowledge_base_id = knowledge_base_id or knowledgeBaseId

    print("🔍 Retrieving from Knowledge Base...")
    raw_results = retrieve_results(query, knowledge_base_id)
    bedrock_contexts = parse_results(raw_results)

    print("🔍 Retrieving from ChromaDB...")
    local_contexts = retrieve_from_chroma(query)

    # Merge — local docs first, then Bedrock
    contexts = local_contexts + bedrock_contexts

    print(f"Found {len(contexts)} total chunks ({len(local_contexts)} local, {len(bedrock_contexts)} Bedrock)")

    if is_ollama_running():
        print("✅ Ollama is running — using local generation...")
        answer_generator = generate_with_ollama(query, contexts, history, ollama_model)
    else:
        print("⚠️ Ollama not running — falling back to Bedrock...")
        answer_generator = generate_with_bedrock(query, contexts, history)

    return {
        "query": query,
        "answer": answer_generator,
        "accuracy": round(contexts[0]["score"] * 100) if contexts else 0,
        "chunk_count": len(contexts),
        "sources": [c["source"] for c in contexts],
        "engine": "ollama" if is_ollama_running() else "bedrock"
    }


if __name__ == "__main__":
    history = []
    while True:
        query = input(">>> ")
        if query.lower() in ("exit", "quit"):
            break

        result = rag_pipeline(query=query, history=history)

        answer = ""
        for token in result["answer"]:
            print(token, end="", flush=True)
            answer += token
        print()

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})