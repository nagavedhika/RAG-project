import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & Groq imports
from langchain_groq import ChatGroq
from groq import Groq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from graph_extractor import build_knowledge_graph

load_dotenv(override=True)

app = FastAPI(title="Apple-Grade RAG Backend", version="1.0.0")

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active document and vector store cache
ACTIVE_DOCS = {}
PERSIST_BASE_DIR = os.path.abspath("chroma_db")
os.makedirs(PERSIST_BASE_DIR, exist_ok=True)

# Default embedding model
def get_embeddings():
    try:
        return OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        print(f"Warning: OllamaEmbeddings error: {e}")
        raise RuntimeError("Embedding model initialization failed. Ensure Ollama is running nomic-embed-text.")

# System prompts
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert assistant providing clear, sophisticated, and polished explanations.

Answer ONLY using the provided context.
If the answer is not present in the context, say "I don't know."

Writing & Formatting Instructions:
- Write in clean, flowing, natural prose with well-structured paragraphs.
- Do NOT use raw markdown asterisks (no **bold** or *italic*), and do NOT use leading dashes (-) or bullet hyphens.
- Explain terms and categories fluidly in complete, professional sentences.
- Avoid robotic checklists with symbols; synthesize the information gracefully.

Context:
{context}

Question:
{question}

Answer:
""")

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Given the conversation history and the user's latest question,
rewrite the latest question as a standalone question.

The standalone question should be understandable without
the conversation history.

Conversation history:
{chat_history}

Latest question:
{question}

Standalone question:
""")

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    document_name: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []
    model: Optional[str] = None

def get_groq_api_key() -> str:
    # Always reload to pick up any changes in .env
    load_dotenv(override=True)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is not set in the .env file. Please add GROQ_API_KEY=gsk_... to your .env file."
        )
    return key

def get_available_chat_models(key: str) -> List[str]:
    try:
        client = Groq(api_key=key)
        all_models = [m.id for m in client.models.list().data]
        # Filter out whisper audio models, prompt-guard, and safeguards
        chat_models = [
            m for m in all_models 
            if not m.startswith("whisper") and "guard" not in m and "safeguard" not in m
        ]
        return chat_models if chat_models else ["openai/gpt-oss-120b"]
    except Exception as e:
        print(f"Failed to fetch models from Groq: {e}")
        return ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

@app.get("/api/status")
def get_status():
    load_dotenv(override=True)
    key = os.getenv("GROQ_API_KEY", "").strip()
    has_key = bool(key)
    
    available_models = []
    if has_key:
        available_models = get_available_chat_models(key)

    # list indexed documents in chroma_db folder
    docs = []
    if os.path.exists(PERSIST_BASE_DIR):
        for item in os.listdir(PERSIST_BASE_DIR):
            item_path = os.path.join(PERSIST_BASE_DIR, item)
            if os.path.isdir(item_path):
                docs.append(item)

    default_model = "openai/gpt-oss-120b"
    if available_models and default_model not in available_models:
        default_model = available_models[0]

    return {
        "status": "online",
        "has_env_groq_key": has_key,
        "indexed_documents": docs,
        "available_models": available_models,
        "default_model": default_model
    }

@app.get("/api/graph")
def get_knowledge_graph(document_name: Optional[str] = None, force_refresh: bool = False):
    try:
        data = build_knowledge_graph(PERSIST_BASE_DIR, target_doc=document_name, force_refresh=force_refresh)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build graph: {str(e)}")

@app.post("/api/graph/refresh")
def refresh_knowledge_graph(document_name: Optional[str] = None):
    try:
        data = build_knowledge_graph(PERSIST_BASE_DIR, target_doc=document_name, force_refresh=True)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh graph: {str(e)}")

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
    
    clean_name = os.path.splitext(os.path.basename(file.filename))[0]
    safe_name = "".join(c for c in clean_name if c.isalnum() or c in ("-", "_")).strip() or "document"
    persist_dir = os.path.join(PERSIST_BASE_DIR, safe_name)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        if not documents:
            raise HTTPException(status_code=400, detail="The PDF appears to be empty or unreadable.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = get_embeddings()

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        ACTIVE_DOCS[safe_name] = vector_store

        return {
            "status": "success",
            "document_name": safe_name,
            "original_filename": file.filename,
            "page_count": len(documents),
            "chunk_count": len(chunks),
            "sample_snippet": chunks[0].page_content[:200] if chunks else ""
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    api_key = get_groq_api_key()
    
    # Determine document
    doc_name = req.document_name
    if not doc_name:
        if ACTIVE_DOCS:
            doc_name = next(iter(ACTIVE_DOCS.keys()))
        else:
            dirs = [d for d in os.listdir(PERSIST_BASE_DIR) if os.path.isdir(os.path.join(PERSIST_BASE_DIR, d))]
            if dirs:
                doc_name = dirs[0]
            else:
                raise HTTPException(status_code=400, detail="No document has been uploaded yet. Please upload a PDF first.")

    persist_dir = os.path.join(PERSIST_BASE_DIR, doc_name)
    if not os.path.exists(persist_dir):
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' vector store not found.")

    # Initialize / retrieve vector store
    if doc_name in ACTIVE_DOCS:
        vector_store = ACTIVE_DOCS[doc_name]
    else:
        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        ACTIVE_DOCS[doc_name] = vector_store

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Pick valid model
    available = get_available_chat_models(api_key)
    selected_model = req.model
    if not selected_model or selected_model not in available:
        if "openai/gpt-oss-120b" in available:
            selected_model = "openai/gpt-oss-120b"
        elif available:
            selected_model = available[0]
        else:
            selected_model = "openai/gpt-oss-120b"

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=selected_model,
        temperature=0.2
    )

    # Rewrite question if chat history exists
    user_question = req.question
    rewritten_question = user_question

    if req.chat_history and len(req.chat_history) > 0:
        history_text = "\n".join(
            f"{msg.role}: {msg.content}" for msg in req.chat_history
        )
        try:
            rewrite_prompt_formatted = REWRITE_PROMPT.invoke({
                "chat_history": history_text,
                "question": user_question
            })
            rewritten_response = llm.invoke(rewrite_prompt_formatted)
            rewritten_question = rewritten_response.content.strip()
        except Exception as e:
            print(f"Rewrite question failed, falling back to original: {e}")
            rewritten_question = user_question

    # Retrieve relevant documents
    docs = retriever.invoke(rewritten_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = ANSWER_PROMPT.invoke({
        "context": context,
        "question": user_question
    })

    try:
        response = llm.invoke(formatted_prompt)
        raw_answer = response.content.strip()
        # Clean unwanted markdown asterisks and leading dashes
        import re
        no_stars = re.sub(r'\*{1,3}', '', raw_answer)
        cleaned_lines = [re.sub(r'^\s*[-•–—]\s*', '', line) for line in no_stars.split('\n')]
        answer = '\n'.join(cleaned_lines).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq generation failed: {str(e)}")

    sources = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]

    return {
        "answer": answer,
        "rewritten_question": rewritten_question,
        "sources": sources,
        "document_name": doc_name,
        "model": selected_model
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
