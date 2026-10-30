from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (e.g. GROQ_API_KEY if configured)
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title(" 🤖 RAG Chatbot ")

# Sidebar configuration
st.sidebar.header("Configuration")
groq_key = os.getenv("GROQ_API_KEY", "").strip()

# Model selection: support Ollama (local) and Groq (cloud LPU)
available_providers = ["Ollama (llama3.2 - Local)"]
if groq_key:
    available_providers.append("Groq (openai/gpt-oss-120b - Fast Cloud)")
    available_providers.append("Groq (qwen/qwen3.8-27b - Cloud)")

selected_provider = st.sidebar.selectbox("LLM Provider", available_providers)
st.sidebar.caption("💡 Ollama runs 100% locally and privately. Groq provides ultra-fast cloud inference.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

# Store conversation history across Streamlit reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cache for active vector store in session state
if "loaded_doc_name" not in st.session_state:
    st.session_state.loaded_doc_name = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

#-------------------------------------------------------
# llm = ChatOllama(
#     model="llama3.2"
# )

# response = llm.invoke("Who are you?")

# print(response.content)


# loading the documents
# loader = PyPDFLoader("data/Java_Notes.pdf")
# documents = loader.load()
# #print(documents[0]) # prints the first document


# #chunking
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100
# )
# chunks = text_splitter.split_documents(documents)
#print("Number of pages:", len(documents))
#print("Number of chunks:", len(chunks))
#print(chunks[0]) # prints the first chunk


#creating the embedding model
#------------------------------------------------------------------
def get_vector_store(uploaded_file):
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    #storing in chroma db- the commented part creates the embeddings everytime the code runs which takes a lot of time
    # vector_store = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embedding_model,
    #     persist_directory="chroma_db" #folder name where the embeddings get stored
    # )

    #this checks if the embedding already exists. if so, it loads it. else a new file is created,the document loader and textsplitter is carried on and the vectors are stored. note how we use 2 different chromadb functions
    document_name = os.path.splitext(uploaded_file.name)[0]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(
        base_dir,
        "chroma_db",
        document_name
    )
    
    # Return cached vector store if already loaded in this session
    if st.session_state.loaded_doc_name == document_name and st.session_state.vector_store is not None:
        return st.session_state.vector_store

    if os.path.exists(persist_directory):
        print("Loading existing ChromaDB...")
        st.toast(f"Loading existing embeddings for {document_name}...", icon="📂")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )

    else:
        print("Creating ChromaDB...")
        st.toast(f"Processing and indexing {document_name}...", icon="⚡")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(documents)

            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=persist_directory
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    st.session_state.loaded_doc_name = document_name
    st.session_state.vector_store = vector_store
    return vector_store

#---------------------------------------------------
def rewrite_question(question, chat_history, llm, rewrite_prompt):
    if chat_history: #this checks if we have chat history and uses the prompt required for it else, we simply use the question
        history_text = "\n".join(
            f"{role}: {message}"
            for role, message in chat_history
        )

        rewrite_prompt_formatted = rewrite_prompt.invoke({
            "chat_history": history_text,
            "question": question
        })

        rewritten_question = llm.invoke(
            rewrite_prompt_formatted
        ).content.strip()

    else:
        rewritten_question = question

    return rewritten_question


#----------------------------------------------------------------

def get_answer(question, retriever, llm, prompt):
    docs = retriever.invoke(question)
    context = "\n\n".join(
        doc.page_content for doc in docs
    )
    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )   
    response = llm.invoke(formatted_prompt)
    
    return response.content

#------------------------------------------------------------------------

# Initialize the selected LLM
if selected_provider.startswith("Groq"):
    from langchain_groq import ChatGroq
    model_id = "openai/gpt-oss-120b" if "gpt-oss-120b" in selected_provider else "qwen/qwen3.8-27b"
    llm = ChatGroq(model_name=model_id, api_key=groq_key)
else:
    llm = ChatOllama(
        model="llama3.2"
    )

#creating the prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

#for rag-memory
rewrite_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and the user's latest question,
rewrite the latest question as a standalone question.

The standalone question should be understandable without
the conversation history. Return ONLY the rewritten question.

Conversation history:
{chat_history}

Latest question:
{question}

Standalone question:
""")

#keep a record of the previous questions and answers
#chat_history=[]    not used if we use streamlit

if uploaded_file: #we check using if condition since we need to run the below code only if we have an uploaded file
    vector_store = get_vector_store(uploaded_file)

    #retrieval
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Display conversation history
    for role, message in st.session_state.chat_history:
        if role == "Human":
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)

    # Chat input
    question = st.chat_input("Ask something about the uploaded PDF...")

    if question:
        # Display the user question immediately
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and thinking..."):
                rewritten_question = rewrite_question(
                    question,
                    st.session_state.chat_history,
                    llm,
                    rewrite_prompt
                )

                answer = get_answer(
                    rewritten_question,
                    retriever,
                    llm,
                    prompt
                )
                st.write(answer)

        st.session_state.chat_history.append(
            ("Human", question)
        )

        st.session_state.chat_history.append(
            ("AI", answer)
        )
else:
    st.info("👆 Please upload a PDF document above to start chatting!")
