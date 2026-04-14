from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import shutil
import os
import re

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "data/chromadb"

_embeddings = None
_llm = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
        )
    return _embeddings

def get_llm():
    global _llm
    if _llm is None:
        _llm = OllamaLLM(
            model="deepseek-r1",
            temperature=0.1,
            num_predict=800,   
            num_ctx=3000,      
            repeat_penalty=1.1,
        )
    return _llm

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )
    return splitter.split_text(text)

def save_to_vector_db(text_chunks, filename):
    embeddings = get_embeddings()
    metadatas = [{"source": filename} for _ in text_chunks]

    safe_name = re.sub(r'[^\w]', '_', filename)
    unique_db_path = os.path.join(DB_PATH, safe_name)

    if os.path.exists(unique_db_path):
        return Chroma(
            persist_directory=unique_db_path,
            embedding_function=embeddings
        )

    vectordb = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=unique_db_path,
        metadatas=metadatas
    )
    return vectordb

def answer_with_deepseek(query, context_chunks):
    llm = get_llm()

    template = """INTERDICTION D'INVENTER. RÉPONDS UNIQUEMENT AVEC LE CONTEXTE FOURNI.
Tu es un analyste qui ne connaît QUE les extraits ci-dessous.
Si l'info n'est pas dans les extraits, réponds : "Désolé, je ne trouve pas cette info dans le document."

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE (cite le document, sois précis) :"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    unique_contents = list(dict.fromkeys([res.page_content for res in context_chunks]))[:5]
    context_text = "\n---\n".join(unique_contents)

    full_response = chain.invoke({"context": context_text, "question": query})

    cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()

    if not cleaned_response:
        cleaned_response = full_response.strip()

    if not cleaned_response:
        cleaned_response = "⚠️ DeepSeek n'a pas pu générer de réponse. Reformule ta question."

    return cleaned_response

def clear_memory():
    global _embeddings, _llm
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    os.makedirs(DB_PATH, exist_ok=True)
    _embeddings = None
    _llm = None