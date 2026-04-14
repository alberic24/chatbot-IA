import streamlit as st
import os
from src.processor import process_file
from src.brain import split_text, save_to_vector_db, answer_with_deepseek, clear_memory, get_embeddings
from langchain_chroma import Chroma

st.set_page_config(page_title="DeepSeek Chatbot", layout="wide")
st.title("💬 Chatbot IA avec Historique")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

@st.cache_resource
def load_embeddings():
    return get_embeddings()

embeddings = load_embeddings()

def get_existing_db():
    """Recherche une base de données existante dans les sous-dossiers."""
    if os.path.exists("data/chromadb"):
        subdirs = [
            d for d in os.listdir("data/chromadb")
            if os.path.isdir(os.path.join("data/chromadb", d))
        ]
        if subdirs:
            path = os.path.join("data/chromadb", subdirs[0])
            return Chroma(persist_directory=path, embedding_function=embeddings)
    return None

if "db" not in st.session_state or st.session_state.db is None:
    st.session_state.db = get_existing_db()

with st.sidebar:
    st.header("⚙️ Paramètres")

    if st.button("🗑️ Effacer la discussion"):
        st.session_state.messages = []
        st.rerun()

    if st.button("📁 Réinitialiser les documents"):
        clear_memory()
        st.cache_resource.clear()
        st.session_state.db = None
        st.session_state.messages = []
        st.session_state.processed_files = set()
        st.success("Base de données effacée.")
        st.rerun()

    st.divider()
    st.header("1. Charger un document")
    uploaded_file = st.file_uploader("PDF ou Image", type=["pdf", "png", "jpg"])

    if uploaded_file:
        os.makedirs("data/uploads", exist_ok=True)
        file_path = os.path.join("data/uploads", uploaded_file.name)

        if uploaded_file.name not in st.session_state.processed_files:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Analyse et indexation..."):
                raw_text = process_file(file_path)

            if raw_text:
                with st.spinner("Vectorisation en cours..."):
                    chunks = split_text(raw_text)
                    st.session_state.db = save_to_vector_db(chunks, uploaded_file.name)
                    st.session_state.processed_files.add(uploaded_file.name)
                st.success("Document prêt !")
            else:
                st.error("Impossible de lire le fichier.")
        else:
            st.success(f"'{uploaded_file.name}' déjà chargé ✓")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question sur vos documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.db is not None:
            with st.spinner("DeepSeek analyse le document..."):
                results = st.session_state.db.similarity_search(prompt, k=5)
                response = answer_with_deepseek(prompt, results)

            st.markdown(response)

            # 🐛 Debug temporaire
            with st.expander("🐛 Debug réponse brute"):
                st.code(response)

            with st.expander("🔍 Passages extraits du document"):
                for i, res in enumerate(results):
                    st.info(f"Extrait {i+1} :\n{res.page_content[:300]}...")
        else:
            response = "⚠️ Veuillez d'abord charger un document dans la barre latérale."
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})