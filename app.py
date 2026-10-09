
import streamlit as st
import numpy as np
import faiss
import tempfile
import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


st.set_page_config(
    page_title="Study Material AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Study Material AI Agent")
st.write("Upload a PDF and ask questions from your study material.")


# -----------------------------
# Gemini API
# -----------------------------
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key is not configured.")
    st.stop()

client = genai.Client(api_key=api_key)


# -----------------------------
# Embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Extract PDF text
# -----------------------------
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Split text into chunks
# -----------------------------
def create_chunks(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# -----------------------------
# Create FAISS database
# -----------------------------
@st.cache_resource
def create_index(chunks):

    embeddings = embedding_model.encode(chunks)

    embeddings_array = np.array(
        embeddings
    ).astype("float32")

    index = faiss.IndexFlatL2(
        embeddings_array.shape[1]
    )

    index.add(embeddings_array)

    return index


# -----------------------------
# Search PDF
# -----------------------------
def search_pdf(query, chunks, index, k=3):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        min(k, len(chunks))
    )

    results = []

    for i in indices[0]:

        if i >= 0:
            results.append(chunks[i])

    return results


# -----------------------------
# LangGraph State
# -----------------------------
class AgentState(TypedDict):

    question: str
    context: str
    answer: str


# -----------------------------
# Build LangGraph
# -----------------------------
def build_agent(chunks, index):

    def retrieve_document(state: AgentState):

        chunks_found = search_pdf(
            state["question"],
            chunks,
            index,
            k=3
        )

        context = "\n\n".join(chunks_found)

        return {
            "context": context
        }


    def generate_answer(state: AgentState):

        prompt = f"""
You are a Study Material AI Agent.

Answer the user's question using ONLY the
provided document context.

If the answer is not present in the context,
say:

"I couldn't find this information in the uploaded document."

DOCUMENT CONTEXT:
{state["context"]}

USER QUESTION:
{state["question"]}

Give a clear and concise answer.
"""

        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt,
            generation_config={
                "thinking_level": "minimal"
            }
        )

        return {
            "answer": interaction.output_text
        }


    graph_builder = StateGraph(AgentState)

    graph_builder.add_node(
        "retrieve",
        retrieve_document
    )

    graph_builder.add_node(
        "generate",
        generate_answer
    )

    graph_builder.add_edge(
        START,
        "retrieve"
    )

    graph_builder.add_edge(
        "retrieve",
        "generate"
    )

    graph_builder.add_edge(
        "generate",
        END
    )

    return graph_builder.compile()


# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    with st.spinner("Processing PDF..."):

        text = extract_pdf_text(
            uploaded_file
        )

        if not text.strip():

            st.error(
                "Could not extract text from this PDF."
            )

            st.stop()

        chunks = create_chunks(text)

        index = create_index(chunks)

        agent_graph = build_agent(
            chunks,
            index
        )

    st.success(
        f"✅ PDF processed successfully! "
        f"{len(chunks)} chunks created."
    )


    # -----------------------------
    # Text Question
    # -----------------------------
    question = st.text_input(
        "💬 Ask a question about your PDF"
    )


    # -----------------------------
    # Voice Question
    # -----------------------------
    audio = st.audio_input(
        "🎤 Or ask your question using your microphone"
    )


    # -----------------------------
    # Ask Agent
    # -----------------------------
    if st.button("Ask AI Agent"):

        if not question and not audio:

            st.warning(
                "Please enter a question or record your voice."
            )

        else:

            # Voice processing can be added next.
            # For now, process typed questions.

            if question:

                with st.spinner(
                    "🤖 AI Agent is thinking..."
                ):

                    result = agent_graph.invoke(
                        {
                            "question": question,
                            "context": "",
                            "answer": ""
                        }
                    )

                st.subheader("🤖 AI Agent Answer")

                st.write(
                    result["answer"]
                )

else:

    st.info(
        "Upload a PDF to start using the AI Agent."
    )
