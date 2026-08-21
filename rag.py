# ========= Standard Library =========

import os
from typing import List

# ========= Environment Variables =========

from dotenv import load_dotenv

# ========= LangChain =========

from langchain_groq import ChatGroq 
from langchain_core.tools import tool
from langchain.agents import create_agent

# ========= LlamaIndex =========

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
# ========= Chroma =========

import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

load_dotenv()



#
 #Settings.llm = LI_Groq(model="llama-3.1-8b-instant", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.node_parser = SentenceSplitter(chunk_size=600, chunk_overlap=100)

UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "agentic_ai_knowledge_assistant"
USER_ID = "user_1"

def load_documents(pdf_path: str):
    docs = SimpleDirectoryReader(
        input_files=[pdf_path]
    ).load_data()

    print(f"Loaded {len(docs)} documents")
    return docs


def get_indexed_filenames() -> set[str]:
    """Return file names already present in the Chroma collection.

    This is intentionally simple for the current stage; later we can replace
    filename-based identity with a user_id/document_id scheme.
    """
    try:
        existing = collection.get(include=["metadatas"])
    except Exception as exc:
        print(f"Could not read Chroma metadata: {exc}")
        return set()

    indexed = set()
    for meta in existing.get("metadatas", []) or []:
        if not isinstance(meta, dict):
            continue
        file_name = meta.get("file_name") or meta.get("filename")
        if file_name:
            indexed.add(file_name)

    return indexed


def add_document_metadata(doc, filename: str):
    metadata = dict(doc.metadata or {})
    metadata["document_id"] = filename
    metadata["filename"] = filename
    metadata["file_name"] = metadata.get("file_name") or filename
    metadata["user_id"] = USER_ID
    doc.metadata = metadata


def migrate_existing_user_ids():
    """Add user_id to legacy Chroma rows without re-embedding documents."""
    try:
        existing = collection.get(include=["metadatas"])
    except Exception as exc:
        print(f"Could not read Chroma metadata for migration: {exc}")
        return

    ids = existing.get("ids", []) or []
    metadatas = existing.get("metadatas", []) or []

    updates = []
    for doc_id, meta in zip(ids, metadatas):
        if not isinstance(meta, dict):
            continue
        if meta.get("user_id") == USER_ID:
            continue
        updated = dict(meta)
        updated["user_id"] = USER_ID
        updates.append({"id": doc_id, "metadata": updated})

    if not updates:
        print(f"All existing Chroma records already have user_id={USER_ID}")
        return

    collection.update(
        ids=[item["id"] for item in updates],
        metadatas=[item["metadata"] for item in updates],
    )
    print(f"Updated {len(updates)} existing Chroma records with user_id={USER_ID}")


def index_pdf(pdf_path: str, user_id: str = USER_ID):
    """Index one specific PDF into the existing persistent Chroma collection."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    indexed_filenames = get_indexed_filenames()
    if filename in indexed_filenames:
        print(f"Skipping already indexed PDF: {filename}")
        return

    docs = load_documents(pdf_path)
    for doc in docs:
        add_document_metadata(doc, filename)
        doc.metadata["user_id"] = user_id

    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context
    )
    print(f"Indexed PDF: {filename} (user_id={user_id})")


def index_new_pdfs():
    if not os.path.isdir(UPLOAD_DIR):
        print(f"Upload directory not found: {UPLOAD_DIR}")
        return

    pdf_files = [
        os.path.join(UPLOAD_DIR, file)
        for file in sorted(os.listdir(UPLOAD_DIR))
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDFs found in uploads.")
        return

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        if filename in get_indexed_filenames():
            print(f"Skipping already indexed PDF: {filename}")
            continue
        index_pdf(pdf_path, user_id=USER_ID)

    if not any(os.path.basename(path) not in get_indexed_filenames() for path in pdf_files):
        print("No new PDFs to index.")


client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

migrate_existing_user_ids()
index_new_pdfs()

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)

user_filter = MetadataFilters(
    filters=[
        MetadataFilter(
            key="user_id",
            value=USER_ID,
            operator=FilterOperator.EQ,
        )
    ]
)

print(f"USER_ID = {USER_ID}")
retriever = index.as_retriever(
    similarity_top_k=3,
    filters=user_filter,
)


# Extracts page number safely from varying metadata formats.
def _page_from_meta(meta: dict) -> str:
    # LlamaIndex metadata keys vary by reader; try common ones
    for k in ["page_label", "page_number", "page", "page_idx"]:
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return "?"

def retrieve_with_citations(query: str, top_k: int = 3, max_chars: int = 400) -> str:
    """
    Returns chunks with citations like: [SOURCE: file.pdf p.3 | score=0.812] ...
    """
    if retriever is None:
        return "No index available. Add docs to DATA_DIR and rebuild index."
    # temporarily override top_k if desired
    hits = retriever.retrieve(query)[:top_k]
    out: List[str] = []
    for h in hits:
        meta = h.node.metadata or {}
        src = meta.get("file_name", "doc")
        page = _page_from_meta(meta)
        txt = (h.node.get_content() or "").strip().replace("\n", " ")
        out.append(f"[SOURCE: {src} p.{page} | score={h.score:.3f}] {txt[:max_chars]}")
    return "\n\n".join(out) if out else "No relevant chunks found."

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

@tool
def retrieval_from_docs(question:str)->str:
    """Search private survey papers and return relevant chunks with citations.
    Use for: definitions, taxonomy, limitations, open problems, comparisons. """

    return retrieve_with_citations(question)



tools  = [retrieval_from_docs]

tool_agent = create_agent(
    model = llm,
    tools = tools,
  system_prompt=(
        "You are a helpful assistant that answers questions using the provided "
        "course documents.\n\n"
        
        "Rules:\n"
        "1) When the user asks about information from the documents, "
        "you MUST call the retrieval_from_docs tool.\n"
        
        "2) Use the retrieved document chunks to answer the question.\n"
        
        "3) Always include the source filename and page number when citing "
        "information from the documents.\n"
        
        "4) Never invent information or citations.\n"
        
        "5) If the documents do not contain enough information to answer "
        "the question, say: 'Not found in documents.'\n"
    )
)


query = "explain force"

res = tool_agent.invoke({
    "messages": [
        {"role": "user", "content": query}
    ]
})

print(res["messages"][-1].content)
