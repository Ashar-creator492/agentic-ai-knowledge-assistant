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

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "agentic_ai_knowledge_assistant"

docs = SimpleDirectoryReader(
    input_dir=DATA_DIR
).load_data()

print(f"Loaded {len(docs)} documents")

client = chromadb.PersistentClient(path = CHROMA_DIR)

collection = client.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection= collection)

storage_context = StorageContext.from_defaults(vector_store = vector_store)

if collection.count() == 0:
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context
    )
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
retriever = index.as_retriever(similarity_top_k=3)


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


query = "What are the main algorithms used to solve a CSP?"

res = tool_agent.invoke({
    "messages": [
        {"role": "user", "content": query}
    ]
})

print(res["messages"][-1].content)
