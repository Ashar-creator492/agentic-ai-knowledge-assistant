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

print("API KEY:", os.getenv("GROQ_API_KEY"))

#
 #Settings.llm = LI_Groq(model="llama-3.1-8b-instant", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.node_parser = SentenceSplitter(chunk_size=600, chunk_overlap=100)

DATA_DIR = "data"

docs = SimpleDirectoryReader(
    input_dir=DATA_DIR
).load_data()

print(f"Loaded {len(docs)} documents")

index = VectorStoreIndex.from_documents(docs) if docs else None

retriever = index.as_retriever(similarity_top_k = 4) if index else None



# Extracts page number safely from varying metadata formats.
def _page_from_meta(meta: dict) -> str:
    # LlamaIndex metadata keys vary by reader; try common ones
    for k in ["page_label", "page_number", "page", "page_idx"]:
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return "?"

def retrieve_with_citations(query: str, top_k: int = 5, max_chars: int = 650) -> str:
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


#llm for the agent
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

@tool
def retrieval_from_docs(question:str)->str:
    """Search private survey papers and return relevant chunks with citations.
    Use for: definitions, taxonomy, limitations, open problems, comparisons. """

    return retrieve_with_citations(question)



tools  = [retrieval_from_docs]

tool_agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt= ("You are a TOOL-AUGMENTED assistant for survey papers.\n"
        "Rules (must follow):\n"
        "1) If the user asks any arithmetic, you MUST call calculator and output the numeric result.\n"
        "2) If the user asks for claims from documents, you MUST call private_docs_retriever.\n"
        "3) Final output MUST have exactly 2 sections in this order:\n"
        "   A) Math: <number>\n"
        "   B) Open problems (3 lines):\n"
        "      - <open problem> (Cite: file p.X)\n"
        "      - <open problem> (Cite: file p.X)\n"
        "      - <open problem> (Cite: file p.X)\n"
        "If evidence is missing, write: 'Not found in documents' (no invention).\n"
        "Never invent citations."
    )
)


query = "  list 3 key open problems in CSP"
res = tool_agent.invoke({"messages": [{"role": "user", "content": query}]})
print(res["messages"][-1].content)
