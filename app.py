from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import rag


class ChatRequest(BaseModel):
    question: str


app = FastAPI(title="Agentic AI Knowledge Assistant API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def health_check() -> dict[str, Any]:
    return {"message": "Agentic AI Knowledge Assistant API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    source_name = Path(file.filename).name
    safe_name = source_name
    counter = 1

    while (UPLOAD_DIR / safe_name).exists():
        stem = Path(source_name).stem
        suffix = Path(source_name).suffix
        safe_name = f"{stem}_{counter}{suffix}"
        counter += 1

    save_path = UPLOAD_DIR / safe_name

    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded PDF")

    try:
        rag.index_pdf(str(save_path), "user_1")
    except Exception as exc:
        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="PDF uploaded but indexing failed") from exc

    return {
        "success": True,
        "filename": safe_name,
        "message": "PDF uploaded and indexed successfully",
    }


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = rag.tool_agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        answer = response["messages"][-1].content
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Agent failed to answer the question",
        ) from exc

    return {
        "success": True,
        "question": question,
        "answer": answer,
    }
