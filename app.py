from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag


class ChatRequest(BaseModel):
    question: str


app = FastAPI(title="Agentic AI Knowledge Assistant API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse("templates/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # Keep the original filename.
    source_name = Path(file.filename).name
    save_path = UPLOAD_DIR / source_name

    # Do not create _1, _2, _3 copies.
    if save_path.exists():
        return {
            "success": True,
            "filename": source_name,
            "message": "PDF already exists and is already indexed."
        }

    # Save the PDF.
    try:
        contents = await file.read()

        with open(save_path, "wb") as f:
            f.write(contents)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded PDF"
        ) from exc

    # Index the PDF.
    try:
        rag.index_pdf(
            str(save_path),
            "user_1"
        )

    except Exception as exc:
        # Remove the file if indexing failed.
        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail="PDF uploaded but indexing failed"
        ) from exc

    return {
        "success": True,
        "filename": source_name,
        "message": "PDF uploaded and indexed successfully"
    }


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    question = (payload.question or "").strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        response = rag.tool_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        })

        answer = response["messages"][-1].content

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Agent failed to answer the question"
        ) from exc

    return {
        "success": True,
        "question": question,
        "answer": answer
    }