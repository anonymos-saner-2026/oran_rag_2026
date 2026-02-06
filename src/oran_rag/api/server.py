from __future__ import annotations
import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..rag.answerer import RAGEngine
from .schemas import AskRequest, SearchRequest

app = FastAPI(title="O-RAN RAG (Qwen3-4B)", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE: RAGEngine | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(req: AskRequest):
    assert ENGINE is not None, "Engine not initialized"
    return ENGINE.answer(req.question, filters=req.filters, top_k=req.top_k)

@app.post("/search")
def search(req: SearchRequest):
    assert ENGINE is not None, "Engine not initialized"
    fused = ENGINE.retrieve(req.query, filters=req.filters)
    reranked = ENGINE.rerank(req.query, fused)
    # return top chunks metadata only
    out = []
    for s, c in reranked[:req.top_k]:
        out.append(
            {
                "score": float(s),
                "chunk_id": c["chunk_id"],
                "doc_id": c.get("doc_id",""),
                "version": c.get("version",""),
                "wg": c.get("wg",""),
                "clause_id": c.get("clause_id",""),
                "section_path": c.get("section_path",""),
                "pages": f"p{c.get('page_start','?')}-p{c.get('page_end','?')}",
                "preview": (c.get("text","")[:300] + "...") if c.get("text","") else "",
            }
        )
    return {"results": out}

def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    ENGINE = RAGEngine.from_config(args.config)
    host = ENGINE.cfg["server"]["host"]
    port = int(ENGINE.cfg["server"]["port"])

    import uvicorn
    uvicorn.run("oran_rag.api.server:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
