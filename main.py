import os, torch
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from pydantic import BaseModel
from transformers import pipeline

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
device   = "cuda:0" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Sentiment API (lazy)")
app.add_middleware(
    # keep or tighten CORS as you need
)

# ---------- lazy singleton ----------
def get_pipe():
    if not hasattr(get_pipe, "_pipe"):
        print("[lazy] Loading sentiment model …")
        get_pipe._pipe = pipeline(
            "text-classification",
            model=MODEL_ID,
            top_k=None,
            device=0 if device.startswith("cuda") else -1,
        )
        print("[lazy] Model ready")
    return get_pipe._pipe

# ---------- request schema ----------
class SentReq(BaseModel):
    text: str

# ---------- endpoint ----------
@app.post("/sentiment")
def analyze(req: SentReq, pipe = Depends(get_pipe)):
    txt = req.text.strip()
    if not txt:
        raise HTTPException(400, "text cannot be empty")
    out = pipe(txt)[0]
    scores = {d["label"].lower(): float(d["score"]) for d in out}
    label  = max(scores, key=scores.get)
    return {"label": label, "scores": scores}
