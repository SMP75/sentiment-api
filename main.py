import torch
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware   # ← import
from transformers import pipeline

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
device   = "cuda:0" if torch.cuda.is_available() else "cpu"

sent_pipe = None   # will lazy-load later

# ---------- FastAPI ----------
app = FastAPI(title="Sentiment API")

app.add_middleware(                    # ← FIRST arg is the class
    CORSMiddleware,                    # ← this was missing
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- lazy loader ----------
def get_pipe():
    global sent_pipe
    if sent_pipe is None:
        print("[lazy] loading sentiment model …")
        sent_pipe = pipeline("text-classification",
                             model=MODEL_ID,
                             top_k=None,
                             device=0 if device.startswith("cuda") else -1)
        print("[lazy] model ready")
    return sent_pipe

# ---------- schema ----------
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
