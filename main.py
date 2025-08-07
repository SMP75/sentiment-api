"""
Sentiment Analysis API
----------------------
POST /sentiment
  JSON body: { "text": "<your sentence>" }

Response:
  { "label": "positive", "scores": { "negative": 0.02, "positive": 0.98 } }
"""

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# -------- configuration --------
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"  # 2-class: negative / positive
device   = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------- load model once at startup --------
sent_pipe = pipeline(
    "text-classification",
    model=MODEL_ID,
    top_k=None,                  # return both scores
    device=0 if device.startswith("cuda") else -1,
)

# -------- FastAPI setup --------
app = FastAPI(title="Sentiment API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SentReq(BaseModel):
    text: str

@app.post("/sentiment")
def analyze(req: SentReq):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text cannot be empty")
    result = sent_pipe(text)[0]           # list of dicts, length 2
    scores = {d["label"].lower(): float(d["score"]) for d in result}
    label  = max(scores, key=scores.get)
    return {"label": label, "scores": scores}
