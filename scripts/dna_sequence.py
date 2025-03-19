from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

app = FastAPI()

class DNASequence(BaseModel):
    sequence: str

def validate_dna(seq: str):
    if not re.match("^[ACGTacgt]+$", seq):
        raise HTTPException(status_code=400, detail="Invalid DNA sequence. Only A, C, G, T are allowed.")

def gc_content(seq: str) -> float:
    gc_count = sum(1 for base in seq if base in "GCgc")
    return round((gc_count / len(seq)) * 100, 2) if seq else 0.0

def reverse_complement(seq: str) -> str:
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]

def transcribe(seq: str) -> str:
    return seq.replace("T", "U").replace("t", "u")

@app.post("/gc-content/")
async def get_gc_content(data: DNASequence):
    validate_dna(data.sequence)
    return {"sequence": data.sequence, "gc_content": gc_content(data.sequence)}

@app.post("/reverse-complement/")
async def get_reverse_complement(data: DNASequence):
    validate_dna(data.sequence)
    return {"sequence": data.sequence, "reverse_complement": reverse_complement(data.sequence)}

@app.post("/transcribe/")
async def get_transcription(data: DNASequence):
    validate_dna(data.sequence)
    return {"sequence": data.sequence, "transcription": transcribe(data.sequence)}

@app.get("/")
async def root():
    return {"message": "Welcome to the Bioinformatics REST API"}