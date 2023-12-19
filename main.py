from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastembed.embedding import FlagEmbedding as Embedding
import numpy as np


class Sentences(BaseModel):
    sentences: List[str]


app = FastAPI()

try:
    embedding_model = Embedding(model_name="BAAI/bge-small-en-v1.5", max_length=512)
except Exception as e:
    print(f'failed to initialize the model: {e}')
    raise HTTPException(status_code=500, detail=f"Failed to initialize the model.")

@app.get('/')
def root():
    print("Model is running")
    return {"message": "Application is running"}

@app.post("/embed")
async def create_embeddings(sentences: Sentences):
    try:
        embeddings: List[np.ndarray] = list(embedding_model.embed(sentences.sentences))
        return {"embeddings": [embedding.tolist() for embedding in embeddings]}
    except Exception as e:
        print(f'failed creating embeddings: {e}')
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings.")

