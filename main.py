from typing import Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from joblib import load
from DataModel import DataModel

# Definición de la FastAPI app
app = FastAPI()

# Modelo de datos para la entrada de la reseña
class Review(BaseModel):
    Review: str

# raiz
@app.get("/")
def read_root():
    with open("front.html", "r") as file:
        content = file.read()
    return HTMLResponse(content)

# Post para hacer predicciones
@app.post("/predict")
def make_predictions(dataModel: DataModel):
   df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
   return df["Review"][0]
   

# Get review
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}






