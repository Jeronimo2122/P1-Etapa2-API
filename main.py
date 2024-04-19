from typing import Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from joblib import load
from DataModel import DataModel
from PredictionModel import Model

# Definición de la FastAPI app
app = FastAPI()

# Modelo de datos para la entrada de la reseña
class Review(BaseModel):
    Review: str

# Crear modelo
model = Model()

# raiz
@app.get("/")
def read_root():
    with open("front.html", "r") as file:
        content = file.read()
    return HTMLResponse(content)

# Post para hacer predicciones
@app.post("/predict")
def make_predictions(dataModel: DataModel):
   text = dataModel.Review
   result = model.make_predictions(dataModel.Review)
   return [text, result.tolist()[0]]
   

# Get review
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}






