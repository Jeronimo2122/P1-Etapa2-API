from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status

class DataModel(BaseModel):

    Review: str

#Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ['Review']
    
