from joblib import load
import pandas as pd

from DataModel import DataModel

class Model:

    def __init__(self,columns):
        self.model = load("assets\modelo_NB.pkl")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result

