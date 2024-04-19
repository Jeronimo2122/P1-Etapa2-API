from joblib import load

class Model:
    def __init__(self):
        self.model = load("assets/modelo.joblib")
        self.pipe = load("assets/vectorizador.joblib")

    def make_predictions(self, data):
        # Asegurarse de que 'data' es una lista, incluso si solo hay una entrada
        if isinstance(data, str):
            data = [data]
        proces = self.pipe.transform(data)
        result = self.model.predict(proces)
        return result

