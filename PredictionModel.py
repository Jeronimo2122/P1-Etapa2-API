from joblib import load
import stanza
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Model:
    def __init__(self):
        self.model = load("assets/modelo.joblib")
        self.pipe = load("assets/vectorizador.joblib")
        stanza.download('es')
        self.stop_words = set(stopwords.words('spanish'))
        self.nlp = stanza.Pipeline(lang='es', processors='tokenize,lemma')

    @staticmethod
    def remove_noise(text):
        text = re.sub(r'<.*?>', '', text)  # Eliminar HTML tags
        text = re.sub(r'[0-9]+', '', text) # Eliminar números
        text = re.sub(r'[^a-záéíóúñA-ZÁÉÍÓÚÑ\s]', '', text)  # Eliminar caracteres especiales
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text

    @staticmethod
    def clean_text(text, stop_words):
        words = word_tokenize(text)  # Tokenizar el texto en palabras
        filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
        # Reconstrucción del texto
        return " ".join(filtered_words)

    @staticmethod
    def preprocessing(text, stop_words):
        cleaned_text = Model.remove_noise(text)  # Remover ruido del texto
        cleaned_words = Model.clean_text(cleaned_text, stop_words)  # Limpiar el texto
        return cleaned_words
    
    @staticmethod
    def process_text_stanza(text, nlp):
        doc = nlp(text)
        lemmatized_text = ' '.join([word.lemma for sent in doc.sentences for word in sent.words])
        return lemmatized_text

    def make_predictions(self, data):
        # Asegurarse de que 'data' es una lista, incluso si solo hay una entrada
        if isinstance(data, str):
            data = [data]

        cleaned_data = [self.preprocessing(text, self.stop_words) for text in data]
        processed_data = [self.process_text_stanza(text, self.nlp) for text in cleaned_data]

        proces = self.pipe.transform(processed_data)
        result = self.model.predict(proces)
        return result

