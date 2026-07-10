import re
import joblib

MODEL_PATH = "spam_text_model.joblib"

_model_cache = None

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model(path: str = MODEL_PATH):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    data = joblib.load(path)
    _model_cache = data
    return data


def classify(text: str, path: str = MODEL_PATH):
    model_data = load_model(path)
    pipeline = model_data["pipeline"]
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    prob = pipeline.predict_proba([cleaned])[0][1]
    return bool(pred), float(prob)
