from flask import Flask, request, jsonify, make_response

import re
import nltk
import argparse
from torch import load
from waitress import serve
from torch.cuda import is_available
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

FOR_CONTAINER = True
model_path = './code/deployment/api/model_dir/model.pth'
if FOR_CONTAINER:
    model_path = './code/model_dir/model.pth'


def lower_text(text: str):
    return text.lower()

def remove_numbers(text: str):
    text_nonum = re.sub(r'\d+', ' ', text)
    return text_nonum

def remove_punctuation(text: str):
    pattern = r'[^\w\s]'
    text_nopunct = re.sub(pattern, ' ', text)
    return text_nopunct

def remove_multiple_spaces(text: str):
    pattern = r'\s{2,}'
    text_no_doublespace = re.sub(pattern, ' ', text)
    return text_no_doublespace

def tokenize_text(text: str) -> list[str]:
    return nltk.tokenize.word_tokenize(text)

def remove_stop_words(tokenized_text: list[str]) -> list[str]:
    stopwords = nltk.corpus.stopwords.words('english')
    filtered = [w for w in tokenized_text if not w in stopwords]
    return filtered

def stem_words(tokenized_text: list[str]) -> list[str]:
    stemmer = nltk.stem.PorterStemmer()
    stemmed = [stemmer.stem(w) for w in tokenized_text]
    return stemmed

def preprocessing_stage_text(text):
    _lowered = lower_text(text)
    _without_numbers = remove_numbers(_lowered)
    _without_punct = remove_punctuation(_without_numbers)
    _single_spaced = remove_multiple_spaces(_without_punct)
    _tokenized = tokenize_text(_single_spaced)
    _without_sw = remove_stop_words(_tokenized)
    _stemmed = stem_words(_without_sw)

    return _stemmed

device = "cuda" if is_available() else "cpu"
model = load(model_path)
model.eval()
app = Flask(__name__)

def model_predict(df, model):
    df['text'] = df['text'].apply(preprocessing_stage_text)
    return model(df['text'].to_numpy(), device)

@app.route("/info", methods = ["GET"])
def info():
	response = make_response("Model for predicting positiveness of tweet", 200)
	response.content_type = "text/plain"
	return response

@app.route("/predict", methods=["POST"])
def predict():
    data = None
    try:
        data = request.json
        if 'inputs' not in data:
            raise ValueError("Missing 'inputs' in request data")
        inputs = data['inputs']
        df = pd.DataFrame([inputs])
        prediction = model_predict(df, model)
        prediction_list = prediction.tolist()
        result = {
            'result': 'success',
            'prediction': prediction_list
        }
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app for model prediction")
    parser.add_argument('--port', type=int, default=5001, help="Port number for the Flask app")
    parser.add_argument('--host', type=str, default="localhost", help="Host IP address for the Flask app")
    
    args = parser.parse_args()
    serve(app, host=args.host, port=args.port)