from flask import Flask, request, jsonify, render_template
import spacy
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
model_path = "./Labtop - AI"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def extract_aspects(sentence):
    doc = nlp(sentence)
    aspects = [
        chunk.text for chunk in doc.noun_chunks
        if len(chunk.text.split()) > 1 and not chunk.root.pos_ in ['PRON']
    ]
    return aspects

SENTIMENT_LABELS = {
    0: "Conflict",
    1: "Negative",
    2: "Neutral",
    3: "Positive"
}

def predict_sentiment(sentence, aspect_term):
    inputs = tokenizer(
        text=sentence,
        text_pair=aspect_term,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_label_idx = int(torch.argmax(probs, dim=-1).item())
    return SENTIMENT_LABELS[predicted_label_idx]

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    sentence = data.get("sentence", "")
    aspects = extract_aspects(sentence)
    results = []
    for aspect in aspects:
        sentiment = predict_sentiment(sentence, aspect)
        results.append({"aspect": aspect, "sentiment": sentiment})
    return jsonify(results)

@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
