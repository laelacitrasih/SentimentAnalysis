from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

# === Model & Tokenizer ===
@torch.no_grad()
def load_model_and_tokenizer():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    label_dict = model.config.id2label
    return model, tokenizer, label_dict

# === Prediksi Sentimen Tunggal ===
@torch.no_grad()
def predict_sentiment(text, model, tokenizer, label_dict):
    encoded_input = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    encoded_input = {k: v.to("cpu") for k, v in encoded_input.items()}
    outputs = model(**encoded_input)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    label_raw = label_dict[pred_id]
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    return label_map.get(label_raw, "Unknown")

# === Load Dataset TweetEval ===
def load_tweet_dataset():
    return load_dataset("cardiffnlp/tweet_eval", "sentiment")

# === Mapping Label Angka → Nama Sentimen ===
def get_label_map():
    return {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
