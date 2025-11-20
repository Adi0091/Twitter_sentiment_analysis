import os
import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import gdown
import zipfile

MODEL_ZIP_URL = "https://drive.google.com/uc?export=download&id=1UKN34y0z36gLn2B52U8XXRxcwdNrDf1q"
MODEL_DIR = "artifacts/transformer_model"
ZIP_PATH = "transformer_model.zip"

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        return

    gdown.download(MODEL_ZIP_URL, ZIP_PATH, quiet=False)
    
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    os.remove(ZIP_PATH)

download_and_extract_model()

# MODEL_DIR = "artifacts/transformer_model"


tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

st.title("Twitter Sentiment Classifier")

text = st.text_area("Enter a tweet:")

if st.button("Predict"):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

    logits = model(inputs).logits
    pred = tf.argmax(logits, axis=1).numpy()[0]

    label = "Positive" if pred == 1 else "Negative"

    st.success(f"Prediction: {label}")

