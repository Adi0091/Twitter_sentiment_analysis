import os
import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

MODEL_DIR = "artifacts/transformer_model"

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
