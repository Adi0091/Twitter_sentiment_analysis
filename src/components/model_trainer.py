import os
import sys
import pandas as pd
import tensorflow as tf
import numpy as np

from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.config import (MAX_LEN, BATCH_SIZE, EPOCHS)

@dataclass
class ModelTrainerConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    model_dir: str = os.path.join("artifacts", "model")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self):
        logging.info("training starting")

        try:
            train = pd.read_csv(self.config.train_data_path)
            test = pd.read_csv(self.config.test_data_path)
            logging.info("read train and test data")

            train.dropna(inplace=True)
            test.dropna(inplace=True)

            X_train = train["stemmed_content"].tolist()[:60000]
            y_train = train["target"].tolist()[:60000]
            X_test = test["stemmed_content"].tolist()[:10000]
            y_test = test["target"].tolist()[:10000]

            logging.info("Prepared training and testing data")

            MODEL_NAME = "distilbert-base-uncased"
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

            train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LEN)
            test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LEN)

            logging.info("Tokenization completed")

            train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(10000).batch(BATCH_SIZE)

            test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(BATCH_SIZE)

            model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, from_pt=True)

            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

            logging.info("Starting model training...")

            model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS)

            logging.info("Model training completed.")

            y_pred_logits = model.predict(test_dataset).logits
            y_pred = np.argmax(y_pred_logits, axis=1)

            print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

            os.makedirs(self.config.model_dir, exist_ok=True)
            model.save_pretrained(self.config.model_dir)
            tokenizer.save_pretrained(self.config.model_dir)

            logging.info("Model and tokenizer saved successfully.")

            return self.config.model_dir
        
        
        except Exception as e:
            raise CustomException(e, sys)
            














































import os
import pandas as pd
import tensorflow as tf
import numpy as np

from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report

train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')

train.dropna(inplace=True)
test.dropna(inplace=True)

X_train, y_train = train['stemmed_content'].tolist()[:60000], train['target'].tolist()[:60000]
X_test, y_test   = test['stemmed_content'].tolist()[:10000], test['target'].tolist()[:10000]

MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings  = tokenizer(X_test, truncation=True, padding=True, max_length=128)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(10000).batch(8)
test_dataset  = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(8)

model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, from_pt=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_dataset, validation_data=test_dataset, epochs=2)

y_pred_logits = model.predict(test_dataset).logits
y_pred = np.argmax(y_pred_logits, axis=1)
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

os.makedirs('artifacts/', exist_ok=True)
model.save_pretrained('artifacts/')
tokenizer.save_pretrained('artifacts/')