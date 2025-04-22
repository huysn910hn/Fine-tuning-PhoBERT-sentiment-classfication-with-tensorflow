from fastapi import FastAPI
from tensorflow.keras.models import load_model
from transformers import TFAutoModel, AutoTokenizer
from keras.saving import register_keras_serializable
import tensorflow as tf
from pydantic import BaseModel
from tensorflow.keras.layers import Input
import re

@register_keras_serializable(package="CustomLayers")
class PhoBERT(tf.keras.layers.Layer):
    def __init__(self, model_name="vinai/phobert-base", **kwargs):
        super(PhoBERT, self).__init__(**kwargs)
        self.bert = TFAutoModel.from_pretrained(model_name)
        self.bert.trainable = False

    def call(self, inputs):
        input_ids, attention_mask = inputs
        input_ids = tf.cast(input_ids, tf.int32)
        attention_mask = tf.cast(attention_mask, tf.int32)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return output[:, 0, :]

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": "vinai/phobert-base"})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model_path = "sentiment_classification_model.keras"
model = load_model("sentiment_classification_model.keras")


class_names = ['0', '1']
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

app = FastAPI()

class TextInput(BaseModel):
    text: str

def prepare_data(input_text, tokenizer):
    tokenized_data = tokenizer(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='np'
    )
    return {
        'input_ids': tokenized_data['input_ids'],
        'attention_mask': tokenized_data['attention_mask']
    }
def preprocess_text(text):
    letters = set('aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvwxyýỳỷỹỵz0123456789. ')
    cleaned_text = ''.join(letter.lower() for letter in text if letter.lower() in letters)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

@app.post("/predict")
def predict(input: TextInput):
    input.text = preprocess_text(input.text)
    input_data = prepare_data(input.text, tokenizer)
    predictions = model.predict([input_data['input_ids'], input_data['attention_mask']])
    predicted_class = class_names[predictions.argmax()]
    return {"text": input.text, "predicted_class": predicted_class}
