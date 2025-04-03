from fastapi import FastAPI
from tensorflow.keras.models import load_model
from transformers import TFAutoModel, AutoTokenizer
from keras.saving import register_keras_serializable
import tensorflow as tf
from pydantic import BaseModel
from tensorflow.keras.layers import Input

# 🔹 Định nghĩa lớp trước khi load mô hình
@register_keras_serializable()
class PhoBERTEmbedding(tf.keras.layers.Layer):
    def __init__(self, model_name="vinai/phobert-base", **kwargs):
        super(PhoBERTEmbedding, self).__init__(**kwargs)
        self.bert = TFAutoModel.from_pretrained(model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return output[:, 0, :]

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": "vinai/phobert-base"})
        return config

# 🔹 Load model sau khi đã định nghĩa PhoBERTEmbedding
model_path = "phobert_sentiment_model5.keras"
model = load_model(model_path, custom_objects={"PhoBERTEmbedding": PhoBERTEmbedding})


class_names = ['0', '1', '2']
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

@app.post("/predict")
def predict(input: TextInput):
    input_data = prepare_data(input.text, tokenizer)
    predictions = model.predict([input_data['input_ids'], input_data['attention_mask']])
    predicted_class = class_names[predictions.argmax()]
    return {"text": input.text, "predicted_class": predicted_class}
